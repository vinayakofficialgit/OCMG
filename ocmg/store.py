"""Stage 3: two-tier typed storage and the utility ledger.

Tier 1 holds typed summaries plus pointers and is what retrieval scans. Tier 2
holds full records with provenance and is expanded only for survivors of the
first pass.

The ledger is SQLite rather than an in-memory dict for a substantive reason:
it requires transactional per-record updates that a vector index alone does not
support, and persisting it is what allows the outcome evidence to be released
as a dataset and re-fitted across sessions.

Eviction is non-destructive. A record is flagged inactive and its Tier 2 body
moved to a cold table, so that every removal decision stays auditable and
recoverable (Section 5).
"""

from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .features import IdfIndex
from .ingest import summarise, count_tokens
from .types import Candidate, LedgerRow, Record


class MemoryStore:
    def __init__(self, db_path: str = ":memory:", embedding_dim: int = 256):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=MEMORY")
        self._init_schema()
        self.embedding_dim = embedding_dim

        self.records: Dict[int, Record] = {}
        self.ledger: Dict[int, LedgerRow] = {}
        self.idf = IdfIndex()
        self._next_id = 1
        # Dense index over active records, partitioned by conversation and
        # rebuilt lazily. The partition is not an optimisation: a memory store
        # belongs to one dialogue, and letting a query reach another
        # conversation's records both leaks evidence and lets deduplication
        # merge unrelated users' turns together.
        # Dense index, two tiers, keyed by conversation id and also under
        # None for the cross-conversation view. Held as a preallocated buffer
        # plus a row count rather than one vstack per change, because
        # build_store extracts features for every candidate against the store
        # as it stands, so the index is read once per admission. Rebuilding it
        # on each admission made that O(n^2): measured at 10,206 full rebuilds
        # costing 27s of a 136s run. Appending into spare capacity with
        # amortised doubling makes it O(1) per record.
        self._hot_buf: Dict[Optional[str], np.ndarray] = {}
        self._hot_n: Dict[Optional[str], int] = {}
        self._hot_ids: Dict[Optional[str], List[int]] = {}
        self._cold_buf: Dict[Optional[str], np.ndarray] = {}
        self._cold_n: Dict[Optional[str], int] = {}
        self._cold_ids: Dict[Optional[str], List[int]] = {}
        self._dirty = True
        self.cold_store: Dict[int, Record] = {}

    # ------------------------------------------------------------------ schema

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS ledger (
                   record_id INTEGER PRIMARY KEY,
                   retrieval_count INTEGER NOT NULL DEFAULT 0,
                   credit_total REAL NOT NULL DEFAULT 0.0,
                   weight_total REAL NOT NULL DEFAULT 0.0,
                   last_retrieved_at REAL NOT NULL DEFAULT 0.0)"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS eviction_log (
                   record_id INTEGER, evicted_at REAL,
                   running_utility REAL, retrieval_count INTEGER, reason TEXT)"""
        )
        self.conn.commit()

    # ------------------------------------------------------------------- write

    def admit(
        self,
        cand: Candidate,
        write_features: Optional[np.ndarray] = None,
        hot: bool = True,
    ) -> int:
        """Write a candidate to the store.

        `hot=False` admits it to the cold tier instead of the hot one: the
        record is kept, indexed and retrievable, but it does not sit in the
        tier retrieval scans first. This is what the admission gate uses in
        place of discarding.

        The distinction matters because of what the estimator can support. A
        gate that discards makes an irreversible decision, so every error is
        permanent and caps the accuracy any later retrieval can reach. A gate
        that demotes makes a recoverable one, and a record wrongly demoted can
        still answer a question when it is clearly the best match available.
        """
        rid = self._next_id
        self._next_id += 1
        rec = Record(
            record_id=rid,
            conversation_id=cand.conversation_id,
            record_type=cand.record_type,
            summary=summarise(cand.text),
            text=cand.text,
            embedding=cand.embedding,
            entities=cand.entities,
            timestamp=cand.timestamp,
            session_id=cand.session_id,
            start_turn=cand.start_turn,
            end_turn=cand.end_turn,
            speaker=cand.speaker,
            has_temporal_anchor=cand.has_temporal_anchor,
            fact_id=cand.fact_id,
            write_features=write_features,
            source_ids=list(cand.source_ids),
        )
        self.records[rid] = rec
        self.ledger[rid] = LedgerRow(record_id=rid)
        self.conn.execute("INSERT OR IGNORE INTO ledger(record_id) VALUES (?)", (rid,))
        self.idf.add(cand.text)
        if not hot:
            rec.active = False
            rec.demoted_at_write = True
            self.cold_store[rid] = rec
            self.conn.execute(
                "INSERT INTO eviction_log VALUES (?,?,?,?,?)",
                (rid, cand.timestamp, 0.0, 0, "low_utility_admission"),
            )
        if self._dirty:
            # An invalidation is already pending; appending now would be
            # discarded by the rebuild and counted twice if it were not.
            return rid
        self._append(rec)
        return rid

    def merge_into(self, target_id: int, cand: Candidate) -> None:
        """Fold a near-duplicate into an existing record rather than storing it.

        The union of entities is kept and the longer text retained, so merging
        never loses information that retrieval could have used.
        """
        rec = self.records[target_id]
        rec.merged_count += 1
        rec.entities = sorted(set(rec.entities) | set(cand.entities))
        rec.source_ids = list(dict.fromkeys(list(rec.source_ids) + list(cand.source_ids)))
        if len(cand.text) > len(rec.text):
            rec.text = cand.text
            rec.summary = summarise(cand.text)
            rec.embedding = cand.embedding
            self._dirty = True
        rec.has_temporal_anchor = rec.has_temporal_anchor or cand.has_temporal_anchor

    # ------------------------------------------------------------------ evict

    def evict(self, record_id: int, now: float, reason: str = "utility") -> None:
        rec = self.records.get(record_id)
        if rec is None or not rec.active:
            return
        rec.active = False
        self.cold_store[record_id] = rec
        row = self.ledger[record_id]
        self.conn.execute(
            "INSERT INTO eviction_log VALUES (?,?,?,?,?)",
            (record_id, now, row.running_utility, row.retrieval_count, reason),
        )
        self._dirty = True

    # ----------------------------------------------------------------- ledger

    def update_ledger(self, record_id: int, credit: float, weight: float, now: float) -> None:
        row = self.ledger[record_id]
        row.retrieval_count += 1
        row.credit_total += credit
        row.weight_total += weight
        row.last_retrieved_at = now
        self.conn.execute(
            """UPDATE ledger SET retrieval_count=?, credit_total=?,
                   weight_total=?, last_retrieved_at=? WHERE record_id=?""",
            (row.retrieval_count, row.credit_total, row.weight_total, now, record_id),
        )

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ index

    def active_records(self, conv: Optional[str] = None) -> List[Record]:
        return [
            r
            for r in self.records.values()
            if r.active and (conv is None or r.conversation_id == conv)
        ]

    def cold_records(self, conv: Optional[str] = None) -> List[Record]:
        """The retrievable cold tier: records demoted at write time only.

        Records that were *evicted* also sit in `cold_store`, but they are
        deliberately excluded here. Eviction is non-destructive for audit and
        recovery, which is a different thing from remaining in service: a
        forgetting policy that still answers queries has not forgotten
        anything. Including them silently converted every eviction into a
        demotion — in the decay baseline as well as in OCMG — and a baseline
        that does not implement its own heuristic is not a baseline.
        """
        return [
            r
            for r in self.cold_store.values()
            if r.demoted_at_write and (conv is None or r.conversation_id == conv)
        ]

    def _append(self, rec: Record) -> None:
        """Add one record to the dense index in place.

        Used instead of invalidating the whole index, which is what made
        admission quadratic. Membership changes that are not appends - an
        eviction, a promotion, a merge that replaces an embedding - still mark
        the index dirty for a full rebuild, because they move or rewrite rows
        rather than adding one at the end. Those are rare next to admissions.
        """
        if rec.active:
            buf, cnt, ids = self._hot_buf, self._hot_n, self._hot_ids
        elif rec.demoted_at_write:
            buf, cnt, ids = self._cold_buf, self._cold_n, self._cold_ids
        else:
            return                     # evicted: out of service, kept for audit

        emb = np.asarray(rec.embedding)
        for key in (rec.conversation_id, None):
            n = cnt.get(key, 0)
            arr = buf.get(key)
            if arr is None:
                # dtype follows the data rather than being pinned to float32,
                # so a different embedder cannot be silently downcast.
                arr = np.zeros((8, self.embedding_dim), dtype=emb.dtype)
                buf[key] = arr
            elif n == arr.shape[0]:
                bigger = np.zeros((n * 2, self.embedding_dim), dtype=arr.dtype)
                bigger[:n] = arr
                buf[key] = arr = bigger
            arr[n] = emb
            cnt[key] = n + 1
            ids.setdefault(key, []).append(rec.record_id)

    def _rebuild(self) -> None:
        self._hot_buf, self._hot_n, self._hot_ids = {}, {}, {}
        self._cold_buf, self._cold_n, self._cold_ids = {}, {}, {}
        # Insertion order, so a rebuilt index has the same row order as one
        # grown by appends and the two are interchangeable.
        for r in self.records.values():
            self._append(r)
        self._dirty = False

    def _view(self, buf, cnt, ids, conv):
        """Rows actually in use. A view, not a copy: callers read it."""
        n = cnt.get(conv, 0)
        if not n:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        return buf[conv][:n], ids.get(conv, [])

    def matrix(self, conv: Optional[str] = None) -> tuple[np.ndarray, List[int]]:
        """Dense index over active records in one conversation (Tier 1)."""
        if self._dirty:
            self._rebuild()
        return self._view(self._hot_buf, self._hot_n, self._hot_ids, conv)

    def matrix_cold(self, conv: Optional[str] = None) -> tuple[np.ndarray, List[int]]:
        """Dense index over the cold tier: demoted and evicted records.

        Kept separate from the hot index rather than merged, so that retrieval
        can price the two tiers differently instead of treating a demoted
        record as though it had never been judged.
        """
        if self._dirty:
            self._rebuild()
        return self._view(self._cold_buf, self._cold_n, self._cold_ids, conv)

    def promote(self, record_id: int, now: float) -> bool:
        """Return a cold record to the hot tier, auditably."""
        rec = self.records.get(record_id)
        if rec is None or rec.active:
            return False
        rec.active = True
        self.cold_store.pop(record_id, None)
        self.conn.execute(
            "INSERT INTO eviction_log VALUES (?,?,?,?,?)",
            (record_id, now, self.ledger[record_id].running_utility,
             self.ledger[record_id].retrieval_count, "promoted"),
        )
        self._dirty = True
        return True

    def nearest(
        self, embedding: np.ndarray, k: int = 5, conv: Optional[str] = None
    ) -> List[tuple[int, float]]:
        from .embedding import cosine_matrix

        mat, ids = self.matrix(conv)
        if not ids:
            return []
        sims = cosine_matrix(embedding, mat)
        order = np.argsort(sims)[::-1][:k]
        return [(ids[i], float(sims[i])) for i in order]

    # ------------------------------------------------------------------ stats

    def token_footprint(self) -> int:
        return sum(count_tokens(r.text) for r in self.active_records())

    def stats(self) -> Dict[str, float]:
        active = self.active_records()
        by_type: Dict[str, int] = {}
        for r in active:
            by_type[r.record_type] = by_type.get(r.record_type, 0) + 1
        return {
            "n_active": len(active),
            "n_evicted": len(self.cold_store),
            "n_total_written": len(self.records),
            "tokens": self.token_footprint(),
            **{f"n_{k}": v for k, v in by_type.items()},
        }

    def export_ledger(self) -> List[dict]:
        """The reusable outcome dataset described in Section 5."""
        out = []
        for rid, row in self.ledger.items():
            rec = self.records[rid]
            out.append(
                {
                    "record_id": rid,
                    "conversation_id": rec.conversation_id,
                    "record_type": rec.record_type,
                    "active": rec.active,
                    "retrieval_count": row.retrieval_count,
                    "credit_total": row.credit_total,
                    "weight_total": row.weight_total,
                    "running_utility": row.running_utility,
                    "write_features": None
                    if rec.write_features is None
                    else rec.write_features.tolist(),
                }
            )
        return out
