"""The frozen backbone and the correctness judge (Stage 4).

Two implementations behind one interface.

SimulatedReader is the default. It does not call a language model; it models
the answer-generation step as a probabilistic function of whether the prompt
contains the supporting evidence, with a penalty for irrelevant context. This
is not a shortcut for convenience — it is what makes the pipeline testable. The
causal structure OCMG depends on (evidence in prompt -> answer correct ->
credit to record) is preserved exactly, deterministically, and at zero cost, so
the governance mechanism can be validated end to end before a GPU is involved.
Absolute accuracy numbers from this reader are meaningless; the relative
ordering of conditions is what it is for.

HFReader is the real thing: an open-weight instruction model held frozen, with
greedy decoding at temperature zero. Swap it in for reported results.

Any number quoted as an experimental finding must come from HFReader on a real
benchmark. Numbers from SimulatedReader validate the machinery, nothing more.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .config import ReaderConfig
from .embedding import tokenize
from .types import Question, Record


# Phrases that mark an answer as a refusal to answer. Adversarial LoCoMo
# questions are judged correct only when the reader abstains.
_ABSTAIN_PATTERNS = re.compile(
    r"unable to determine|not mentioned|no information|not (?:stated|specified|provided)"
    r"|cannot be determined|can't be determined|do not know|don't know|no mention",
    re.I,
)


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_PATTERNS.search(text or ""))


def token_f1(prediction: str, reference: str) -> float:
    """Lexical overlap F1, reported alongside the judge as a cross-check."""
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return float(p == r)
    common = 0
    r_counts: dict[str, int] = {}
    for t in r:
        r_counts[t] = r_counts.get(t, 0) + 1
    for t in p:
        if r_counts.get(t, 0) > 0:
            common += 1
            r_counts[t] -= 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


class Reader(Protocol):
    def answer(
        self, question: Question, context: Sequence[Record]
    ) -> Tuple[str, int, float]:
        """Return (answer_text, generative_calls, p_correct)."""
        ...


def record_source_ids(rec: Record) -> set:
    """All source ids a record covers (falls back to the single fact id)."""
    ids = set(getattr(rec, "source_ids", None) or [])
    if rec.fact_id:
        ids.add(rec.fact_id)
    return ids


class SimulatedReader:
    """Deterministic, seeded model of the read step."""

    supports_counterfactual = True  # re-execution is cheap here

    def __init__(self, cfg: ReaderConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed

    def _draw(self, question: Question) -> float:
        """One fixed uniform draw per question (common random numbers).

        Leave-one-out credit compares the outcome on the full retrieved set
        against the outcome on the set minus one record. Drawing a fresh random
        number for each of those two calls makes the comparison a difference of
        two independent coin flips: the expectation is the quantity wanted, but
        the variance swamps it, and the labels that come out are noise. Holding
        the draw fixed per question turns the comparison into a deterministic
        function of the change in p_correct, which is the standard
        common-random-numbers variance reduction.

        The draw is derived from the question id rather than cached, so it does
        not depend on the order in which the reader is called. That matters
        because ablation loops call the reader a variable number of times per
        question, and an order-dependent draw would reintroduce the noise this
        removes.
        """
        key = f"{self.seed}:{question.question_id}".encode("utf8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, "big") / float(1 << 64)

    def p_correct(self, question: Question, context: Sequence[Record]) -> float:
        supporting = set(question.supporting_fact_ids)
        present = set()
        for r in context:
            present.update(record_source_ids(r))
        if supporting:
            hit_ratio = len(supporting & present) / len(supporting)
        else:
            # Adversarial / unanswerable questions: the correct behaviour is to
            # decline, and extra context makes spurious answering more likely.
            hit_ratio = 0.0
            base = self.cfg.p_correct_with_evidence
            penalty = min(self.cfg.max_penalty, self.cfg.distractor_penalty * len(context))
            return max(0.02, base - penalty)

        base = (
            self.cfg.p_correct_without_evidence
            + (self.cfg.p_correct_with_evidence - self.cfg.p_correct_without_evidence)
            * hit_ratio
        )
        n_distractors = sum(
            1 for r in context if not (record_source_ids(r) & supporting)
        )
        penalty = min(self.cfg.max_penalty, self.cfg.distractor_penalty * n_distractors)
        return float(np.clip(base - penalty, 0.01, 0.99))

    def answer(
        self, question: Question, context: Sequence[Record]
    ) -> Tuple[str, int, float]:
        p = self.p_correct(question, context)
        correct = self._draw(question) < p
        if correct:
            text = question.answer
        elif question.adversarial_answer:
            # Failing an adversarial question means taking the bait.
            text = question.adversarial_answer
        else:
            text = "unable to determine from memory"
        return text, 0, p

    def answer_many(self, items):
        """Same contract as the batched reader; nothing to batch here."""
        return [self.answer(q, c) for q, c in items]


class HFReader:
    """Open-weight backbone, frozen, greedy decoding."""

    supports_counterfactual = True  # correct but costly: |R| extra calls

    def __init__(self, cfg: ReaderConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self.torch = torch
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg.dtype != "auto":
            dtype = getattr(torch, cfg.dtype)
        else:
            # bfloat16 is the right default on a GPU, but on a CPU without AMX
            # or AVX512-BF16 every matmul is emulated and runs slower than
            # float32, so the dtype follows the device rather than the model.
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.tok = AutoTokenizer.from_pretrained(cfg.hf_model)
        # Decoder-only batching requires left padding: with right padding the
        # pad tokens sit between the prompt and the first generated token and
        # the model continues from padding instead of from the question.
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model, torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.cfg = cfg
        # Greedy decoding at temperature zero is a deterministic function of
        # the prompt, so a prompt seen twice can only produce the text it
        # produced the first time. Caching is therefore exactly equivalent to
        # regenerating, which makes it the only speed lever here that trades
        # nothing away. It pays because the condition arms put the same
        # questions to overlapping record sets, the leave-one-out batches
        # repeat the full-set prompt, and the full-context arm re-sends one
        # identical prompt for every question in a conversation.
        self._cache: dict = {}
        self.cache_hits = 0
        self.generated = 0

    def _prompt(self, question: Question, context: Sequence[Record]) -> str:
        blocks = [
            f"[{i + 1}] ({r.record_type}) {r.text}" for i, r in enumerate(context)
        ]
        joined = "\n".join(blocks) if blocks else "(no memory retrieved)"
        return (
            "You are answering from a memory store. Use only the records below.\n"
            "If the records do not contain the answer, reply exactly: "
            "unable to determine from memory.\n\n"
            f"MEMORY:\n{joined}\n\nQUESTION: {question.text}\nANSWER:"
        )

    def answer(
        self, question: Question, context: Sequence[Record]
    ) -> Tuple[str, int, float]:
        return self.answer_many([(question, context)])[0]

    def answer_many(self, items):
        """Greedy-decode a list of (question, context) pairs.

        Decoding is sequential per token, so the batch is the only place
        generation cost can be amortised. Greedy decoding makes the batched
        result identical to the one-at-a-time result up to floating-point
        accumulation order, which matters because the leave-one-out labels are
        differences between answers in the same batch.
        """
        return [(t, 0, float("nan")) for t in
                self.generate([self._prompt(q, c) for q, c in items])]

    def generate(self, prompts, max_new_tokens: int = 0):
        """Greedy-decode raw prompts in batches, reusing repeated prompts.

        Used by the reader and by the generative admission gate, which must
        share the loaded model rather than hold a second copy of it in memory.

        The token limit is part of the cache key: the same prompt asked for 4
        tokens and for 24 tokens are different requests, and the gate asks for
        4 while the reader asks for 24.
        """
        limit = max_new_tokens or self.cfg.max_new_tokens
        keys = [(limit, p) for p in prompts]

        # Deduplicate within the call as well as against earlier calls; a
        # leave-one-out batch over a set with repeated text would otherwise
        # pay twice inside one batch.
        pending: list = []
        queued: set = set()
        for key in keys:
            if key in self._cache or key in queued:
                continue
            queued.add(key)
            pending.append(key)
        self.cache_hits += len(keys) - len(pending)
        self.generated += len(pending)

        bs = max(1, int(self.cfg.batch_size))
        for start in range(0, len(pending), bs):
            chunk = pending[start:start + bs]
            inputs = self.tok(
                [p for _, p in chunk], return_tensors="pt",
                padding=True, truncation=False,
            ).to(self.device)
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=limit,
                    do_sample=False, temperature=None, top_p=None,
                    pad_token_id=self.tok.pad_token_id,
                )
            width = inputs["input_ids"].shape[1]
            for key, row in zip(chunk, out):
                self._cache[key] = self.tok.decode(
                    row[width:], skip_special_tokens=True
                ).strip()
        return [self._cache[key] for key in keys]


class GenerativeGate:
    """Per-candidate admission decision from one generative call.

    This is the cost baseline the whole efficiency argument is measured
    against, so it has to make real decisions: a gate that admits everything
    makes the ungoverned store look like a governed one and reports a price
    for nothing. Before 2026-10-30 that is exactly what the condition did -
    it ran an ungoverned build and only counted calls, producing a row
    identical to naive RAG to the last digit.

    Fails open. An unparseable reply admits the candidate, because a curator
    that drops records when it cannot decide would beat the gate on storage
    for the wrong reason. Unparsed replies are counted so the rate is visible.
    """

    PROMPT = (
        "You are curating a long-term memory store for an assistant.\n"
        "Keep the memory below if it records a specific fact, preference, "
        "event, place or date that a later question could ask about. Discard "
        "it if it is small talk or carries no reusable information.\n\n"
        "MEMORY: {text}\n\n"
        "Answer with one word, keep or discard.\nANSWER:"
    )

    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.unparsed = 0

    def __call__(self, cand) -> bool:
        self.calls += 1
        reply = self.model.generate(
            [self.PROMPT.format(text=cand.text)], max_new_tokens=4
        )[0].strip().lower()
        if reply.startswith("keep"):
            return True
        if reply.startswith("discard"):
            return False
        self.unparsed += 1
        return True


def build_generative_gate(reader):
    """A gate only where real generation exists; None otherwise.

    With the simulated reader there is no model to ask. Returning None makes
    the condition skip and say so, which is the honest alternative to either
    inventing a judge's accuracy or silently reporting a duplicate row.
    """
    return GenerativeGate(reader) if hasattr(reader, "generate") else None


class Judge:
    """Correctness assessment.

    'exact' is a cheap deterministic judge used with SimulatedReader. 'llm' is
    the protocol used for reported results, where a model scores the generated
    answer against the reference; it costs one generative call per judgement,
    which is why Section 9.3 budgets for it.
    """

    def __init__(self, kind: str = "exact", model=None):
        self.kind = kind
        self.model = model
        self.calls = 0

    def score(self, prediction: str, reference: str) -> Tuple[int, float]:
        f1 = token_f1(prediction, reference)
        if is_abstention(reference):
            # Unanswerable question: the only correct response is to abstain.
            ok = int(is_abstention(prediction))
            return ok, float(ok)
        if self.kind == "exact":
            correct = int(prediction.strip().lower() == reference.strip().lower())
            return correct, f1
        if self.kind == "f1":
            return int(f1 >= 0.6), f1
        if self.kind == "llm":
            self.calls += 1
            correct = int(self.model(prediction, reference))
            return correct, f1
        raise ValueError(f"unknown judge kind: {self.kind}")


def build_reader(cfg: ReaderConfig, seed: int = 0) -> Reader:
    if cfg.kind == "hf":
        return HFReader(cfg)
    return SimulatedReader(cfg, seed=seed)
