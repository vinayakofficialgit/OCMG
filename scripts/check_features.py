#!/usr/bin/env python3
"""Which features actually carry information, and which are dead weight.

Permutation importance says what the model used. It does not say why a
feature scored zero, and the two reasons are very different: a feature can be
present and uninformative, or it can be structurally constant and therefore
incapable of informing anything. The second is invisible in an importance
table - a constant column and a useless one both show 0.0000 - and it is the
one that points at a bug or a design limit.

This reports the training matrix column by column, and separately checks that
the categorical features are actually matching the values the dataset uses.
A feature that tests for a string the data never contains fails silently.

    python scripts/check_features.py
    python scripts/check_features.py --dataset synthetic

Exit status is 1 when any constant column is found, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocmg import console as C                      # noqa: E402
from ocmg.config import OCMGConfig                 # noqa: E402
from ocmg.data.benchmarks import load_dataset      # noqa: E402
from ocmg.experiment import split_conversations    # noqa: E402
from ocmg.features import FEATURE_NAMES            # noqa: E402
from ocmg.orchestrator import OCMGRunner           # noqa: E402

# Features whose value is fixed at admission time because the quantity they
# describe does not exist yet. Listed so the report can separate "structurally
# constant" from "unexpectedly constant", which is the distinction that
# matters when deciding whether a zero is a bug.
WRITE_TIME_UNAVAILABLE = {
    "retrieval_count", "log_retrieval_count", "recency_gap",
    "running_utility", "credit_total", "weight_total", "ever_retrieved",
}

# Categorical features and the exact value they test for, so a mismatch
# between the code's expectation and the dataset's vocabulary is visible.
CATEGORICAL_EXPECTATIONS = {
    "is_user": ("speaker", "user"),
    "is_assistant": ("speaker", "assistant"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="audit the feature matrix")
    ap.add_argument("--dataset", default="locomo")
    ap.add_argument("--path", default="data/locomo10.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    C.refresh_glyphs()
    if args.no_color:
        C.set_color(False)

    data = load_dataset(args.dataset, args.path, seed=args.seed)
    warm, _ = split_conversations(data, 0.5, args.seed)

    cfg = OCMGConfig(seed=args.seed)
    runner = OCMGRunner(cfg)
    out = runner.run_phase_a(warm)
    X = out.X

    print(C.section("1", "TRAINING MATRIX"))
    print(C.kv("dataset", f"{args.dataset}  seed {args.seed}"))
    print(C.kv("shape", f"{X.shape[0]:,} rows x {X.shape[1]} features"))
    print(C.kv("positive rate", f"{out.y.mean():.1%}"))

    sd = X.std(axis=0)
    constant = [FEATURE_NAMES[i] for i in range(X.shape[1]) if sd[i] == 0]
    expected = [n for n in constant if n in WRITE_TIME_UNAVAILABLE]
    unexpected = [n for n in constant if n not in WRITE_TIME_UNAVAILABLE]

    print("")
    print(C.section("2", "CONSTANT COLUMNS   a feature that never varies "
                         "cannot inform anything"))
    print(C.kv("constant", f"{len(constant)} of {X.shape[1]}"))

    if expected:
        print("")
        print(C.note(
            "Structurally constant, and not a bug: these describe a record's "
            "usage history, and write-time features are computed at admission "
            "when the record has never been retrieved and carries no credit. "
            "The outcome feedback trains the LABELS; it cannot appear in the "
            "FEATURES at the moment admission is decided. Worth stating "
            "plainly rather than leaving as zeros in an importance table.",
            "muted"))
        for n in expected:
            print(f"    {n}")

    if unexpected:
        print("")
        print(C.note(
            f"UNEXPECTEDLY CONSTANT - {len(unexpected)} feature(s) that should "
            "vary do not. This is the signature of a predicate that never "
            "matches: no error, no warning, just a column of zeros. Check "
            "section 3.", "critical", glyph=C.G.bad))
        for n in unexpected:
            print(f"    {n}")

    live = sorted(((float(sd[i]), FEATURE_NAMES[i])
                   for i in range(X.shape[1]) if sd[i] > 0), reverse=True)
    print("")
    print(C.kv("columns with variance", str(len(live))))
    print(C.table(["feature", "sd"],
                  [[n, f"{s:.4f}"] for s, n in live[:10]]))

    # ------------------------------------------------ categorical agreement
    print("")
    print(C.section("3", "CATEGORICAL FEATURES   does the predicate match "
                         "the data?"))
    seen = collections.Counter()
    for _, turns, _ in data:
        for t in turns:
            seen[t.speaker] += 1

    rows = []
    bad = False
    for feat, (field, wanted) in CATEGORICAL_EXPECTATIONS.items():
        n = seen.get(wanted, 0)
        if n == 0:
            bad = True
        rows.append([feat, f"{field} == {wanted!r}", f"{n:,}",
                     "ok" if n else "NEVER MATCHES"])
    print(C.table(["feature", "tests for", "matching turns", "status"], rows))

    print("")
    print(C.kv("distinct speaker values",
               ", ".join(f"{k!r}" for k, _ in seen.most_common(6))
               + (" ..." if len(seen) > 6 else "")))

    if bad:
        print("")
        print(C.note(
            "A categorical feature is testing for a value this dataset never "
            "uses, so it is zero on every record. Either map the dataset's "
            "vocabulary onto the expected values, or drop the feature and say "
            "why - but do not describe it in writing as something the model "
            "uses.", "critical", glyph=C.G.bad))

    return 1 if constant else 0


if __name__ == "__main__":
    raise SystemExit(main())
