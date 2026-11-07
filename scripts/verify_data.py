#!/usr/bin/env python3
"""Check the LoCoMo file loads correctly and print the dataset statistics.

    python scripts/verify_data.py
    python scripts/verify_data.py --path data/locomo10.json

Writes results/dataset_summary.json, which feeds the dataset description
table in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocmg.data.benchmarks import load_locomo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the LoCoMo dataset")
    ap.add_argument("--path", default=os.path.join("data", "locomo10.json"))
    ap.add_argument("--out", default=os.path.join("results", "dataset_summary.json"))
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"Dataset not found at {args.path}. See data/README.md.")
        return 1

    data, report = load_locomo(args.path, return_report=True)

    per_conv = []
    for cid, turns, questions in data:
        sessions = sorted({t.session_id for t in turns})
        first = datetime.fromtimestamp(min(t.timestamp for t in turns)).date()
        last = datetime.fromtimestamp(max(t.timestamp for t in turns)).date()
        speakers = sorted({t.speaker for t in turns})
        per_conv.append({
            "conversation": cid,
            "speakers": speakers,
            "sessions": len(sessions),
            "turns": len(turns),
            "words": sum(len(t.text.split()) for t in turns),
            "questions": len(questions),
            "first_session": str(first),
            "last_session": str(last),
        })

    ev_counts = {}
    for _, _, questions in data:
        for q in questions:
            ev_counts.setdefault(q.category, []).append(len(q.evidence_ids))

    print("LoCoMo dataset check")
    print("-" * 72)
    print(f"conversations : {report['conversations']}")
    print(f"sessions      : {report['sessions']}")
    print(f"turns         : {report['turns']}  (image turns with captions: {report['image_turns']})")
    print(f"questions     : {report['questions']}")
    print(f"evidence ids  : {report['evidence_ids']}  (dropped, not in release: "
          f"{len(report['evidence_unresolved'])})")
    print()
    print(f"{'category':<14}{'questions':>10}{'mean evidence':>15}{'multi-evidence':>16}")
    for cat in sorted(ev_counts):
        vals = ev_counts[cat]
        multi = sum(v > 1 for v in vals) / len(vals)
        print(f"{cat:<14}{len(vals):>10}{statistics.mean(vals):>15.2f}{multi:>15.0%}")
    print()
    print(f"{'conversation':<14}{'sessions':>9}{'turns':>7}{'words':>8}{'questions':>10}"
          f"  {'date range'}")
    for row in per_conv:
        print(f"{row['conversation']:<14}{row['sessions']:>9}{row['turns']:>7}"
              f"{row['words']:>8}{row['questions']:>10}  "
              f"{row['first_session']} to {row['last_session']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf8") as fh:
        json.dump({
            "source": args.path,
            "totals": {k: v for k, v in report.items() if k != "evidence_unresolved"},
            "evidence_unresolved": report["evidence_unresolved"],
            "per_category_evidence": {
                c: {"questions": len(v), "mean_evidence": statistics.mean(v),
                    "multi_evidence_share": sum(x > 1 for x in v) / len(v)}
                for c, v in ev_counts.items()
            },
            "per_conversation": per_conv,
        }, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
