#!/usr/bin/env python3
"""
main.py  —  Fraud Detection System  |  entry point

Run:
    python main.py               # full demo (250 normal + fraud patterns)
    python main.py --quick       # 50 normal transactions only
    python main.py --no-save     # don't write output files
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.detector  import FraudDetector
from src.simulator import mixed_stream, generate_normal_stream
from src.reporter  import print_score, print_summary, save_json, save_csv


def parse_args():
    p = argparse.ArgumentParser(description="Fraud Detection System Demo")
    p.add_argument("--quick",   action="store_true", help="Run with 50 normal txs only")
    p.add_argument("--no-save", action="store_true", help="Skip writing output files")
    p.add_argument("--verbose", action="store_true", help="Print features for flagged txs")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 68)
    print("  FRAUD DETECTION SYSTEM  —  Real-time Transaction Scoring")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("═" * 68 + "\n")

    # ── Build transaction stream ──────────────────────────────────────────
    if args.quick:
        stream = generate_normal_stream(50)
    else:
        stream = mixed_stream(normal_count=250)

    print(f"  Processing {len(stream)} transactions ...\n")

    detector = FraudDetector()
    scores   = []
    silent_low = 0

    for tx in stream:
        result = detector.score(tx)
        scores.append(result)

        if result.risk_level == "LOW":
            silent_low += 1
        else:
            print_score(result, verbose=args.verbose)

    print(f"\n  (+ {silent_low} LOW-risk transactions approved silently)\n")

    # ── Summary ──────────────────────────────────────────────────────────
    print_summary(scores, detector.stats())

    # ── Highlight critical blocks ─────────────────────────────────────────
    critical = [s for s in scores if s.risk_level == "CRITICAL"]
    if critical:
        print(f"  🚨  {len(critical)} CRITICAL transaction(s) blocked:")
        for s in critical:
            print(f"      TX {s.transaction_id}  |  user={s.user_id}"
                  f"  |  score={s.score:.3f}"
                  f"  |  {', '.join(s.triggered_rules)}")
        print()

    # ── Save outputs ──────────────────────────────────────────────────────
    if not args.no_save:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.dirname(__file__), "outputs")
        save_json(scores, os.path.join(out, f"fraud_report_{ts}.json"))
        save_csv (scores, os.path.join(out, f"fraud_report_{ts}.csv"))
        print()


if __name__ == "__main__":
    main()
