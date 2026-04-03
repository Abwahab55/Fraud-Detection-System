"""
reporter.py  —  Console output, JSON export, and CSV export.
"""

import csv
import json
import os
from typing import Dict, List

from src.models import FraudScore


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_C = {
    "LOW":      "\033[92m",
    "MEDIUM":   "\033[93m",
    "HIGH":     "\033[91m",
    "CRITICAL": "\033[95m",
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
    "DIM":      "\033[2m",
}

_DECISION_ICON = {"APPROVE": "✓", "REVIEW": "⚠", "BLOCK": "✗"}


def _bar(score: float, width: int = 22) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Per-transaction output
# ---------------------------------------------------------------------------

def print_score(fs: FraudScore, verbose: bool = False):
    c  = _C.get(fs.risk_level, "")
    rs = _C["RESET"]
    icon = _DECISION_ICON.get(fs.decision, "?")

    print(
        f"  {_C['DIM']}TX {fs.transaction_id:<12}{rs}"
        f"  user={fs.user_id:<12}"
        f"  score={fs.score:.3f} {_bar(fs.score)}"
        f"  {c}[{fs.risk_level:<8}]{rs}"
        f"  {icon} {fs.decision}"
        f"  ({fs.processing_time_ms:.1f} ms)"
    )
    if fs.triggered_rules:
        print(f"    {_C['DIM']}rules : {', '.join(fs.triggered_rules)}{rs}")
    if verbose:
        f = fs.features
        print(
            f"    {_C['DIM']}feats : "
            f"ratio={f.get('amount_ratio_to_avg'):.2f}  "
            f"vel={f.get('tx_count_last_hour')}  "
            f"country_mis={f.get('country_mismatch')}  "
            f"night={f.get('is_night')}{rs}"
        )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(scores: List[FraudScore], stats: Dict):
    total  = len(scores)
    if total == 0:
        return

    by_lvl  = stats.get("by_risk_level", {})
    blocked = sum(1 for s in scores if s.decision == "BLOCK")
    review  = sum(1 for s in scores if s.decision == "REVIEW")
    approve = total - blocked - review
    avg_ms  = sum(s.processing_time_ms for s in scores) / total

    print(f"\n{'═' * 68}")
    print(f"  {_C['BOLD']}FRAUD DETECTION — SUMMARY REPORT{_C['RESET']}")
    print(f"{'═' * 68}")
    print(f"  Transactions processed  : {total}")
    print(f"  Average latency         : {avg_ms:.2f} ms / transaction")
    print()

    for lvl in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        count = by_lvl.get(lvl, 0)
        pct   = 100 * count / total
        bar   = "█" * int(pct / 4)
        c, rs = _C[lvl], _C["RESET"]
        print(f"  {c}{lvl:<10}{rs}  {count:>4}  ({pct:5.1f}%)  {bar}")

    print()
    print(f"  ✓  APPROVE  {approve:>4}")
    print(f"  ⚠  REVIEW   {review:>4}")
    print(f"  ✗  BLOCK    {blocked:>4}")
    print(f"{'═' * 68}\n")


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def save_json(scores: List[FraudScore], path: str):
    _ensure_dir(path)
    records = [
        {
            "transaction_id":     s.transaction_id,
            "user_id":            s.user_id,
            "score":              s.score,
            "risk_level":         s.risk_level,
            "decision":           s.decision,
            "triggered_rules":    s.triggered_rules,
            "model_score":        s.model_score,
            "rule_score":         s.rule_score,
            "features":           s.features,
            "processing_time_ms": s.processing_time_ms,
        }
        for s in scores
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, default=str)
    print(f"  JSON saved → {path}  ({len(scores)} records)")


def save_csv(scores: List[FraudScore], path: str):
    _ensure_dir(path)
    fields = [
        "transaction_id", "user_id", "score", "risk_level",
        "decision", "triggered_rules", "model_score", "rule_score",
        "processing_time_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in scores:
            writer.writerow({
                "transaction_id":     s.transaction_id,
                "user_id":            s.user_id,
                "score":              s.score,
                "risk_level":         s.risk_level,
                "decision":           s.decision,
                "triggered_rules":    "|".join(s.triggered_rules),
                "model_score":        s.model_score,
                "rule_score":         s.rule_score,
                "processing_time_ms": s.processing_time_ms,
            })
    print(f"  CSV saved  → {path}  ({len(scores)} records)")
