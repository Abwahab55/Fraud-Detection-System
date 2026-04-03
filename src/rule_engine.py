"""
rule_engine.py  —  Deterministic rule engine (always runs before the ML model).

Rules are cheap and interpretable — they catch obvious fraud immediately and
feed a rule_score into the final blended decision.
"""

from typing import Dict, List, Tuple

from src.models import Transaction


# ---------------------------------------------------------------------------
# Rule definitions
# Each rule is a (name, predicate) pair.
# Predicates receive (features_dict, transaction) and return bool.
# ---------------------------------------------------------------------------

_RULES = [
    # Amount-based
    ("LARGE_AMOUNT",        lambda f, tx: tx.amount > 5_000),
    ("VERY_LARGE_AMOUNT",   lambda f, tx: tx.amount > 15_000),
    ("AMOUNT_SPIKE_5X",     lambda f, tx: f["amount_ratio_to_avg"] >= 5.0),
    ("AMOUNT_SPIKE_10X",    lambda f, tx: f["amount_ratio_to_avg"] >= 10.0),

    # Velocity-based
    ("VELOCITY_HIGH",       lambda f, tx: f["tx_count_last_hour"] > 8),
    ("VELOCITY_EXTREME",    lambda f, tx: f["tx_count_last_hour"] > 15),
    ("HIGH_HOURLY_SPEND",   lambda f, tx: f["spend_last_hour"] > 8_000),

    # Geographic
    ("COUNTRY_MISMATCH",    lambda f, tx: f["country_mismatch"] == 1),

    # Behavioural
    ("UNUSUAL_CATEGORY",    lambda f, tx: f["unusual_category"] == 1),
    ("NIGHT_TRANSACTION",   lambda f, tx: f["is_night"] == 1),

    # Combined
    ("ONLINE_LARGE_AMOUNT", lambda f, tx: tx.is_online and tx.amount > 2_500),
]

# Contribution weight of each rule to the rule_score (capped at 0.85)
_WEIGHTS = {
    "LARGE_AMOUNT":        0.15,
    "VERY_LARGE_AMOUNT":   0.30,
    "AMOUNT_SPIKE_5X":     0.25,
    "AMOUNT_SPIKE_10X":    0.40,
    "VELOCITY_HIGH":       0.20,
    "VELOCITY_EXTREME":    0.40,
    "HIGH_HOURLY_SPEND":   0.25,
    "COUNTRY_MISMATCH":    0.25,
    "UNUSUAL_CATEGORY":    0.10,
    "NIGHT_TRANSACTION":   0.10,
    "ONLINE_LARGE_AMOUNT": 0.15,
}

_SCORE_CAP = 0.85


class RuleEngine:
    """Evaluates all rules and returns (triggered_rule_names, rule_score)."""

    def evaluate(self, features: Dict, tx: Transaction) -> Tuple[List[str], float]:
        triggered = []
        for name, predicate in _RULES:
            try:
                if predicate(features, tx):
                    triggered.append(name)
            except Exception:
                pass   # never crash the pipeline on a bad feature value

        raw_score = sum(_WEIGHTS.get(r, 0.0) for r in triggered)
        score = round(min(raw_score, _SCORE_CAP), 4)
        return triggered, score
