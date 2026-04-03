"""
detector.py  —  FraudDetector: orchestrates features → rules → model → decision.
"""

import time
from typing import Dict

from src.models import Transaction, FraudScore
from src.feature_store import FeatureStore
from src.rule_engine import RuleEngine
from src.ml_model import FraudModel


# ---------------------------------------------------------------------------
# Risk bands and their corresponding decisions
# ---------------------------------------------------------------------------

_RISK_BANDS = [
    ("LOW",      0.00, 0.30, "APPROVE"),
    ("MEDIUM",   0.30, 0.55, "REVIEW"),
    ("HIGH",     0.55, 0.75, "REVIEW"),
    ("CRITICAL", 0.75, 1.01, "BLOCK"),
]


def _classify(score: float):
    for level, lo, hi, decision in _RISK_BANDS:
        if lo <= score < hi:
            return level, decision
    return "CRITICAL", "BLOCK"   # fallback


class FraudDetector:
    """
    End-to-end transaction fraud scorer.

    Usage
    -----
    detector = FraudDetector()
    result   = detector.score(transaction)

    The detector maintains per-user history internally.
    Thread-safety: not guaranteed; use one instance per thread in production.
    """

    # Blend weights: model contributes 60%, rules 40%
    _MODEL_WEIGHT = 0.60
    _RULE_WEIGHT  = 0.40

    def __init__(self):
        self.feature_store = FeatureStore()
        self.rule_engine   = RuleEngine()
        self.model         = FraudModel()

        # Counters for statistics
        self._total     = 0
        self._by_level: Dict[str, int] = {lvl: 0 for lvl, *_ in _RISK_BANDS}

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def score(self, tx: Transaction) -> FraudScore:
        """
        Score a single transaction.

        Steps
        -----
        1. Extract features from the feature store (read-only).
        2. Evaluate rule engine → rule_score.
        3. Run ML model         → model_score.
        4. Blend scores         → final_score.
        5. Classify into risk band and decision.
        6. Commit transaction to feature store history.
        7. Return FraudScore dataclass.
        """
        t_start = time.perf_counter()

        features              = self.feature_store.get_features(tx)
        triggered, rule_score = self.rule_engine.evaluate(features, tx)
        model_score           = self.model.predict(features)

        final_score = round(
            self._MODEL_WEIGHT * model_score + self._RULE_WEIGHT * rule_score, 4
        )

        risk_level, decision = _classify(final_score)

        # Commit to history AFTER scoring so history doesn't include current tx
        self.feature_store.record(tx)

        elapsed_ms = round((time.perf_counter() - t_start) * 1_000, 3)

        # Update stats
        self._total += 1
        self._by_level[risk_level] = self._by_level.get(risk_level, 0) + 1

        return FraudScore(
            transaction_id     = tx.transaction_id,
            user_id            = tx.user_id,
            score              = final_score,
            risk_level         = risk_level,
            decision           = decision,
            triggered_rules    = triggered,
            features           = features,
            model_score        = model_score,
            rule_score         = rule_score,
            processing_time_ms = elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """Return aggregate processing statistics."""
        flagged = self._by_level.get("HIGH", 0) + self._by_level.get("CRITICAL", 0)
        return {
            "total_processed": self._total,
            "by_risk_level":   dict(self._by_level),
            "flagged_count":   flagged,
            "flag_rate_pct":   round(100 * flagged / self._total, 2) if self._total else 0.0,
        }
