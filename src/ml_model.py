"""
ml_model.py  —  Statistical anomaly scoring model.

In a production deployment this module would load a trained XGBoost or
Isolation Forest model from cloud storage (e.g. S3 / GCS).  For this
self-contained project we implement a calibrated scoring function that
reproduces the behaviour of a trained model without requiring external
training data or heavy ML dependencies.

The feature weights below were derived by reasoning about the relative
importance of each signal in real-world card-fraud datasets (e.g. the
Kaggle Credit Card Fraud Detection dataset where velocity, amount ratio,
and geographic anomalies are consistently the top predictors).
"""

import math
from typing import Dict


class FraudModel:
    """
    Computes a fraud probability in [0.0, 1.0] from a feature dictionary.

    Feature weights (sum to ~1.0 at max plausible values):
      - amount_ratio_to_avg : strongest single predictor
      - tx_count_last_hour  : card-testing / velocity attacks
      - country_mismatch    : geographic anomaly
      - unusual_category    : behavioural deviation
      - is_night            : night-time bias
      - high_spend_flag     : derived from spend_last_hour
    """

    def predict(self, features: Dict) -> float:
        score = 0.0

        # --- Amount ratio (sigmoid centred on 3× average) ----------------
        ar = features.get("amount_ratio_to_avg", 1.0)
        score += 0.35 * self._sigmoid(ar - 3.0, k=0.9)

        # --- Velocity (sigmoid centred on 8 tx/hour) ---------------------
        vel = features.get("tx_count_last_hour", 0)
        score += 0.25 * self._sigmoid(vel - 8.0, k=0.35)

        # --- Binary / categorical features -------------------------------
        score += 0.20 * features.get("country_mismatch",  0)
        score += 0.10 * features.get("unusual_category",  0)
        score += 0.05 * features.get("is_night",          0)

        # --- High hourly spend flag  (> €6 000) --------------------------
        spend_flag = 1 if features.get("spend_last_hour", 0) > 6_000 else 0
        score += 0.05 * spend_flag

        return round(min(max(score, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: float, k: float = 1.0) -> float:
        """Logistic function. k controls steepness."""
        try:
            return 1.0 / (1.0 + math.exp(-k * x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
