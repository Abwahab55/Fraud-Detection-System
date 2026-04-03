"""
feature_store.py  —  In-memory per-user rolling feature store.

In production this would be backed by Redis with TTL keys.
Here we use Python deques so the project runs without external dependencies.
"""

from collections import defaultdict, deque
from datetime import datetime
from typing import Dict

import numpy as np

from src.models import Transaction


class FeatureStore:
    """
    Computes and caches per-user behavioural features from a sliding window
    of recent transactions.

    Parameters
    ----------
    window_seconds : int
        Length of the rolling window used for velocity calculations (default 1 h).
    """

    def __init__(self, window_seconds: int = 3600):
        self.window_seconds = window_seconds

        # raw deque per user: (unix_ts, amount, category, country)
        self._history: Dict[str, deque] = defaultdict(deque)

        # derived aggregates — updated on record()
        self._home_country:  Dict[str, str]   = {}
        self._running_avg:   Dict[str, float]  = {}
        self._category_freq: Dict[str, Dict]   = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self, tx: Transaction) -> Dict:
        """
        Return a feature dictionary for *tx* **without** mutating state.
        Call record() afterwards to commit the transaction to history.
        """
        uid = tx.user_id
        now = tx.timestamp.timestamp()

        # Prune stale entries before computing features
        self._prune(uid, now)

        window     = self._history[uid]
        tx_count   = len(window)
        hour_spend = sum(e[1] for e in window)

        # Amount ratio vs. historical average
        avg = self._running_avg.get(uid)
        if avg is None or avg == 0:
            # No history — bootstrap: anything > €200 is treated as elevated
            amount_ratio = max(tx.amount / 200.0, 1.0)
        else:
            amount_ratio = tx.amount / avg

        # Country mismatch
        home = self._home_country.get(uid)
        country_mismatch = 0 if (home is None or home == tx.country) else 1

        # Unusual merchant category
        freq = self._category_freq[uid]
        if freq:
            top_cat = max(freq, key=freq.get)
            unusual_category = 0 if tx.merchant_category == top_cat else 1
        else:
            unusual_category = 0

        hour = tx.timestamp.hour
        is_night = int(hour < 6 or hour >= 23)

        return {
            "amount":                tx.amount,
            "amount_ratio_to_avg":   round(amount_ratio, 3),
            "tx_count_last_hour":    tx_count,
            "spend_last_hour":       round(hour_spend, 2),
            "country_mismatch":      country_mismatch,
            "unusual_category":      unusual_category,
            "is_online":             int(tx.is_online),
            "hour_of_day":           hour,
            "is_night":              is_night,
        }

    def record(self, tx: Transaction):
        """Commit *tx* to history and update derived aggregates."""
        uid  = tx.user_id
        now  = tx.timestamp.timestamp()

        self._prune(uid, now)
        self._history[uid].append((now, tx.amount, tx.merchant_category, tx.country))

        # Set home country on first transaction
        if uid not in self._home_country:
            self._home_country[uid] = tx.country

        # Update running average
        amounts = [e[1] for e in self._history[uid]]
        self._running_avg[uid] = float(np.mean(amounts))

        # Update category frequency
        self._category_freq[uid][tx.merchant_category] += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, uid: str, now: float):
        """Remove entries older than the window from the user's deque."""
        dq = self._history[uid]
        while dq and (now - dq[0][0]) > self.window_seconds:
            dq.popleft()
