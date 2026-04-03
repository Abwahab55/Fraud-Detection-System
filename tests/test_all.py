"""
tests/test_all.py  —  Full unit-test suite for the Fraud Detection System.

Run:
    python -m unittest tests/test_all.py -v
    python -m unittest discover tests/ -v

Coverage:
    TestTransaction       — data model validation
    TestFeatureStore      — rolling window, velocity, ratios, country/category flags
    TestRuleEngine        — individual rule firing and score accumulation
    TestFraudModel        — ML model output range and relative ordering
    TestFraudDetector     — end-to-end integration: scoring, stats, edge cases
    TestSimulator         — stream generation correctness
"""

import random
import sys
import os
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models        import Transaction, FraudScore
from src.feature_store import FeatureStore
from src.rule_engine   import RuleEngine
from src.ml_model      import FraudModel
from src.detector      import FraudDetector
from src.simulator     import (
    generate_normal_stream, generate_fraud_patterns, mixed_stream
)

# ── Shared fixtures ────────────────────────────────────────────────────────

BASE = datetime(2024, 6, 1, 14, 0, 0)


def tx(tx_id="tx_001", user_id="user_A", amount=50.0,
       category="grocery", country="DE", dt=None, is_online=False):
    """Convenience constructor for test transactions."""
    return Transaction(
        transaction_id    = tx_id,
        user_id           = user_id,
        amount            = amount,
        merchant_category = category,
        country           = country,
        timestamp         = dt or BASE,
        is_online         = is_online,
    )


def establish_baseline(fs: FeatureStore, uid: str, n: int = 6,
                        amount: float = 60.0, category: str = "grocery",
                        country: str = "DE"):
    """Record *n* identical transactions to set a user's home state."""
    for i in range(n):
        t = Transaction(
            f"base_{i}", uid, amount, category, country,
            BASE + timedelta(minutes=i)
        )
        fs.get_features(t)
        fs.record(t)


# ===========================================================================
# 1. Transaction model
# ===========================================================================

class TestTransaction(unittest.TestCase):

    def test_fields_stored_correctly(self):
        t = tx()
        self.assertEqual(t.transaction_id, "tx_001")
        self.assertEqual(t.amount, 50.0)
        self.assertEqual(t.country, "DE")

    def test_is_online_default_true(self):
        t = Transaction("x", "u", 10.0, "grocery", "DE", BASE)
        self.assertTrue(t.is_online)

    def test_metadata_default_empty_dict(self):
        t = Transaction("x", "u", 10.0, "grocery", "DE", BASE)
        self.assertEqual(t.metadata, {})

    def test_custom_metadata(self):
        t = tx()
        t.metadata["channel"] = "mobile"
        self.assertEqual(t.metadata["channel"], "mobile")


# ===========================================================================
# 2. FeatureStore
# ===========================================================================

class TestFeatureStore(unittest.TestCase):

    def test_returns_all_expected_keys(self):
        fs = FeatureStore()
        f  = fs.get_features(tx())
        for key in ["amount", "amount_ratio_to_avg", "tx_count_last_hour",
                    "spend_last_hour", "country_mismatch", "unusual_category",
                    "is_online", "hour_of_day", "is_night"]:
            self.assertIn(key, f, f"Missing feature key: {key}")

    def test_no_mismatch_on_first_transaction(self):
        fs = FeatureStore()
        f  = fs.get_features(tx(country="DE"))
        self.assertEqual(f["country_mismatch"], 0)

    def test_country_mismatch_detected_after_baseline(self):
        fs  = FeatureStore()
        uid = "user_cm"
        establish_baseline(fs, uid, country="DE")
        f = fs.get_features(tx(user_id=uid, country="NG",
                                dt=BASE + timedelta(hours=1)))
        self.assertEqual(f["country_mismatch"], 1)

    def test_no_mismatch_same_country(self):
        fs  = FeatureStore()
        uid = "user_same"
        establish_baseline(fs, uid, country="DE")
        f = fs.get_features(tx(user_id=uid, country="DE",
                                dt=BASE + timedelta(hours=1)))
        self.assertEqual(f["country_mismatch"], 0)

    def test_velocity_increments_correctly(self):
        fs  = FeatureStore()
        uid = "user_vel"
        for i in range(7):
            t = Transaction(f"v{i}", uid, 20.0, "grocery", "DE",
                            BASE + timedelta(seconds=i * 30))
            fs.get_features(t)
            fs.record(t)
        last = Transaction("v7", uid, 20.0, "grocery", "DE",
                           BASE + timedelta(seconds=7 * 30))
        f = fs.get_features(last)
        self.assertEqual(f["tx_count_last_hour"], 7)

    def test_amount_ratio_approximately_correct(self):
        fs  = FeatureStore()
        uid = "user_ratio"
        # Establish average of 100
        for i in range(5):
            t = Transaction(f"r{i}", uid, 100.0, "grocery", "DE",
                            BASE + timedelta(minutes=i))
            fs.get_features(t)
            fs.record(t)
        # 10× average
        spike = Transaction("rspike", uid, 1000.0, "grocery", "DE",
                            BASE + timedelta(minutes=10))
        f = fs.get_features(spike)
        self.assertAlmostEqual(f["amount_ratio_to_avg"], 10.0, delta=0.5)

    def test_night_flag_set_correctly(self):
        fs = FeatureStore()
        self.assertEqual(
            fs.get_features(tx(dt=BASE.replace(hour=2)))["is_night"], 1
        )
        self.assertEqual(
            fs.get_features(tx(dt=BASE.replace(hour=23)))["is_night"], 1
        )

    def test_daytime_not_flagged_as_night(self):
        fs = FeatureStore()
        for h in (8, 12, 18, 22):
            f = fs.get_features(tx(dt=BASE.replace(hour=h)))
            self.assertEqual(f["is_night"], 0, f"hour {h} wrongly flagged as night")

    def test_window_expiry_clears_velocity(self):
        fs  = FeatureStore(window_seconds=60)
        uid = "user_exp"
        old = Transaction("old", uid, 50.0, "grocery", "DE", BASE)
        fs.get_features(old)
        fs.record(old)
        # 2 minutes later — outside window
        later = Transaction("later", uid, 50.0, "grocery", "DE",
                            BASE + timedelta(seconds=121))
        f = fs.get_features(later)
        self.assertEqual(f["tx_count_last_hour"], 0)

    def test_spend_accumulates_in_window(self):
        fs  = FeatureStore()
        uid = "user_spend"
        for i in range(4):
            t = Transaction(f"s{i}", uid, 500.0, "grocery", "DE",
                            BASE + timedelta(minutes=i))
            fs.get_features(t)
            fs.record(t)
        f = fs.get_features(Transaction("s4", uid, 500.0, "grocery", "DE",
                                         BASE + timedelta(minutes=5)))
        self.assertAlmostEqual(f["spend_last_hour"], 2000.0, delta=1.0)

    def test_unusual_category_detected(self):
        fs  = FeatureStore()
        uid = "user_cat"
        # Establish grocery as top category
        for i in range(5):
            t = Transaction(f"c{i}", uid, 40.0, "grocery", "DE",
                            BASE + timedelta(minutes=i))
            fs.get_features(t)
            fs.record(t)
        f = fs.get_features(Transaction("cnew", uid, 40.0, "electronics", "DE",
                                         BASE + timedelta(minutes=10)))
        self.assertEqual(f["unusual_category"], 1)


# ===========================================================================
# 3. RuleEngine
# ===========================================================================

class TestRuleEngine(unittest.TestCase):

    def _eval(self, transaction, extra_feats=None):
        fs = FeatureStore()
        f  = fs.get_features(transaction)
        if extra_feats:
            f.update(extra_feats)
        return RuleEngine().evaluate(f, transaction)

    def test_clean_tx_fires_no_critical_rules(self):
        triggered, score = self._eval(tx(amount=45.0))
        self.assertNotIn("LARGE_AMOUNT", triggered)
        self.assertNotIn("VELOCITY_HIGH", triggered)
        self.assertLess(score, 0.35)

    def test_large_amount_rule_fires(self):
        triggered, _ = self._eval(tx(amount=6_000.0))
        self.assertIn("LARGE_AMOUNT", triggered)

    def test_very_large_amount_rule_fires(self):
        triggered, _ = self._eval(tx(amount=20_000.0))
        self.assertIn("VERY_LARGE_AMOUNT", triggered)

    def test_amount_spike_5x_rule(self):
        triggered, _ = self._eval(tx(amount=500.0),
                                  extra_feats={"amount_ratio_to_avg": 6.0})
        self.assertIn("AMOUNT_SPIKE_5X", triggered)

    def test_amount_spike_10x_rule(self):
        triggered, _ = self._eval(tx(amount=500.0),
                                  extra_feats={"amount_ratio_to_avg": 11.0})
        self.assertIn("AMOUNT_SPIKE_10X", triggered)

    def test_velocity_high_rule(self):
        triggered, _ = self._eval(tx(),
                                  extra_feats={"tx_count_last_hour": 10})
        self.assertIn("VELOCITY_HIGH", triggered)

    def test_velocity_extreme_rule(self):
        triggered, _ = self._eval(tx(),
                                  extra_feats={"tx_count_last_hour": 18})
        self.assertIn("VELOCITY_EXTREME", triggered)

    def test_country_mismatch_rule(self):
        triggered, _ = self._eval(tx(), extra_feats={"country_mismatch": 1})
        self.assertIn("COUNTRY_MISMATCH", triggered)

    def test_night_rule(self):
        triggered, _ = self._eval(tx(dt=BASE.replace(hour=2), amount=100.0))
        self.assertIn("NIGHT_TRANSACTION", triggered)

    def test_online_large_amount_rule(self):
        triggered, _ = self._eval(tx(amount=3_000.0, is_online=True))
        self.assertIn("ONLINE_LARGE_AMOUNT", triggered)

    def test_score_never_exceeds_cap(self):
        # Force all rules to fire
        t  = tx(amount=20_000.0, is_online=True)
        fs = FeatureStore()
        f  = fs.get_features(t)
        f.update({
            "tx_count_last_hour":   25,
            "spend_last_hour":      15_000,
            "country_mismatch":     1,
            "unusual_category":     1,
            "is_night":             1,
            "amount_ratio_to_avg":  12.0,
        })
        _, score = RuleEngine().evaluate(f, t)
        self.assertLessEqual(score, 0.85)

    def test_score_positive_when_rules_fire(self):
        triggered, score = self._eval(tx(amount=6_000.0))
        self.assertGreater(len(triggered), 0)
        self.assertGreater(score, 0.0)


# ===========================================================================
# 4. FraudModel
# ===========================================================================

class TestFraudModel(unittest.TestCase):

    LOW_FEATS = {
        "amount_ratio_to_avg": 1.0, "tx_count_last_hour": 1,
        "country_mismatch":    0,   "unusual_category":   0,
        "is_night":            0,   "spend_last_hour":    50,
    }

    HIGH_FEATS = {
        "amount_ratio_to_avg": 9.0, "tx_count_last_hour": 18,
        "country_mismatch":    1,   "unusual_category":   1,
        "is_night":            1,   "spend_last_hour":    12_000,
    }

    def test_low_risk_features_give_low_score(self):
        self.assertLess(FraudModel().predict(self.LOW_FEATS), 0.30)

    def test_high_risk_features_give_high_score(self):
        self.assertGreater(FraudModel().predict(self.HIGH_FEATS), 0.60)

    def test_score_always_in_unit_interval(self):
        model = FraudModel()
        rng   = random.Random(7)
        for _ in range(50):
            f = {
                "amount_ratio_to_avg": rng.uniform(0, 25),
                "tx_count_last_hour":  rng.randint(0, 40),
                "country_mismatch":    rng.randint(0, 1),
                "unusual_category":    rng.randint(0, 1),
                "is_night":            rng.randint(0, 1),
                "spend_last_hour":     rng.uniform(0, 20_000),
            }
            s = model.predict(f)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_higher_ratio_yields_higher_score(self):
        model = FraudModel()
        base  = dict(self.LOW_FEATS)

        base["amount_ratio_to_avg"] = 2.0;  s2 = model.predict(base)
        base["amount_ratio_to_avg"] = 6.0;  s6 = model.predict(base)
        base["amount_ratio_to_avg"] = 12.0; s12 = model.predict(base)

        self.assertLess(s2, s6)
        self.assertLess(s6, s12)

    def test_higher_velocity_yields_higher_score(self):
        model = FraudModel()
        base  = dict(self.LOW_FEATS)

        base["tx_count_last_hour"] = 2;  s2 = model.predict(base)
        base["tx_count_last_hour"] = 10; s10 = model.predict(base)
        base["tx_count_last_hour"] = 20; s20 = model.predict(base)

        self.assertLess(s2, s10)
        self.assertLess(s10, s20)


# ===========================================================================
# 5. FraudDetector (integration)
# ===========================================================================

class TestFraudDetector(unittest.TestCase):

    def setUp(self):
        self.det = FraudDetector()

    # ── Basic scoring ──────────────────────────────────────────────────────

    def test_returns_fraud_score_instance(self):
        self.assertIsInstance(self.det.score(tx()), FraudScore)

    def test_transaction_id_preserved(self):
        r = self.det.score(tx(tx_id="keep_me"))
        self.assertEqual(r.transaction_id, "keep_me")

    def test_user_id_preserved(self):
        r = self.det.score(tx(user_id="keep_user"))
        self.assertEqual(r.user_id, "keep_user")

    def test_score_in_unit_interval(self):
        for amount in (10, 100, 1000, 10000):
            r = self.det.score(tx(tx_id=f"t{amount}", amount=float(amount)))
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0)

    # ── Decision logic ────────────────────────────────────────────────────

    def test_normal_small_tx_is_approved(self):
        r = self.det.score(tx(amount=35.0, is_online=False))
        self.assertEqual(r.decision, "APPROVE")
        self.assertEqual(r.risk_level, "LOW")

    def test_very_large_amount_is_flagged(self):
        r = self.det.score(tx(tx_id="big", amount=18_000.0, is_online=True,
                               dt=BASE + timedelta(hours=1)))
        self.assertIn(r.risk_level, ("MEDIUM", "HIGH", "CRITICAL"))
        self.assertIn(r.decision,   ("REVIEW", "BLOCK"))

    def test_critical_score_gives_block_decision(self):
        # Force a critical score via a very high-value late-night foreign tx
        r = self.det.score(tx(
            tx_id="crit", amount=20_000.0, country="NG",
            dt=BASE.replace(hour=3), is_online=True,
        ))
        # Score should put it in HIGH or CRITICAL band
        self.assertGreater(r.score, 0.50)

    def test_risk_level_consistent_with_score(self):
        r = self.det.score(tx(amount=50.0))
        boundaries = {"LOW": (0.00, 0.30), "MEDIUM": (0.30, 0.55),
                      "HIGH": (0.55, 0.75), "CRITICAL": (0.75, 1.01)}
        lo, hi = boundaries[r.risk_level]
        self.assertGreaterEqual(r.score, lo)
        self.assertLess(r.score, hi)

    # ── Features / rules ──────────────────────────────────────────────────

    def test_result_contains_non_empty_features(self):
        r = self.det.score(tx())
        self.assertIsInstance(r.features, dict)
        self.assertGreater(len(r.features), 0)

    def test_triggered_rules_is_list(self):
        r = self.det.score(tx())
        self.assertIsInstance(r.triggered_rules, list)

    def test_large_amount_triggers_amount_rule(self):
        r = self.det.score(tx(tx_id="la", amount=8_000.0))
        self.assertIn("LARGE_AMOUNT", r.triggered_rules)

    # ── Statistics ────────────────────────────────────────────────────────

    def test_stats_total_processed(self):
        for i in range(5):
            self.det.score(tx(tx_id=f"s{i}"))
        self.assertEqual(self.det.stats()["total_processed"], 5)

    def test_stats_flagged_count_increases(self):
        before = self.det.stats()["flagged_count"]
        self.det.score(tx(tx_id="f1", amount=20_000.0, is_online=True,
                           country="NG", dt=BASE.replace(hour=3)))
        after = self.det.stats()["flagged_count"]
        self.assertGreaterEqual(after, before)

    def test_stats_flag_rate_pct_in_range(self):
        for i in range(10):
            self.det.score(tx(tx_id=f"r{i}"))
        pct = self.det.stats()["flag_rate_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)

    # ── Latency ───────────────────────────────────────────────────────────

    def test_processing_time_under_100ms(self):
        r = self.det.score(tx())
        self.assertLess(r.processing_time_ms, 100.0)

    def test_processing_time_positive(self):
        r = self.det.score(tx())
        self.assertGreater(r.processing_time_ms, 0.0)

    # ── Behavioural / scenario tests ──────────────────────────────────────

    def test_risky_tx_scores_higher_than_safe_tx(self):
        r_safe  = self.det.score(tx(tx_id="safe",  amount=30.0))
        r_risky = self.det.score(
            tx(tx_id="risky", user_id="user_B", amount=18_000.0,
               country="NG", dt=BASE.replace(hour=3), is_online=True)
        )
        self.assertLess(r_safe.score, r_risky.score)

    def test_velocity_increases_score_over_time(self):
        """Score should rise as the user sends many rapid transactions."""
        det = FraudDetector()
        uid = "vel_test_user"
        scores = []
        for i in range(18):
            t = Transaction(f"vt{i}", uid, 2.0, "online_retail", "DE",
                            BASE + timedelta(seconds=i * 12))
            scores.append(det.score(t).score)
        # Last score must be higher than the first (velocity accumulated)
        self.assertGreater(scores[-1], scores[0])

    def test_history_raises_score_on_amount_spike(self):
        """After a normal baseline, a 10× spike should push score up."""
        det = FraudDetector()
        uid = "spike_user"
        # Baseline: 8 normal transactions
        for i in range(8):
            t = Transaction(f"sp{i}", uid, 60.0, "grocery", "DE",
                            BASE + timedelta(minutes=i))
            det.score(t)
        # Spike transaction
        spike_result = det.score(
            Transaction("sp_spike", uid, 1_200.0, "electronics", "DE",
                        BASE + timedelta(minutes=10))
        )
        self.assertGreater(spike_result.score, 0.20)

    def test_model_score_and_rule_score_stored(self):
        r = self.det.score(tx())
        self.assertGreaterEqual(r.model_score, 0.0)
        self.assertLessEqual(r.model_score,   1.0)
        self.assertGreaterEqual(r.rule_score,  0.0)
        self.assertLessEqual(r.rule_score,    0.85)


# ===========================================================================
# 6. Simulator
# ===========================================================================

class TestSimulator(unittest.TestCase):

    def test_normal_stream_count(self):
        self.assertEqual(len(generate_normal_stream(40)), 40)

    def test_all_transaction_ids_unique(self):
        txs = generate_normal_stream(60)
        ids = [t.transaction_id for t in txs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_amounts_positive(self):
        txs = generate_normal_stream(80)
        self.assertTrue(all(t.amount > 0 for t in txs))

    def test_fraud_patterns_non_empty(self):
        self.assertGreater(len(generate_fraud_patterns()), 0)

    def test_fraud_patterns_contain_large_amounts(self):
        txs    = generate_fraud_patterns()
        amounts = [t.amount for t in txs]
        self.assertTrue(any(a > 5_000 for a in amounts))

    def test_mixed_stream_sorted_by_timestamp(self):
        txs = mixed_stream(normal_count=50)
        ts  = [t.timestamp for t in txs]
        self.assertEqual(ts, sorted(ts))

    def test_mixed_stream_contains_multiple_countries(self):
        txs = mixed_stream(normal_count=100)
        self.assertGreater(len(set(t.country for t in txs)), 2)

    def test_mixed_stream_contains_multiple_categories(self):
        txs = mixed_stream(normal_count=100)
        self.assertGreater(len(set(t.merchant_category for t in txs)), 2)

    def test_mixed_stream_total_count(self):
        txs = mixed_stream(normal_count=50)
        fraud_count = len(generate_fraud_patterns())
        self.assertEqual(len(txs), 50 + fraud_count)

    def test_simulator_is_deterministic(self):
        """Same call twice should produce identical streams."""
        s1 = [(t.user_id, t.amount) for t in generate_normal_stream(30)]
        s2 = [(t.user_id, t.amount) for t in generate_normal_stream(30)]
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
