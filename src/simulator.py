"""
simulator.py  —  Generates realistic transaction streams for testing.

Produces both benign traffic and five canonical fraud patterns:
  1. Card-testing  — rapid micro-transactions before a large hit
  2. Account takeover — sudden high-value purchase from a new country
  3. Night-time online — large purchase at 03:00 from an unusual location
  4. Amount spike  — 10× normal spend after baseline established
  5. Velocity burst — 20 transactions in < 5 minutes
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List

from src.models import Transaction


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "petrol", "electronics",
    "clothing", "travel", "pharmacy", "entertainment", "online_retail",
]

COUNTRIES = ["DE", "FR", "GB", "NL", "ES", "IT", "US", "CN", "BR", "NG"]

# Pre-generate stable user profiles (seeded for reproducibility)
_rng = random.Random(42)

USER_PROFILES = {
    f"user_{i:04d}": {
        "home_country":    _rng.choice(COUNTRIES[:6]),
        "avg_spend":       _rng.uniform(30, 180),
        "top_category":    _rng.choice(MERCHANT_CATEGORIES),
    }
    for i in range(1, 201)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tx(uid: str, amount: float, category: str, country: str,
        ts: datetime, is_online: bool = True, suffix: str = "") -> Transaction:
    return Transaction(
        transaction_id    = str(uuid.uuid4())[:12] + suffix,
        user_id           = uid,
        amount            = round(amount, 2),
        merchant_category = category,
        country           = country,
        timestamp         = ts,
        is_online         = is_online,
    )


# ---------------------------------------------------------------------------
# Normal traffic
# ---------------------------------------------------------------------------

def generate_normal_stream(n: int = 300, start: datetime = None) -> List[Transaction]:
    """
    Generate *n* realistic, benign transactions spread across multiple users.
    Amounts follow a Gaussian distribution around each user's average.
    """
    if start is None:
        start = datetime(2024, 6, 1, 9, 0, 0)

    txs, t = [], start
    rng = random.Random(99)
    user_ids = list(USER_PROFILES.keys())

    for _ in range(n):
        uid      = rng.choice(user_ids)
        profile  = USER_PROFILES[uid]
        amount   = max(1.0, rng.gauss(profile["avg_spend"], profile["avg_spend"] * 0.3))
        category = profile["top_category"] if rng.random() < 0.72 else rng.choice(MERCHANT_CATEGORIES)
        country  = profile["home_country"] if rng.random() < 0.88 else rng.choice(COUNTRIES[:6])
        is_online = rng.random() < 0.45
        t += timedelta(seconds=rng.randint(10, 90))
        txs.append(_tx(uid, amount, category, country, t, is_online))

    return txs


# ---------------------------------------------------------------------------
# Fraud patterns
# ---------------------------------------------------------------------------

def generate_fraud_patterns(base: datetime = None) -> List[Transaction]:
    """
    Inject five textbook fraud scenarios into the stream.
    Returns a flat list sorted by timestamp.
    """
    if base is None:
        base = datetime(2024, 6, 1, 10, 0, 0)

    txs: List[Transaction] = []

    # ── Pattern 1: Card-testing (rapid micro-txs → large hit) ─────────────
    uid1, t1 = "user_0042", base
    for i in range(16):
        t1 += timedelta(seconds=random.randint(8, 25))
        txs.append(_tx(uid1, round(random.uniform(0.50, 2.00), 2),
                       "online_retail", "DE", t1, suffix=f"_ct{i}"))
    t1 += timedelta(minutes=3)
    txs.append(_tx(uid1, 14_500.00, "electronics", "DE", t1, suffix="_ct_hit"))

    # ── Pattern 2: Account takeover — country jump ─────────────────────────
    uid2, t2 = "user_0099", base + timedelta(minutes=25)
    txs.append(_tx(uid2, 320.00, "travel",       "DE", t2))
    t2 += timedelta(minutes=18)
    txs.append(_tx(uid2, 3_800.00, "electronics", "NG", t2, suffix="_ato"))

    # ── Pattern 3: Night-time high-value online purchase ──────────────────
    uid3 = "user_0150"
    t3   = datetime(2024, 6, 1, 3, 14, 0)
    txs.append(_tx(uid3, 4_600.00, "online_retail", "CN", t3, suffix="_night"))

    # ── Pattern 4: 10× amount spike after normal baseline ─────────────────
    uid4, t4 = "user_0077", base + timedelta(hours=1)
    for j in range(6):                       # establish baseline
        t4 += timedelta(minutes=4)
        txs.append(_tx(uid4, 55.00, "grocery", "FR", t4, is_online=False))
    t4 += timedelta(minutes=8)
    txs.append(_tx(uid4, 11_200.00, "electronics", "FR", t4, suffix="_spike"))

    # ── Pattern 5: Velocity burst — 20 txs in 4 minutes ──────────────────
    uid5, t5 = "user_0111", base + timedelta(hours=2)
    for k in range(20):
        t5 += timedelta(seconds=random.randint(5, 15))
        txs.append(_tx(uid5, round(random.uniform(1.0, 5.0), 2),
                       "online_retail", "US", t5, suffix=f"_vb{k}"))
    t5 += timedelta(seconds=30)
    txs.append(_tx(uid5, 8_900.00, "electronics", "US", t5, suffix="_vb_hit"))

    return sorted(txs, key=lambda tx: tx.timestamp)


# ---------------------------------------------------------------------------
# Combined stream
# ---------------------------------------------------------------------------

def mixed_stream(normal_count: int = 250) -> List[Transaction]:
    """
    Return normal traffic interleaved with fraud patterns, sorted by timestamp.
    """
    base   = datetime(2024, 6, 1, 8, 0, 0)
    normal = generate_normal_stream(normal_count, start=base)
    fraud  = generate_fraud_patterns(base=base + timedelta(hours=2))
    return sorted(normal + fraud, key=lambda tx: tx.timestamp)
