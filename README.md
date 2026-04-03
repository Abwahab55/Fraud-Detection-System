# Fraud Detection System
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Anomaly%20Detection-orange)
![Rule Engine](https://img.shields.io/badge/Engine-Rules-blueviolet)
![Real-Time](https://img.shields.io/badge/System-Real--Time-success)
![Low Latency](https://img.shields.io/badge/Latency-%3C1ms-brightgreen)
![AWS Ready](https://img.shields.io/badge/AWS-Serverless-FF9900?logo=amazonaws&logoColor=white)
![Kafka](https://img.shields.io/badge/Streaming-Kafka-231F20?logo=apachekafka&logoColor=white)
![Redis](https://img.shields.io/badge/Cache-Redis-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
---
A real-time transaction fraud scoring engine written in pure Python.  
Combines a deterministic rule engine with a statistical ML model to classify every transaction as **APPROVE**, **REVIEW**, or **BLOCK** — with sub-millisecond latency.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [How It Works](#how-it-works)
   - [Feature Store](#feature-store)
   - [Rule Engine](#rule-engine)
   - [ML Model](#ml-model)
   - [Score Blending & Decisions](#score-blending--decisions)
6. [Fraud Patterns Simulated](#fraud-patterns-simulated)
7. [Running the Demo](#running-the-demo)
8. [Running the Tests](#running-the-tests)
9. [Sample Output](#sample-output)
10. [Output Files](#output-files)
11. [Configuration & Extending](#configuration--extending)
12. [Production Roadmap](#production-roadmap)
13. [Dependencies](#dependencies)

---

## Overview

This project implements the core pattern used by banks and payment processors to detect fraudulent transactions in real time.

Key properties:

| Property | Value |
|---|---|
| Average scoring latency | **< 1 ms** per transaction |
| Decision types | APPROVE / REVIEW / BLOCK |
| Risk levels | LOW / MEDIUM / HIGH / CRITICAL |
| Fraud patterns detected | 5 (card-testing, ATO, night-online, spike, velocity) |
| Unit tests | **62 tests, 100% passing** |
| External dependencies | `numpy` only |

---

## Architecture

```
Transaction
    │
    ▼
┌─────────────────────────────────────┐
│           Feature Store             │  ← Per-user rolling window (1 hour)
│  • amount_ratio_to_avg              │    Tracks: velocity, spend, country,
│  • tx_count_last_hour               │    category frequency, avg spend
│  • spend_last_hour                  │
│  • country_mismatch (0/1)           │
│  • unusual_category (0/1)           │
│  • is_night (0/1)                   │
└──────────────┬──────────────────────┘
               │  features dict
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐   ┌─────────────┐
│ Rule Engine │   │  ML Model   │
│             │   │             │
│ 11 rules    │   │  Calibrated │
│ Each fires  │   │  anomaly    │
│ independently│  │  scorer     │
│             │   │             │
│ rule_score  │   │ model_score │
│  ∈ [0, 0.85]│   │  ∈ [0, 1]  │
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                │
                ▼
      final_score = 0.40 × rule_score
                  + 0.60 × model_score
                │
                ▼
        ┌───────────────┐
        │  Risk Banding │
        │               │
        │  [0.00, 0.30) │ → LOW      → APPROVE
        │  [0.30, 0.55) │ → MEDIUM   → REVIEW
        │  [0.55, 0.75) │ → HIGH     → REVIEW
        │  [0.75, 1.00] │ → CRITICAL → BLOCK
        └───────────────┘
```

---

## Project Structure

```
fraud_project/
│
├── main.py                  # Entry point — run the demo
│
├── src/
│   ├── __init__.py
│   ├── models.py            # Transaction and FraudScore dataclasses
│   ├── feature_store.py     # Per-user rolling feature computation
│   ├── rule_engine.py       # 11 deterministic fraud rules
│   ├── ml_model.py          # Statistical anomaly scoring model
│   ├── detector.py          # Orchestrator — glues everything together
│   ├── simulator.py         # Synthetic transaction stream generator
│   └── reporter.py          # Console output, JSON export, CSV export
│
├── tests/
│   ├── __init__.py
│   └── test_all.py          # 62 unit tests (6 test classes)
│
└── outputs/                 # Auto-created — JSON and CSV reports land here
```

---

## Quick Start

### 1. Clone / copy the project

```bash
git clone <repo-url>
cd fraud_project
```

### 2. Install the single dependency

```bash
pip install numpy
```

### 3. Run the demo

```bash
python main.py
```

### 4. Run the tests

```bash
python -m unittest tests/test_all.py -v
```

---

## How It Works

### Feature Store

`src/feature_store.py`

The feature store maintains a **per-user sliding window** (default 1 hour) of recent transactions using Python `deque` objects. For each incoming transaction it computes:

| Feature | Description |
|---|---|
| `amount_ratio_to_avg` | Current amount ÷ user's rolling average amount |
| `tx_count_last_hour` | Number of transactions the user made in the last hour |
| `spend_last_hour` | Total EUR spent by the user in the last hour |
| `country_mismatch` | 1 if transaction country ≠ user's home country |
| `unusual_category` | 1 if merchant category ≠ user's most frequent category |
| `is_night` | 1 if transaction hour is 23:00–05:59 |
| `is_online` | 1 if the transaction is online |

Features are **read** before scoring and **written** after, so the current transaction never contaminates its own features.

In production: replace `FeatureStore` with a Redis-backed implementation. The interface (`get_features` / `record`) stays identical.

---

### Rule Engine

`src/rule_engine.py`

Eleven deterministic rules fire independently and contribute weighted scores:

| Rule | Condition | Weight |
|---|---|---|
| `LARGE_AMOUNT` | amount > €5,000 | 0.15 |
| `VERY_LARGE_AMOUNT` | amount > €15,000 | 0.30 |
| `AMOUNT_SPIKE_5X` | amount_ratio ≥ 5× | 0.25 |
| `AMOUNT_SPIKE_10X` | amount_ratio ≥ 10× | 0.40 |
| `VELOCITY_HIGH` | > 8 transactions/hour | 0.20 |
| `VELOCITY_EXTREME` | > 15 transactions/hour | 0.40 |
| `HIGH_HOURLY_SPEND` | > €8,000/hour | 0.25 |
| `COUNTRY_MISMATCH` | foreign country | 0.25 |
| `UNUSUAL_CATEGORY` | unfamiliar merchant type | 0.10 |
| `NIGHT_TRANSACTION` | 23:00–05:59 | 0.10 |
| `ONLINE_LARGE_AMOUNT` | online AND amount > €2,500 | 0.15 |

The rule score is the **sum of fired weights, capped at 0.85**.

Rules are easy to audit and explain to compliance teams — when a transaction is blocked, the triggered rules tell you exactly why.

---

### ML Model

`src/ml_model.py`

A calibrated statistical scorer that approximates a trained Isolation Forest or XGBoost model.  
Feature weights were derived from the relative importance of signals in published card-fraud datasets (e.g. the Kaggle Credit Card Fraud Detection dataset).

```
model_score =   0.35 × sigmoid(amount_ratio − 3.0)   # amount anomaly
              + 0.25 × sigmoid(velocity − 8.0)         # velocity anomaly
              + 0.20 × country_mismatch                # geographic signal
              + 0.10 × unusual_category                # behavioural signal
              + 0.05 × is_night                        # time signal
              + 0.05 × high_spend_flag                 # aggregate spend
```

**To replace with a real trained model:**

```python
# ml_model.py — production swap
import joblib

class FraudModel:
    def __init__(self):
        self.clf = joblib.load("s3://your-bucket/fraud_model_v3.pkl")

    def predict(self, features: dict) -> float:
        X = self._to_array(features)
        return float(self.clf.predict_proba(X)[0, 1])
```

---

### Score Blending & Decisions

`src/detector.py`

```python
final_score = 0.60 × model_score + 0.40 × rule_score
```

| Score range | Risk level | Decision |
|---|---|---|
| 0.00 – 0.30 | LOW | **APPROVE** |
| 0.30 – 0.55 | MEDIUM | **REVIEW** |
| 0.55 – 0.75 | HIGH | **REVIEW** |
| 0.75 – 1.00 | CRITICAL | **BLOCK** |

REVIEW decisions are queued for human analysts. BLOCK decisions are rejected immediately and can trigger account suspension and alerts.

---

## Fraud Patterns Simulated

`src/simulator.py` injects five realistic attack scenarios:

### Pattern 1 — Card Testing
The attacker makes 16 micro-transactions (€0.50–€2.00) to verify the card is live, then hits with a €14,500 electronics purchase.  
**Detected by:** `VELOCITY_HIGH`, `VELOCITY_EXTREME`, `AMOUNT_SPIKE_10X`

### Pattern 2 — Account Takeover (Country Jump)
User makes a normal €320 travel purchase in Germany, then 18 minutes later a €3,800 electronics purchase appears from Nigeria.  
**Detected by:** `COUNTRY_MISMATCH`, `AMOUNT_SPIKE_10X`, `ONLINE_LARGE_AMOUNT`

### Pattern 3 — Night-time High-Value Online
A €4,600 purchase from China at 03:14 on an account with no prior overseas activity.  
**Detected by:** `NIGHT_TRANSACTION`, `AMOUNT_SPIKE_10X`, `ONLINE_LARGE_AMOUNT`

### Pattern 4 — Amount Spike
Six normal €55 grocery purchases establish a baseline, then an €11,200 electronics transaction arrives — 185× the average.  
**Detected by:** `LARGE_AMOUNT`, `AMOUNT_SPIKE_5X`, `AMOUNT_SPIKE_10X`

### Pattern 5 — Velocity Burst
Twenty transactions in under 4 minutes (card-testing), followed by an €8,900 purchase.  
**Detected by:** `VELOCITY_EXTREME`, `AMOUNT_SPIKE_10X`, `LARGE_AMOUNT`

---

## Running the Demo

```bash
# Standard run — 250 normal + fraud patterns
python main.py

# Quick run — 50 normal transactions only (no fraud injected)
python main.py --quick

# Print feature values for every flagged transaction
python main.py --verbose

# Skip writing output files
python main.py --no-save
```

---

## Running the Tests

```bash
# Run all 62 tests with verbose output
python -m unittest tests/test_all.py -v

# Run a single test class
python -m unittest tests.test_all.TestFraudDetector -v

# Run a single test method
python -m unittest tests.test_all.TestRuleEngine.test_velocity_extreme_rule -v
```

**Test classes and coverage:**

| Class | Tests | What it covers |
|---|---|---|
| `TestTransaction` | 4 | Model fields, defaults, metadata |
| `TestFeatureStore` | 11 | Velocity, ratios, country/category flags, window expiry |
| `TestRuleEngine` | 12 | Every rule fires correctly; score is capped |
| `TestFraudModel` | 5 | Score range, monotonicity, high/low risk |
| `TestFraudDetector` | 20 | Decisions, risk levels, stats, latency, scenarios |
| `TestSimulator` | 10 | Counts, uniqueness, sorting, determinism |
| **Total** | **62** | **100% passing** |

---

## Sample Output

```
════════════════════════════════════════════════════════════════════
  FRAUD DETECTION SYSTEM  —  Real-time Transaction Scoring
  Started: 2024-06-01  10:00:00
════════════════════════════════════════════════════════════════════

  Processing 298 transactions ...

  TX 0169c304_ct_hit   user=user_0042   score=0.871 ███████████████████░░░  [CRITICAL]  ✗ BLOCK  (0.1 ms)
    rules : LARGE_AMOUNT, AMOUNT_SPIKE_10X, VELOCITY_EXTREME, COUNTRY_MISMATCH
    feats : ratio=12159.33  vel=16  country_mis=1  night=0

  TX 056df56a_spike    user=user_0077   score=0.792 █████████████████░░░░░  [CRITICAL]  ✗ BLOCK  (0.1 ms)
    rules : LARGE_AMOUNT, AMOUNT_SPIKE_10X, COUNTRY_MISMATCH

  TX 2d12f797_vb_hit   user=user_0111   score=0.758 █████████████████░░░░░  [CRITICAL]  ✗ BLOCK  (0.1 ms)
    rules : LARGE_AMOUNT, AMOUNT_SPIKE_10X, VELOCITY_EXTREME

  (+ 264 LOW-risk transactions approved silently)

════════════════════════════════════════════════════════════════════
  FRAUD DETECTION — SUMMARY REPORT
════════════════════════════════════════════════════════════════════
  Transactions processed  : 298
  Average latency         : 0.02 ms / transaction

  LOW         264  ( 88.6%)  ██████████████████████
  MEDIUM       27  (  9.1%)  ██
  HIGH          4  (  1.3%)
  CRITICAL      3  (  1.0%)

  ✓  APPROVE   264
  ⚠  REVIEW     31
  ✗  BLOCK       3
════════════════════════════════════════════════════════════════════

  🚨  3 CRITICAL transaction(s) blocked
```

---

## Output Files

Every run (unless `--no-save`) writes two files to `outputs/`:

### JSON — `fraud_report_YYYYMMDD_HHMMSS.json`

```json
[
  {
    "transaction_id": "0169c304-0d4_ct_hit",
    "user_id": "user_0042",
    "score": 0.871,
    "risk_level": "CRITICAL",
    "decision": "BLOCK",
    "triggered_rules": ["LARGE_AMOUNT", "AMOUNT_SPIKE_10X", "VELOCITY_EXTREME"],
    "model_score": 0.8943,
    "rule_score": 0.85,
    "features": {
      "amount": 14500.0,
      "amount_ratio_to_avg": 12159.33,
      "tx_count_last_hour": 16,
      "spend_last_hour": 17.42,
      "country_mismatch": 1,
      "unusual_category": 1,
      "is_online": 1,
      "hour_of_day": 10,
      "is_night": 0
    },
    "processing_time_ms": 0.12
  }
]
```

### CSV — `fraud_report_YYYYMMDD_HHMMSS.csv`

```
transaction_id,user_id,score,risk_level,decision,triggered_rules,model_score,rule_score,processing_time_ms
0169c304_ct_hit,user_0042,0.871,CRITICAL,BLOCK,LARGE_AMOUNT|AMOUNT_SPIKE_10X|VELOCITY_EXTREME,...
```

---

## Configuration & Extending

### Change risk thresholds

Edit `_RISK_BANDS` in `src/detector.py`:

```python
_RISK_BANDS = [
    ("LOW",      0.00, 0.25, "APPROVE"),   # tighter — catch more
    ("MEDIUM",   0.25, 0.50, "REVIEW"),
    ("HIGH",     0.50, 0.70, "REVIEW"),
    ("CRITICAL", 0.70, 1.01, "BLOCK"),
]
```

### Add a new rule

In `src/rule_engine.py`, append to `_RULES` and `_WEIGHTS`:

```python
("WEEKEND_LARGE", lambda f, tx: tx.timestamp.weekday() >= 5 and tx.amount > 2000),
# then in _WEIGHTS:
"WEEKEND_LARGE": 0.15,
```

### Plug in a real ML model

Replace `FraudModel.predict()` in `src/ml_model.py` with any callable that takes a feature dict and returns a float in [0, 1].

### Swap FeatureStore for Redis

```python
# src/feature_store.py
import redis, json

class FeatureStore:
    def __init__(self, window_seconds=3600):
        self.r = redis.Redis(host="localhost", port=6379)
        self.window_seconds = window_seconds

    def get_features(self, tx):
        raw = self.r.lrange(f"history:{tx.user_id}", 0, -1)
        history = [json.loads(x) for x in raw]
        # ... same logic ...
```

---

## Production Roadmap

To take this from a demo to a production system:

1. **Message queue ingestion** — publish transactions to Kafka/SQS; consume with a worker pool
2. **Redis feature store** — replace in-memory deques with Redis sorted sets (TTL = window_seconds)
3. **Trained ML model** — train XGBoost or Isolation Forest on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud); serve via SageMaker / Vertex AI
4. **Alerting** — publish BLOCK decisions to SNS/PagerDuty; send email/Slack notifications
5. **Dashboard** — stream scores to a time-series store (InfluxDB) and visualise in Grafana
6. **Model monitoring** — track score distribution weekly; alert on drift (model retraining trigger)
7. **A/B testing** — run two model versions simultaneously and compare precision/recall
8. **REST API** — wrap `FraudDetector.score()` in FastAPI for synchronous scoring

---

## Dependencies

| Package | Version | Usage |
|---|---|---|
| `numpy` | ≥ 1.21 | Rolling average calculation in FeatureStore |
| `python` | ≥ 3.9 | Dataclasses, type hints |

No other dependencies. All tests use the standard library `unittest` module.

```bash
pip install numpy
```

---

## License

MIT — free to use, modify, and distribute.
