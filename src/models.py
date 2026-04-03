"""
models.py  —  Core data models for the Fraud Detection System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class Transaction:
    """Represents a single financial transaction to be scored."""
    transaction_id: str
    user_id:        str
    amount:         float           # transaction amount in EUR
    merchant_category: str          # e.g. 'grocery', 'electronics'
    country:        str             # ISO-2 country code, e.g. 'DE'
    timestamp:      datetime
    is_online:      bool = True
    metadata:       Dict = field(default_factory=dict)


@dataclass
class FraudScore:
    """Result produced by the FraudDetector for a single transaction."""
    transaction_id:     str
    user_id:            str
    score:              float        # 0.0 = clean  →  1.0 = definite fraud
    risk_level:         str          # LOW | MEDIUM | HIGH | CRITICAL
    decision:           str          # APPROVE | REVIEW | BLOCK
    triggered_rules:    List[str]
    features:           Dict
    model_score:        float
    rule_score:         float
    processing_time_ms: float
