"""Fraud data contracts and validation."""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class TransactionRecord(BaseModel):
    """Transaction record"""
    transaction_id: str
    event_time: datetime
    processing_time: datetime
    amount: float
    amount_usd: float
    currency: str  # USD, EUR, GBP, CNY, JPY, OTHER
    payment_method: str  # credit_card, debit_card, bank_transfer, digital_wallet, crypto, other
    merchant_category: str
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    ip_address_country: Optional[str] = None

    @validator('event_time')
    def event_time_not_future(cls, v):
        """Ensure event_time is not in the future"""
        if v > datetime.now():
            raise ValueError("event_time cannot be in the future")
        return v

    @validator('amount')
    def amount_positive(cls, v):
        """Ensure amount is positive"""
        if v < 0:
            raise ValueError("amount must be positive")
        return v


class UserBehaviorRecord(BaseModel):
    """User behavior record (transaction history)"""
    user_id: str
    transaction_id: str
    event_time: datetime
    amount: float
    currency: str
    payment_method: str
    device_id: Optional[str] = None
    ip_address_country: Optional[str] = None


class RiskSignalRecord(BaseModel):
    """Risk signal record"""
    transaction_id: str
    user_id: str
    event_time: datetime
    signal_type: str  # chargeback, fraud_flag, velocity_alert, device_alert
    signal_value: float  # Risk score or count
    metadata: dict = Field(default_factory=dict)


class UserProfileRecord(BaseModel):
    """User profile record"""
    user_id: str
    account_created_at: datetime
    account_age_days: int
    kyc_status: str  # verified, pending, rejected, not_started
    account_status: str  # active, suspended, closed, restricted
    last_updated: datetime


class FraudLabel(BaseModel):
    """Fraud label definition"""
    transaction_id: str
    as_of_time: datetime
    fraud_label: int = Field(ge=0, le=1)  # 0=legitimate, 1=fraud
    label_time: datetime  # Time when label was determined (after transaction + investigation period)
    investigation_days: int = 7  # Days after transaction to determine fraud


class FeatureRecord(BaseModel):
    """Feature record for training/serving"""
    transaction_id: str
    event_time: datetime
    feature_set_version: str = "fs_fraud_v1"
    features: dict
    label_time: Optional[datetime] = None  # For point-in-time join

    @validator('transaction_id')
    def transaction_id_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("transaction_id cannot be empty")
        return v.strip()

    @validator('event_time')
    def event_time_not_future(cls, v):
        if v > datetime.now():
            raise ValueError("event_time cannot be in the future")
        return v


# Domain sets for validation
CURRENCY_DOMAIN = {"USD", "EUR", "GBP", "CNY", "JPY", "OTHER"}
PAYMENT_METHOD_DOMAIN = {"credit_card", "debit_card", "bank_transfer", "digital_wallet", "crypto", "other"}
KYC_STATUS_DOMAIN = {"verified", "pending", "rejected", "not_started"}
ACCOUNT_STATUS_DOMAIN = {"active", "suspended", "closed", "restricted"}


def validate_currency(value: str) -> bool:
    """Validate currency is in domain set"""
    return value in CURRENCY_DOMAIN


def validate_payment_method(value: str) -> bool:
    """Validate payment_method is in domain set"""
    return value in PAYMENT_METHOD_DOMAIN


def validate_kyc_status(value: str) -> bool:
    """Validate kyc_status is in domain set"""
    return value in KYC_STATUS_DOMAIN


def validate_account_status(value: str) -> bool:
    """Validate account_status is in domain set"""
    return value in ACCOUNT_STATUS_DOMAIN
