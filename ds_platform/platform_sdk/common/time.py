"""Time Utilities"""

from datetime import datetime, timezone, timedelta
from typing import Optional

def now_utc() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)

def parse_datetime(s: str) -> datetime:
    """Parse datetime string to UTC datetime"""
    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def format_datetime(dt: datetime) -> str:
    """Format datetime to ISO string"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()