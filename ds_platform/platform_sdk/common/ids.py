"""ID Generation Utilities"""

import uuid
from datetime import datetime

def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())

def generate_job_id() -> str:
    """Generate unique job ID"""
    return str(uuid.uuid4())

def generate_trace_id() -> str:
    """Generate unique trace ID"""
    return str(uuid.uuid4())
