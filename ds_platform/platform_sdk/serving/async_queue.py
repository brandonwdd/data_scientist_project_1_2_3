"""Async Queue (Celery): Queue component and state machine"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json
from celery import Celery
from celery.result import AsyncResult

import sys
from pathlib import Path

# Ensure platform_sdk is importable
sdk_path = Path(__file__).parent.parent.parent
if str(sdk_path) not in sys.path:
    sys.path.insert(0, str(sdk_path))

from platform_sdk.common.config import Config
from platform_sdk.db.pg import get_db
from platform_sdk.common.ids import generate_job_id
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)

# Create Celery app
celery_app = Celery(
    "ds_platform",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


class AsyncJobManager:
    """Manage async jobs with state machine tracking"""

    def __init__(self):
        self.db = get_db()

    def enqueue_job(
        self,
        domain: str,
        task_name: str,
        payload: Dict[str, Any],
        callback_url: Optional[str] = None
    ) -> str:
        """
        Enqueue async job
        
        Args:
            domain: Domain name (churn, fraud, rag)
            task_name: Celery task name
            payload: Job payload
            callback_url: Optional callback URL
        
        Returns:
            job_id
        """
        job_id = generate_job_id()
        
        # Write to DB first (queued status) so API can return job_id even if Redis is down
        self._update_job_status(
            job_id=job_id,
            domain=domain,
            status="queued",
            payload=payload,
            callback_url=callback_url
        )
        
        # Enqueue to Celery (optional: if Redis down, job stays queued in DB)
        try:
            celery_app.send_task(
                task_name,
                args=[job_id, domain, payload],
                task_id=job_id
            )
            logger.info(f"Enqueued job {job_id} for domain {domain}")
        except Exception as e:
            logger.warning(f"Celery enqueue failed (Redis down?); job {job_id} is in DB as queued: {e}")
        
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from database"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            
            schema = Config.POSTGRES_SCHEMA
            query = text(f"""
                SELECT job_id, domain, status, payload, result, error,
                       callback_url, created_at, updated_at
                FROM {schema}.async_jobs
                WHERE job_id = :job_id
            """)
            
            result = session.execute(query, {"job_id": job_id}).fetchone()
            
            if result is None:
                return {"status": "not_found"}
            
            return {
                "job_id": result.job_id,
                "domain": result.domain,
                "status": result.status,
                "payload": result.payload if isinstance(result.payload, dict) else json.loads(result.payload) if result.payload else None,
                "result": result.result if isinstance(result.result, dict) else json.loads(result.result) if result.result else None,
                "error": result.error,
                "callback_url": result.callback_url,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "updated_at": result.updated_at.isoformat() if result.updated_at else None
            }
        finally:
            session.close()

    def _update_job_status(
        self,
        job_id: str,
        domain: str,
        status: str,
        payload: Optional[Dict] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        callback_url: Optional[str] = None
    ):
        """Update job status in database"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            
            schema = Config.POSTGRES_SCHEMA
            query = text(f"""
                INSERT INTO {schema}.async_jobs 
                    (job_id, domain, status, payload, result, error, callback_url, created_at, updated_at)
                VALUES 
                    (:job_id, :domain, :status, :payload, :result, :error, :callback_url, :created_at, :updated_at)
                ON CONFLICT (job_id) 
                DO UPDATE SET
                    status = :status,
                    result = :result,
                    error = :error,
                    updated_at = :updated_at
            """)
            
            # Use placeholder for null so DB column is not empty
            NA = "n/a"
            payload_json = json.dumps(payload) if payload else "{}"
            result_json = json.dumps(result) if result else json.dumps({"status": "pending"})
            error_val = error if error else NA
            callback_val = callback_url if callback_url else NA

            session.execute(
                query,
                {
                    "job_id": job_id,
                    "domain": domain,
                    "status": status,
                    "payload": payload_json,
                    "result": result_json,
                    "error": error_val,
                    "callback_url": callback_val,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
            )
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update job status: {e}")
            raise
        finally:
            session.close()

    def _send_callback(self, callback_url: str, job_status: Dict[str, Any]):
        """Send callback to external URL"""
        try:
            import requests
            requests.post(callback_url, json=job_status, timeout=5)
            logger.info(f"Callback sent to {callback_url}")
        except Exception as e:
            logger.error(f"Failed to send callback: {e}")


def create_async_task(
    task_name: str,
    domain: str
) -> Callable:
    """
    Decorator to create async task with state machine tracking
    
    Usage:
        @create_async_task("churn.score_batch", "churn")
        def score_batch(job_id: str, domain: str, payload: Dict):
            # Task implementation
            pass
    """
    job_manager = AsyncJobManager()
    
    def decorator(func: Callable) -> Callable:
        """
        Wrap user function into a Celery task with platform state machine tracking.
        The user function signature must be: (job_id: str, domain: str, payload: Dict[str, Any]) -> Any
        """

        @celery_app.task(name=task_name, bind=True)
        def async_task(self, job_id: str, domain: str, payload: Dict[str, Any]):
            """Async task wrapper with state machine"""
            try:
                # Update status to running
                job_manager._update_job_status(job_id, domain, "running")
                logger.info(f"Job {job_id} started")

                # Execute user task logic
                result = func(job_id, domain, payload)

                # Update status to succeeded
                job_manager._update_job_status(
                    job_id, domain, "succeeded", result=result
                )
                logger.info(f"Job {job_id} completed")

                # Send callback if provided
                job_status = job_manager.get_job_status(job_id)
                if job_status.get("callback_url"):
                    job_manager._send_callback(
                        job_status["callback_url"],
                        job_status
                    )

                return result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Job {job_id} failed: {error_msg}")

                # Update status to failed
                job_manager._update_job_status(
                    job_id, domain, "failed", error=error_msg
                )

                # Send callback if provided
                job_status = job_manager.get_job_status(job_id)
                if job_status.get("callback_url"):
                    job_manager._send_callback(
                        job_status["callback_url"],
                        job_status
                    )

                raise

        return async_task

    return decorator


# Global job manager instance
job_manager = AsyncJobManager()
