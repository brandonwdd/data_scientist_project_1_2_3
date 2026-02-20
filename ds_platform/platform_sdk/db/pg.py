"""
PostgreSQL Database Connection
"""

from typing import Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from platform_sdk.common.config import Config


class Database:
    """Database connection manager"""

    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            database_url = Config.get_database_url()
        
        self.engine: Engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def close(self):
        """Close database connection"""
        self.engine.dispose()


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get global database instance"""
    global _db
    if _db is None:
        _db = Database()
    return _db
