import os
from datetime import datetime, timezone
from typing import Optional
import uuid

from sqlalchemy import create_engine, Column, String, Integer, text, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://promptops:promptops@postgres:5432/promptops"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "requests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    prompt = Column(Text, nullable=False)
    goal = Column(Text, nullable=False)
    prompt_hash = Column(String, index=True, nullable=False)
    cache_type = Column(String, nullable=False) # "exact" | "semantic" | "none"
    latency_ms = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    error_detail = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()


def log_request(db, prompt:str, goal: str, prompt_hash: str, cache_type: str, 
                latency_ms: int, status: str, error_detail: Optional[str] = None):
    entry = RequestLog(
        prompt=prompt, goal=goal, prompt_hash=prompt_hash,
        cache_type=cache_type, latency_ms=latency_ms,
        status=status, error_detail=error_detail
    )
    db.add(entry)
    db.commit()


def get_stats(db: Session) -> dict:
    result = db.execute(text(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE cache_type = 'exact') AS exact_hits,
            COUNT(*) FILTER (WHERE cache_type = 'semantic') AS semantic_hits,
            COUNT(*) FILTER (WHERE cache_type = 'none') AS cache_misses,
            COUNT(*) FILTER (WHERE status = 'error') AS errors,
            AVG(latency_ms) AS avg_latency_ms,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE status = 'ok') AS p50_latency_ms,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE status = 'ok') AS p95_latency_ms
        FROM requests
    """)).mappings().first()

    assert result is not None
    if result["total"] == 0:
        return {
            "total": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "avg_latency_ms": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None
        }

    return dict(result)

