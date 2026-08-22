import os
from datetime import datetime, timezone
from typing import Optional
import uuid

from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker


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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    prompt = Column(Text, nullable=False)
    goal = Column(Text, nullable=False)
    prompt_hash = Column(String, index=True, nullable=False)
    cache_hit = Column(Boolean, nullable=False)
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


def log_request(db, prompt:str, goal: str, prompt_hash: str, cache_hit: bool, 
                latency_ms: int, status: str, error_detail: Optional[str] = None):
    entry = RequestLog(
        prompt=prompt, goal=goal, prompt_hash=prompt_hash,
        cache_hit=cache_hit, latency_ms=latency_ms,
        status=status, error_detail=error_detail
    )
    db.add(entry)
    db.commit()


