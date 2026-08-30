import os
import pytest
import time
from sqlalchemy import text

from ratelimit import WINDOW_SECONDS, RATE_LIMIT_PER_MINUTE
from cache import redis_client 
from database import engine


api_key = os.environ["API_KEY"]

OPTIMIZE_BODY = {"prompt": "Prompt to do", "goal": "goal i have"}
COMPARE_BODY = {"prompt": "prompt", "goal": "goal", "candidate_a": "prompt a", "candidate_b": "prompt b"}


def test_health_check(test_client):
    response = test_client.get("/health")
    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "ok"


@pytest.mark.parametrize("method,path,body", [
    ("post", "/optimize", OPTIMIZE_BODY),
    ("post", "/compare", COMPARE_BODY),
    ("get", "/stats", None),
])
def test_no_api_key(test_client, method, path, body):
    call = getattr(test_client, method)
    kwargs = {"headers": {}}
    if body is not None:
        kwargs["json"] = body
    response = call(path, **kwargs)
    result = response.json()
    assert response.status_code == 401
    assert result["detail"] == "Not authenticated"


@pytest.mark.parametrize("method,path,body", [
    ("post", "/optimize", OPTIMIZE_BODY),
    ("post", "/compare", COMPARE_BODY),
    ("get", "/stats", None),
])
def test_invalid_api_key(test_client, method, path, body):
    call = getattr(test_client, method)
    kwargs = {"headers": {"X-API-Key": "invalid key"}}
    if body is not None:
        kwargs["json"] = body
    response = call(path, **kwargs)
    result = response.json()
    assert response.status_code == 401
    assert result["detail"] == "Invalid or missing API key"


def test_optimize_valid_api_key(test_client):
    response = test_client.post("/optimize", json=OPTIMIZE_BODY, headers={"X-API-Key": api_key})
    result = response.json()
    assert response.status_code == 200
    assert result["original_prompt"] == "Prompt to do"
    assert result["optimized_prompt"] == "Act as an expert. Prompt to do. Goal: goal i have"
    assert result["changes"] == "Added persona framing, context structure, and clear constraints."


def test_compare_valid_api_key(test_client):
    response = test_client.post("/compare", json=COMPARE_BODY, headers={"X-API-Key": api_key})
    result = response.json()
    assert response.status_code == 200
    assert result["score_a"] == 5.3
    assert result["score_b"] == 8.8
    assert result["winner"] == "b"
    assert result["reasoning"] == "Reasoning"


def test_stats_valid_api_key(test_client):
    response = test_client.get("/stats", headers={"X-API-Key": api_key})
    result = response.json()
    assert response.status_code == 200
    expected_fields = ["total", "exact_hits", "semantic_hits", "cache_misses", "errors", "avg_latency_ms", "p50_latency_ms", "p95_latency_ms"]
    for field in expected_fields:
        assert field in result


def test_stats_empty_database(test_client):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM requests"))
        conn.commit()

    response = test_client.get("/stats", headers={"X-API-key": api_key})
    result = response.json()

    assert response.status_code == 200
    assert result["total"] == 0
    assert result["exact_hits"] == 0
    assert result["semantic_hits"] == 0
    assert result["cache_misses"] == 0
    assert result["errors"] == 0
    assert result["avg_latency_ms"] is None
    assert result["p50_latency_ms"] is None
    assert result["p95_latency_ms"] is None


def test_rate_limit(test_client):
    window = int((time.time() // WINDOW_SECONDS))
    key = f"ratelimit:{api_key}:{window}"
    redis_client.set(key, RATE_LIMIT_PER_MINUTE)
    response = test_client.post("/optimize", json=OPTIMIZE_BODY, headers={"X-API-Key": api_key})
    result = response.json()

    try:
        assert response.status_code == 429
        assert result["detail"] == "Rate limit exceeded. Please slow down."

    finally:
        redis_client.delete(key)
