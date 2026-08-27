import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from cache import create_vector_index, redis_client
import llm
import eval as eval_module
from main import app


@pytest.fixture(autouse=True)
def clear_cache():
    redis_client.flushdb()
    create_vector_index()
    yield
    redis_client.flushdb()


@pytest.fixture(autouse=True)
def reset_breakers():
    for breaker in [llm.llm_breaker, llm.embedding_breaker, eval_module.eval_breaker]:
        breaker.state = "closed"
        breaker.failure_count = 0
        breaker.opened_at = None


@pytest.fixture
def mock_embeddings():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.05] * 1536)]
    with patch("llm.client.embeddings.create", return_value=fake_response):
        yield


@pytest.fixture
def mock_llm_client():
    with patch("llm.client.chat.completions.create") as mock:
        yield mock


@pytest.fixture
def mock_judge_client():
    with patch("llm.judge_client.chat.completions.create") as mock:
        yield mock


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def fast_retries(monkeypatch):
    monkeypatch.setattr("resilience.time.sleep", lambda seconds: None)


def fake_openai_response(content: str):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    return mock_response


def raise_connection_error():
    raise ConnectionError("simulated outage")

