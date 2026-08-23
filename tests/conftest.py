import pytest
from unittest.mock import patch, MagicMock
from cache import create_vector_index, redis_client


@pytest.fixture(autouse=True)
def clear_cache():
    redis_client.flushdb()
    create_vector_index()
    yield
    redis_client.flushdb()


@pytest.fixture
def mock_embeddings():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.05] * 1536)]
    with patch("llm.client.embeddings.create", return_value=fake_response):
        yield


def fake_openai_response(content: str):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    return mock_response


def raise_connection_error():
    raise ConnectionError("simulated outage")

