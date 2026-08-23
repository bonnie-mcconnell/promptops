import time
import pytest
from unittest.mock import patch, MagicMock
import redis

from service import CircuitBreaker, CircuitOpenError, optimize_prompt, redis_client, create_vector_index


@pytest.fixture(autouse=True)
def clear_cache():
    redis_client.flushdb()
    create_vector_index()
    yield
    redis_client.flushdb()


@pytest.fixture(autouse=True)
def mock_embeddings():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.05] * 1536)]
    with patch("service.client.embeddings.create", return_value=fake_response):
        yield


def _fake_openai_response(content: str):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    return mock_response


def test_optimize_prompt_success():
    fake_response = _fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "added detail"}'
    )
    with patch("service.client.chat.completions.create", return_value=fake_response):
        result = optimize_prompt("write about dogs", "engaging intro")

    assert result["optimized_prompt"] == "better prompt"
    assert result["changes"] == "added detail"


def test_optimize_prompt_retries_then_succeeds():
    fake_response = _fake_openai_response(
        '{"optimized_prompt": "recovered", "changes": "worked on retry"}'
    )

    with patch("service.client.chat.completions.create") as mock_create:
        mock_create.side_effect = [
            ConnectionError("network blip"),
            ConnectionError("network blip"),
            fake_response
        ]
        result = optimize_prompt("write about cats", "blog intro")

    assert result["optimized_prompt"] == "recovered"
    assert mock_create.call_count == 3


def test_optimize_prompt_invalid_json_raises_value_error():
    fake_response = _fake_openai_response("not valid json at all")
    with patch("service.client.chat.completions.create", return_value=fake_response):
        with pytest.raises(ValueError):
            optimize_prompt("write about birds", "listicle")


def _raise_connection_error():
    raise ConnectionError("simulated outage")


def test_breaker_opens_after_threshold_failures():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=5)

    def always_fails():
        raise ConnectionError("simulated LLM outage")

    for _ in range(3):
        with pytest.raises(ConnectionError):
            breaker.call(always_fails)

    assert breaker.state == "open"


def test_breaker_fails_fast_while_open():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5)

    with pytest.raises(ConnectionError):
        breaker.call(_raise_connection_error)

    assert breaker.state == "open"

    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: 1 / 0) # should never actually execute


def test_breaker_recovers_after_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=1)

    with pytest.raises(ConnectionError):
        breaker.call(lambda: (_ for _ in ()).throw(ConnectionError))
    assert breaker.state == "open"

    time.sleep(1.1)
    result = breaker.call(lambda: "success")

    assert result == "success"
    assert breaker.state == "closed"



def test_semantic_cache_hit_when_vectors_match():
    same_vector = [0.1] * 1536
    fake_embedding_response = MagicMock()
    fake_embedding_response.data = [MagicMock(embedding=same_vector)]

    fake_llm_response = _fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "done"}'
    )

    with patch("service.client.embeddings.create", return_value=fake_embedding_response), \
        patch("service.client.chat.completions.create", return_value=fake_llm_response):
        first = optimize_prompt("write about dogs", "engaging intro")

    with patch("service.client.embeddings.create", return_value=fake_embedding_response):
        second = optimize_prompt("write something different entirely", "engaging intro")

    assert second["optimized_prompt"] == "better prompt"


def test_semantic_cache_miss_when_vectors_differ():
    vec_a = [1.0] + [0.0] * 1535
    vec_b = [0.0] * 1535 + [1.0]

    response_a = MagicMock()
    response_a.data = [MagicMock(embedding=vec_a)]
    response_b = MagicMock()
    response_b.data = [MagicMock(embedding=vec_b)]

    fake_llm_response = _fake_openai_response(
        '{"optimized_prompt": "first_answer", "changes": "done"}'
    )

    with patch("service.client.embeddings.create", return_value=response_a), \
        patch("service.client.chat.completions.create", return_value=fake_llm_response):
        optimize_prompt("prompt one", "goal one")

    second_llm_response = _fake_openai_response(
        '{"optimized_prompt": "second answer", "changes": "different"}'
    )

    with patch("service.client.embeddings.create", return_value=response_b), \
        patch("service.client.chat.completions.create", return_value=second_llm_response):
        result = optimize_prompt("prompt two", "goal two")

    assert result["optimized_prompt"] == "second answer"