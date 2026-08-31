from unittest.mock import patch

import pytest

from service import optimize_prompt
from conftest import fake_openai_response


def test_optimize_prompt_success(mock_embeddings, mock_llm_client):
    fake_response = fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "added detail"}'
    )

    mock_llm_client.return_value = fake_response
    result = optimize_prompt("write about dogs", "engaging intro")

    assert result[0]["optimized_prompt"] == "better prompt"
    assert result[0]["changes"] == "added detail"


def test_optimize_prompt_retries_then_succeeds(mock_embeddings, mock_llm_client, fast_retries):
    fake_response = fake_openai_response(
        '{"optimized_prompt": "recovered", "changes": "worked on retry"}'
    )

    mock_llm_client.side_effect = [
            ConnectionError("network blip"),
            ConnectionError("network blip"),
            fake_response
        ]
    result = optimize_prompt("write about cats", "blog intro")

    assert result[0]["optimized_prompt"] == "recovered"
    assert mock_llm_client.call_count == 3


def test_optimize_prompt_invalid_json_raises_value_error(mock_embeddings, mock_llm_client):
    fake_response = fake_openai_response("not valid json at all")
    mock_llm_client.return_value = fake_response
    with pytest.raises(ValueError):
        optimize_prompt("write about birds", "listicle")


def test_optimize_prompt_semantic_lookup_failure_still_succeeds(mock_embeddings, mock_llm_client):
    fake_response = fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "added detail"}'
    )
    mock_llm_client.return_value = fake_response
    with patch("service.semantic_cache_lookup", side_effect=ConnectionError("semantic lookup failed")):
        result = optimize_prompt("write about dogs", "engaging intro")

    assert result[0]["optimized_prompt"] == "better prompt"
    assert result[1] == "none"
    assert result[2] is not None
    assert "lookup" in result[2].lower()


def test_optimize_prompt_semantic_write_failure_still_succeeds(mock_embeddings, mock_llm_client):
    fake_response = fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "added detail"}'
    )
    mock_llm_client.return_value = fake_response
    with patch("service.store_semantic_cache", side_effect=ConnectionError("semantic store failed")):
        result = optimize_prompt("write about dogs", "engaging intro")
    
    assert result[0]["optimized_prompt"] == "better prompt"
    assert result[1] == "none"
    assert result[2] is not None
    assert "store" in result[2].lower()