import pytest
from unittest.mock import patch

from service import optimize_prompt
from conftest import fake_openai_response


def test_optimize_prompt_success(mock_embeddings):
    fake_response = fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "added detail"}'
    )
    with patch("llm.client.chat.completions.create", return_value=fake_response):
        result = optimize_prompt("write about dogs", "engaging intro")

    assert result[0]["optimized_prompt"] == "better prompt"
    assert result[0]["changes"] == "added detail"


def test_optimize_prompt_retries_then_succeeds(mock_embeddings):
    fake_response = fake_openai_response(
        '{"optimized_prompt": "recovered", "changes": "worked on retry"}'
    )

    with patch("llm.client.chat.completions.create") as mock_create:
        mock_create.side_effect = [
            ConnectionError("network blip"),
            ConnectionError("network blip"),
            fake_response
        ]
        result = optimize_prompt("write about cats", "blog intro")

    assert result[0]["optimized_prompt"] == "recovered"
    assert mock_create.call_count == 3


def test_optimize_prompt_invalid_json_raises_value_error(mock_embeddings):
    fake_response = fake_openai_response("not valid json at all")
    with patch("llm.client.chat.completions.create", return_value=fake_response):
        with pytest.raises(ValueError):
            optimize_prompt("write about birds", "listicle")