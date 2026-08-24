from llm import call_llm
from conftest import fake_openai_response


def test_call_llm_returns_correctly_shaped_result(mock_llm_client):
    fake_response = fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "added detail"}'
    )
    mock_llm_client.return_value = fake_response
    result = call_llm("write about dogs", "system msg", "user msg")

    assert result == {
        "original_prompt": "write about dogs",
        "optimized_prompt": "better prompt",
        "changes": "added detail"
    }