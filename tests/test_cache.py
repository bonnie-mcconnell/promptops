from unittest.mock import patch, MagicMock

from service import optimize_prompt
from conftest import fake_openai_response

def test_semantic_cache_hit_when_vectors_match(mock_embeddings):
    same_vector = [0.1] * 1536
    fake_embedding_response = MagicMock()
    fake_embedding_response.data = [MagicMock(embedding=same_vector)]

    fake_llm_response = fake_openai_response(
        '{"optimized_prompt": "better prompt", "changes": "done"}'
    )

    with patch("llm.client.embeddings.create", return_value=fake_embedding_response), \
        patch("llm.client.chat.completions.create", return_value=fake_llm_response):
        first = optimize_prompt("write about dogs", "engaging intro")
        second = optimize_prompt("write something different entirely", "engaging intro")

    assert second[0]["optimized_prompt"] == "better prompt"


def test_semantic_cache_miss_when_vectors_differ():
    vec_a = [1.0] + [0.0] * 1535
    vec_b = [0.0] * 1535 + [1.0]

    response_a = MagicMock()
    response_a.data = [MagicMock(embedding=vec_a)]
    response_b = MagicMock()
    response_b.data = [MagicMock(embedding=vec_b)]

    fake_llm_response = fake_openai_response(
        '{"optimized_prompt": "first_answer", "changes": "done"}'
    )

    with patch("llm.client.embeddings.create", return_value=response_a), \
        patch("llm.client.chat.completions.create", return_value=fake_llm_response):
        optimize_prompt("prompt one", "goal one")

    second_llm_response = fake_openai_response(
        '{"optimized_prompt": "second answer", "changes": "different"}'
    )

    with patch("llm.client.embeddings.create", return_value=response_b), \
        patch("llm.client.chat.completions.create", return_value=second_llm_response):
        result = optimize_prompt("prompt two", "goal two")

    assert result[0]["optimized_prompt"] == "second answer"