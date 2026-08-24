from unittest.mock import patch

import pytest

from conftest import fake_openai_response
from eval import run_comparison, judge_output


def test_run_comparison_full_flow(mock_llm_client, mock_judge_client):
    small_eval_set = [
        {"prompt": "write about dogs", "goal": "engaging intro"},
        {"prompt": "explain gravity", "goal": "simple for kids"},
    ]

    mock_llm_client.side_effect = [
        fake_openai_response('{"optimized_prompt": "A1", "changes": "c"}'),
        fake_openai_response('{"optimized_prompt": "B1", "changes": "c"}'),
        fake_openai_response('{"optimized_prompt": "A2", "changes": "c"}'),
        fake_openai_response('{"optimized_prompt": "B2", "changes": "c"}'),
    ]

    mock_judge_client.side_effect = [
        fake_openai_response('{"score_1": 6, "score_2": 8, "reasoning": "B better"}'),
        fake_openai_response('{"score_1": 7, "score_2": 9, "reasoning": "B better again"}'),
    ]

    with patch("random.choice", return_value=True):  # force a_first = True, deterministic
        result = run_comparison("system A", "system B", small_eval_set, min_comparisons=2)

    assert result["n_compared"] == 2
    assert result["n_skipped"] == 0
    assert result["median_difference"] > 0
    assert len(result["per_prompt_results"]) == 2


def test_run_comparison_skips_failed_prompts(mock_llm_client, mock_judge_client, fast_retries):
    eval_set = [{"prompt": f"prompt {i}", "goal": f"goal {i}"} for i in range(1, 7)]  # 6 items

    good_response = fake_openai_response('{"optimized_prompt": "X", "changes": "c"}')
    judge_response = fake_openai_response('{"score_1": 6, "score_2": 7, "reasoning": "ok"}')

    mock_llm_client.side_effect = [
        good_response, good_response,  # item 1: A, B succeed
        good_response, good_response,  # item 2: A, B succeed
        ConnectionError(), ConnectionError(), ConnectionError(),  # item 3: A exhausts retries, fails
        good_response, good_response,  # item 4: A, B succeed
        good_response, good_response,  # item 5: A, B succeed
        good_response, good_response,  # item 6: A, B succeed
    ]
    mock_judge_client.side_effect = [judge_response] * 5  # only 5 judged, item 3 skipped

    result = run_comparison("system A", "system B", eval_set)

    assert result["n_compared"] == 5
    assert result["n_skipped"] == 1


def test_run_comparison_raises_with_too_few_successes(mock_llm_client, mock_judge_client, fast_retries):
    eval_set = [{"prompt": f"prompt {i}", "goal": f"goal {i}"} for i in range(1, 4)]  # only 3 items

    good_response = fake_openai_response('{"optimized_prompt": "X", "changes": "c"}')

    mock_llm_client.side_effect = [
        ConnectionError(), ConnectionError(), ConnectionError(),  # item 1 fails
        good_response, good_response,                            # item 2 succeeds
        good_response, good_response,                            # item 3 succeeds
    ]

    with pytest.raises(ValueError, match="not enough data"):
        run_comparison("system A", "system B", eval_set)


def test_judge_output_unscrambles_position_correctly(mock_judge_client):
    fake_judge_response = fake_openai_response(
        '{"score_1": 6, "score_2": 9, "reasoning": "version 2 much better"}'
    )
    mock_judge_client.return_value = fake_judge_response

    with patch("random.choice", return_value=True):  # a_first=True, so A=version_1
        result = judge_output("some prompt", "some goal", "output A text", "output B text")

    assert result["score_a"] == 6
    assert result["score_b"] == 9

    with patch("random.choice", return_value=False):  # a_first=False, so B=version_1
        result = judge_output("some prompt", "some goal", "output A text", "output B text")

    assert result["score_a"] == 9
    assert result["score_b"] == 6