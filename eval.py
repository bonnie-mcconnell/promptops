from scipy.stats import wilcoxon
from typing import cast
import statistics
import random
import json

from resilience import CircuitBreaker, call_with_retries
from llm import call_llm, judge_client, build_user_message


eval_breaker = CircuitBreaker(failure_threshold = 5, cooldown_seconds = 30)


JUDGE_SYSTEM_MESSAGE = """You are an expert evaluator of prompt engineering quality.
You will be shown an original prompt, a goal, and two candidate optimized versions
labeled "Version 1" and "Version 2". Score EACH version from 1-10 on how well it
would help a language model produce output matching the stated goal.

A score of 5 means "no better than doing nothing", that the optimization added no
real value. Scores above 5 reflect genuine improvement (added useful structure,
reduced ambiguity, clarified intent). Scores below 5 reflect the optimization
making things worse (added irrelevant padding, introduced ambiguity, drifted
from the actual goal).

Return ONLY a JSON object with exactly these fields:
- "score_1": integer 1-10, your score for Version 1
- "score_2": integer 1-10, your score for Version 2
- "reasoning": one sentence explaining the scores

No markdown, no extra text."""


def judge_output(prompt: str, goal: str, output_a: str, output_b: str) -> dict:
    """
    Calls an LLM with two outputs for the same prompt and parse the judges response to return scores for both outputs.
    Randomly decide which output gets shown to the judge first.
    Builds judge system message using rubric (anchored 1-10 scale, 5 is no better than original)
    """
    a_first = random.choice([True, False])
    version_1 = output_a if a_first else output_b
    version_2 = output_b if a_first else output_a

    user_message = f"""Original prompt: {prompt}
    Goal: {goal}

    Version 1: {version_1}
    
    Version 2: {version_2}"""

    def _call_judge():
        response = judge_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        result_text = response.choices[0].message.content
        if not result_text:
            raise ValueError("Judge returned an empty message body.")
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned invalid JSON: {result_text[:200]}")

        if "score_1" not in result or "score_2" not in result:
            raise ValueError(f"Judge response missing required fields. Got: {list(result.keys())}")

        return result

    judged = eval_breaker.call(lambda: call_with_retries(_call_judge))

    score_1, score_2 = float(judged["score_1"]), float(judged["score_2"])
    if a_first:
        return {"score_a": score_1, "score_b": score_2, "reasoning": judged.get("reasoning", "")}
    else:
        return{"score_a": score_2, "score_b": score_1, "reasoning": judged.get("reasoning", "")}


def run_comparison(variant_a_system_message: str, variant_b_system_message: str, eval_prompts: list[dict], min_comparisons: int = 5) -> dict:
    scores_a = []
    scores_b = []
    per_prompt_results = []
    skipped = 0

    for item in eval_prompts:
        user_message = build_user_message(item["prompt"], item["goal"])
        
        try:
            output_a = call_llm(item["prompt"], variant_a_system_message, user_message)
            output_b = call_llm(item["prompt"], variant_b_system_message, user_message)

            scores = judge_output(item["prompt"], item["goal"], output_a["optimized_prompt"], output_b["optimized_prompt"])
        except Exception as e:
            skipped += 1
            continue

        scores_a.append(scores["score_a"])
        scores_b.append(scores["score_b"]) 
        per_prompt_results.append({
            "prompt": item["prompt"],
            "goal": item["goal"],
            "score_a": scores["score_a"],
            "score_b": scores["score_b"],
            "reasoning": scores["reasoning"],
        })

    if len(scores_a) < min_comparisons:
        raise ValueError(f"Only {len(scores_a)} successful comparison - not enough data for a meaningful test.")

    statistic, p_value = wilcoxon(scores_b, scores_a)
    p_value = cast(float, p_value)  

    differences = [b - a for a, b in zip(scores_a, scores_b)]
    median_diff = statistics.median(differences)

    if p_value < 0.05:
        conclusion = f"""Statistically significant difference (p={p_value:.4f}). Median improvement: {median_diff:+.1f} points."""
    else:
        conclusion = f"""No statistically significant difference detected (p={p_value:.4f})."""

    return {
        "p_value": p_value,
        "median_difference": median_diff,
        "n_compared": len(scores_a),
        "n_skipped": skipped,
        "conclusion": conclusion,
        "per_prompt_results": per_prompt_results,
    }



def judge_output_mock(prompt: str, goal: str, output_a: str, output_b: str) -> dict:
    """
    Mock call for testing purposes.
    """
    return {
        "score_a": 5.3,
        "score_b": 8.8,
        "reasoning": "Reasoning"
    } 
