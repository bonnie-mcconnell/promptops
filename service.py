import json

from cache import cache_key, semantic_cache_lookup, store_semantic_cache, redis_client
from llm import call_llm, build_user_message


def optimize_prompt(prompt: str, goal: str) -> tuple[dict, str]:
    """Send a prompt to the LLM for optimization and return structured results."""

    key = cache_key(prompt, goal)
    cached = redis_client.get(key)
    if cached:
         return json.loads(cached), "exact"
    

    sem_cached = semantic_cache_lookup(prompt, goal)
    if sem_cached:
        return sem_cached, "semantic"

    system_message = """You are a prompt engineering expert. Your job is to improve 
    prompts so they produce better results from language models.
    
    You will receive an original prompt and a goal describing what the prompt should 
    accomplish.
    Return your response as a JSON object with exactly these fields:
    - "optimized_prompt": the improved version of the prompt
    - "changes": a brief explanation of what you improved and why
    
    Return ONLY the JSON object. No markdown formatting, no extra text."""

    user_message = build_user_message(prompt, goal)

    result = call_llm(prompt, system_message, user_message)
    redis_client.set(key, json.dumps(result), ex=3600)
    store_semantic_cache(prompt, goal, result)

    return result, "none"


def optimize_prompt_mock(prompt: str, goal: str) -> tuple[dict, str]:
    """Mock implementation for testing FastAPI routes without API credits."""
    key = cache_key(prompt, goal)
    cached = redis_client.get(key)
    if cached:
         return json.loads(cached), "exact"
    
    result = {
        "original_prompt": prompt,
        "optimized_prompt": f"Act as an expert. {prompt}. Goal: {goal}",
        "changes": "Added persona framing, context structure, and clear constraints."
    }

    redis_client.set(key, json.dumps(result), ex=3600)
    return result, "none"