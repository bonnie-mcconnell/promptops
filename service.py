from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List
import json
import os
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import hashlib
import time
import random
import threading
import logging
import numpy as np


load_dotenv()

logger = logging.getLogger(__name__)


redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "redis"), 
    port=6379,
    decode_responses=True
)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)


class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_seconds=30):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.state = "closed"
        self.opened_at: Optional[float] = None
        self.lock = threading.Lock()

    def call(self, func):
        with self.lock:
            if self.state == "open":
                assert self.opened_at is not None
                if time.time() - self.opened_at >= self.cooldown_seconds:
                    self.state = "half_open"
                else:
                    raise CircuitOpenError("Circuit is open, failing fast")

        try:
            result = func()
        except Exception:
            with self.lock:
                self.failure_count += 1
                if self.state == "half_open" or self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    self.opened_at = time.time()
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            raise
        else:
            with self.lock:
                self.failure_count = 0
                self.state = "closed"
            return result
                

class CircuitOpenError(Exception):
    pass


llm_breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
embedding_breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)


# retry logic thats used to call_llm
def call_with_retries(func, max_attempts=3, base_delay=0.5):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            time.sleep(delay)
    raise RuntimeError("call_with_retries called with max_attempts <= 0")


def create_vector_index():
    try:
        redis_client.ft("idx:prompts").info()
        return
    except Exception:
        pass

    schema = [
        TextField("prompt_text"),
        VectorField(
            "embedding", 
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": 1536,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    ]

    redis_client.ft("idx:prompts").create_index(
        schema,
        definition=IndexDefinition(prefix=["semantic:"], index_type=IndexType.HASH),
    )

def _cache_key(prompt: str, goal: str) -> str:
    return "opt:" + hashlib.sha256(f"{prompt}|{goal}".encode()).hexdigest()


def _embed_text(prompt: str, goal: str) -> bytes:
    """Call OpenAI's vector embedding endpoint and return a vector."""
    def _call_embeddings_api():
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=prompt + "|" + goal,
        )
        embedding_list = response.data[0].embedding
        if not embedding_list:
                raise ValueError("OpenAI returned an empty response body.")

        return embedding_list

    embedding_list = embedding_breaker.call(lambda: call_with_retries(_call_embeddings_api))
    return np.array(embedding_list, dtype=np.float32).tobytes()


def _store_semantic_cache(prompt, goal, result):
    embedding = _embed_text(prompt, goal)
    key = "semantic:" + hashlib.sha256(f"{prompt}|{goal}".encode()).hexdigest()
    redis_client.hset(
        key, 
        mapping={
            "prompt_text": prompt + "|" + goal,
            "embedding": embedding,
            "result_json": json.dumps(result)
        },
    )


def _semantic_cache_lookup(prompt, goal, threshold=0.15):
    embedding_bytes = _embed_text(prompt, goal)

    q = Query(f"*=>[KNN 1 @embedding $vec AS score]").sort_by("score").return_fields("result_json", "score").dialect(2)
    results = redis_client.ft("idx:prompts").search(q, query_params={"vec": embedding_bytes})

    if not results.docs: # type: ignore[attr-defined]
        return None

    best_match = results.docs[0] # type: ignore[attr-defined]
    if float(best_match.score) < threshold:
        return json.loads(best_match.result_json)

    return None


def optimize_prompt(prompt: str, goal: str) -> dict:
    """Send a prompt to the LLM for optimization and return structured results."""

    key = _cache_key(prompt, goal)
    cached = redis_client.get(key)
    if cached:
         return json.loads(cached)

    sem_cached = _semantic_cache_lookup(prompt, goal)
    if sem_cached:
        return sem_cached

    system_message = """You are a prompt engineering expert. Your job is to improve 
    prompts so they produce better results from language models.
    
    You will receive an original prompt and a goal describing what the prompt should 
    accomplish.
    Return your response as a JSON object with exactly these fields:
    - "optimized_prompt": the improved version of the prompt
    - "changes": a brief explanation of what you improved and why
    
    Return ONLY the JSON object. No markdown formatting, no extra text."""

    user_message = f"""Original prompt: {prompt}
    Goal: {goal}

    Optimize this prompt to better achieve the stated goal.
    """

    def _call_llm():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        result_text = response.choices[0].message.content

        if not result_text:
                raise ValueError("OpenAI returned an empty response body.")
        
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            raise ValueError(
                f"LLM returned invalid JSON: {result_text[:200]}"
            )

        if "optimized_prompt" not in result or "changes" not in result:
            raise ValueError(f"LLM response missing required fields. Got: {list(result.keys())}")
    
        result = {
            "original_prompt": prompt,
            "optimized_prompt": result["optimized_prompt"],
            "changes": result["changes"]
        }

        return result

    result = llm_breaker.call(lambda: call_with_retries(_call_llm))
    redis_client.set(key, json.dumps(result), ex=3600)
    _store_semantic_cache(prompt, goal, result)

    return result


def optimize_prompt_mock(prompt: str, goal: str) -> dict:
    """Mock implementation for testing FastAPI routes without API credits."""
    key = _cache_key(prompt, goal)
    cached = redis_client.get(key)
    if cached:
         return json.loads(cached)
    result = {
        "original_prompt": prompt,
        "optimized_prompt": f"Act as an expert. {prompt}. Goal: {goal}",
        "changes": "Added persona framing, context structure, and clear constraints."
    }

    redis_client.set(key, json.dumps(result), ex=3600)
    return result