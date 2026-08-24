from openai import OpenAI
import os
import json
import numpy as np
import logging

from resilience import CircuitBreaker, call_with_retries


logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

judge_client = OpenAI(
     api_key = os.environ.get("OPENAI_API_KEY"),
     base_url="https://api.openai.com/v1"
)


llm_breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
embedding_breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)


def build_user_message(prompt: str, goal: str) -> str:
     return f"""Original prompt: {prompt}
Goal: {goal}

Optimize this prompt to better achieve the stated goal."""

def call_llm(prompt: str, system_message: str, user_message: str):
    def _call():
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
            logger.warning(f"LLM returned invalid JSON for prompt: {prompt[:50]}...")
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

    return llm_breaker.call(lambda: call_with_retries(_call))


def embed_text(prompt: str, goal: str) -> bytes:
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
