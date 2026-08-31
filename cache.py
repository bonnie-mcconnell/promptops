import os
import redis
import json
import hashlib
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


from llm import embed_text


redis_client = redis.Redis.from_url(
    url=os.environ.get("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)


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


def _hash_prompt_goal(prompt: str, goal: str) -> str:
    return hashlib.sha256(f"{prompt}|{goal}".encode()).hexdigest()


def cache_key(prompt: str, goal: str) -> str:
    return "opt:" + _hash_prompt_goal(prompt, goal)


def store_semantic_cache(prompt, goal, result):
    embedding = embed_text(prompt, goal)
    key = "semantic:" + _hash_prompt_goal(prompt, goal)
    redis_client.hset(
        key, 
        mapping={
            "prompt_text": prompt + "|" + goal,
            "embedding": embedding,
            "result_json": json.dumps(result)
        },
    )


def semantic_cache_lookup(prompt, goal, threshold=0.15):
    embedding_bytes = embed_text(prompt, goal)

    q = Query(f"*=>[KNN 1 @embedding $vec AS score]").sort_by("score").return_fields("result_json", "score").dialect(2)
    results = redis_client.ft("idx:prompts").search(q, query_params={"vec": embedding_bytes})

    if not results.docs: # type: ignore[attr-defined]
        return None

    best_match = results.docs[0] # type: ignore[attr-defined]
    if float(best_match.score) < threshold:
        return json.loads(best_match.result_json)

    return None

