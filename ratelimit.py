import redis
import time
from fastapi import HTTPException, Depends

from cache import redis_client
from auth import api_key_header


RATE_LIMIT_PER_MINUTE = 20
WINDOW_SECONDS = 60


def check_rate_limit(api_key: str = Depends(api_key_header)) -> None:
    window = int(time.time() // WINDOW_SECONDS)
    key = f"ratelimit:{api_key}:{window}"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, WINDOW_SECONDS)
    if count > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")
