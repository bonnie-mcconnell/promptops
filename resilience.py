from typing import Optional
import threading
import time
import random
import logging

logger = logging.getLogger(__name__)

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
