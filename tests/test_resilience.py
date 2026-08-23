import pytest
import time

from conftest import raise_connection_error
from resilience import CircuitBreaker, CircuitOpenError

def test_breaker_opens_after_threshold_failures():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=5)

    def always_fails():
        raise ConnectionError("simulated LLM outage")

    for _ in range(3):
        with pytest.raises(ConnectionError):
            breaker.call(always_fails)

    assert breaker.state == "open"


def test_breaker_fails_fast_while_open():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5)

    with pytest.raises(ConnectionError):
        breaker.call(raise_connection_error)

    assert breaker.state == "open"

    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: 1 / 0) # should never actually execute


def test_breaker_recovers_after_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=1)

    with pytest.raises(ConnectionError):
        breaker.call(lambda: (_ for _ in ()).throw(ConnectionError))
    assert breaker.state == "open"

    time.sleep(1.1)
    result = breaker.call(lambda: "success")

    assert result == "success"
    assert breaker.state == "closed"

