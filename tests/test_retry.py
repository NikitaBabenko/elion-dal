"""Юнит-тесты retry-утилиты (offline, без реальных задержек)."""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from elion_dal.util.retry import call_with_retry, is_transient


def _fixed_rng(_a: float, b: float) -> float:
    # Детерминированная «задержка» = верхняя граница (jitter off для тестов).
    return b


class _Counter:
    def __init__(self):
        self.sleeps: list[float] = []

    def sleep(self, d: float) -> None:
        self.sleeps.append(d)


def _make_failing(n_fail: int, exc: Exception, result: str = "ok"):
    """fn, который падает первые n_fail вызовов, потом возвращает result."""
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= n_fail:
            raise exc
        return result

    fn.state = state  # type: ignore[attr-defined]
    return fn


def test_succeeds_after_transient_failures():
    c = _Counter()
    fn = _make_failing(2, UnexpectedResponse(500, "ISE", b"boom", {}))
    out = call_with_retry(fn, attempts=3, base_delay_s=0.5, sleep=c.sleep, rng=_fixed_rng)
    assert out == "ok"
    assert fn.state["calls"] == 3
    assert len(c.sleeps) == 2  # два ретрая = два сна


def test_exhausts_and_raises_last():
    c = _Counter()
    exc = UnexpectedResponse(500, "ISE", b"boom", {})
    fn = _make_failing(99, exc)  # всегда падает
    with pytest.raises(UnexpectedResponse):
        call_with_retry(fn, attempts=3, base_delay_s=0.5, sleep=c.sleep, rng=_fixed_rng)
    assert fn.state["calls"] == 3
    assert len(c.sleeps) == 2  # attempts-1 снов


def test_does_not_retry_non_transient():
    c = _Counter()
    fn = _make_failing(99, ValueError("logic bug"))
    with pytest.raises(ValueError):
        call_with_retry(fn, attempts=5, base_delay_s=0.5, sleep=c.sleep, rng=_fixed_rng)
    assert fn.state["calls"] == 1  # без ретраев
    assert c.sleeps == []


def test_response_handling_exception_is_transient():
    c = _Counter()
    fn = _make_failing(1, ResponseHandlingException(TimeoutError("read timeout")))
    out = call_with_retry(fn, attempts=2, base_delay_s=0.1, sleep=c.sleep, rng=_fixed_rng)
    assert out == "ok"
    assert len(c.sleeps) == 1


def test_attempts_one_means_no_retry():
    c = _Counter()
    fn = _make_failing(99, UnexpectedResponse(503, "SU", b"x", {}))
    with pytest.raises(UnexpectedResponse):
        call_with_retry(fn, attempts=1, base_delay_s=0.5, sleep=c.sleep, rng=_fixed_rng)
    assert fn.state["calls"] == 1
    assert c.sleeps == []


def test_on_retry_callback_invoked():
    c = _Counter()
    seen: list[int] = []
    fn = _make_failing(2, UnexpectedResponse(500, "ISE", b"x", {}))
    call_with_retry(
        fn, attempts=3, base_delay_s=0.1, sleep=c.sleep, rng=_fixed_rng,
        on_retry=lambda attempt, _e: seen.append(attempt),
    )
    assert seen == [1, 2]


def test_is_transient_classification():
    assert is_transient(UnexpectedResponse(500, "ISE", b"x", {}))
    assert is_transient(ResponseHandlingException(OSError("net")))
    assert is_transient(ConnectionError("down"))
    assert not is_transient(ValueError("bug"))
    assert not is_transient(KeyError("missing"))


def test_invalid_attempts_raises():
    with pytest.raises(ValueError):
        call_with_retry(lambda: "x", attempts=0, base_delay_s=0.5)
