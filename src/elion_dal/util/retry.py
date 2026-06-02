"""Ретраи с экспоненциальным backoff + full jitter для транзиентных сбоев Qdrant.

Используется в `store/qdrant_repo.py` вокруг сетевых вызовов (search/dense_scores/
upsert/delete). Лечит ТРАНЗИЕНТНЫЕ сбои (сетевое моргание, read/connect timeout,
кратковременная перегрузка/недоступность Qdrant). НЕ лечит детерминированные ошибки
(битый сегмент, неверный запрос) — такие пробрасываются сразу или после исчерпания
попыток, чтобы сервис честно отдал ошибку, а не висел вечно.

`sleep` и `rng` инъектируются — для детерминированных тестов без реальных задержек.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

# ApiException — базовый: покрывает UnexpectedResponse (HTTP 5xx) и
# ResponseHandlingException (обёрнутый transport-сбой).
from qdrant_client.http.exceptions import ApiException

logger = logging.getLogger(__name__)

# Потолок задержки между попытками (сек) — чтобы backoff не разрастался бесконечно.
RETRY_DELAY_CAP_S = 10.0

# Транзиентные исключения: на них ретраим. ApiException ловит и UnexpectedResponse
# (HTTP 5xx от Qdrant — в т.ч. RocksDB IO error -> 500), и ResponseHandlingException
# (обёрнутые httpx connect/read timeout, сетевые сбои). Плюс стандартные транспортные
# на случай embedded-режима или прямого сокета.
TRANSIENT_EXC: tuple[type[BaseException], ...] = (
    ApiException,
    ConnectionError,
    TimeoutError,
    OSError,
)


def is_transient(
    exc: BaseException,
    transient: tuple[type[BaseException], ...] = TRANSIENT_EXC,
) -> bool:
    """True, если исключение стоит ретраить (транзиентный сбой бэкенда)."""
    return isinstance(exc, transient)


def _backoff_delay(
    attempt: int,
    base_delay_s: float,
    rng: Callable[[float, float], float],
) -> float:
    """Экспонента с full jitter: uniform(0, min(cap, base * 2**(attempt-1)))."""
    ceiling = min(RETRY_DELAY_CAP_S, base_delay_s * (2 ** (attempt - 1)))
    return rng(0.0, ceiling)


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay_s: float,
    transient: tuple[type[BaseException], ...] = TRANSIENT_EXC,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[float, float], float] = random.uniform,
    on_retry: Callable[[int, BaseException], None] | None = None,
    op_name: str = "qdrant",
) -> T:
    """Вызвать `fn()`; на транзиентном исключении подождать backoff и повторить.

    - `attempts` — всего попыток (>=1; 1 = без ретраев).
    - Не-транзиентные исключения пробрасываются немедленно (не ретраим баги/4xx).
    - После исчерпания попыток пробрасывается последнее транзиентное исключение.
    - `sleep`/`rng` инъектируются для тестов.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — классификацию делает is_transient
            if not is_transient(e, transient) or attempt == attempts:
                raise
            last_exc = e
            delay = _backoff_delay(attempt, base_delay_s, rng)
            if on_retry is not None:
                on_retry(attempt, e)
            else:
                logger.warning(
                    "%s: транзиентный сбой (попытка %d/%d): %s — повтор через %.2fs",
                    op_name,
                    attempt,
                    attempts,
                    type(e).__name__,
                    delay,
                )
            sleep(delay)
    # Недостижимо (последняя попытка либо вернёт, либо пробросит), но для типизации:
    assert last_exc is not None
    raise last_exc
