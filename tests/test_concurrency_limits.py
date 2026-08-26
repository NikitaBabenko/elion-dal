"""Пределы параллелизма DAL.

Ручки REST объявлены обычными `def`, поэтому FastAPI выполняет их в пуле
потоков anyio. Это единственное место во всём стеке, где запросы разных
пользователей обрабатываются настоящими потоками, — и дефолты там неудачные:
40 потоков дерутся за ядра на CPU-bound эмбеддинге и за 15 соединений к PG.
Здесь проверяется, что оба предела заданы явно и согласованы.
"""

from __future__ import annotations

import asyncio

import anyio.to_thread
import pytest
from fastapi.testclient import TestClient

from elion_dal.config import Settings
from elion_dal.service.rest_api import create_api
from elion_dal.store.pg_repo import PgRepo

from .test_rest_api import FakeIndex


def test_pg_pool_is_configurable() -> None:
    repo = PgRepo(
        "postgresql+psycopg://elion:secret@localhost:5432/elion",
        pool_size=9,
        max_overflow=4,
    )
    try:
        assert repo.engine.pool.size() == 9
        # Публичного геттера у QueuePool нет, а проверить переполнение надо:
        # именно оно задаёт пиковое число соединений.
        assert repo.engine.pool._max_overflow == 4
    finally:
        repo.engine.dispose()


def test_pg_pool_defaults_match_settings() -> None:
    """Пул соединений не должен быть уже пула потоков REST."""
    settings = Settings()

    assert settings.pg_pool_size >= settings.rest_workers - settings.pg_max_overflow


def test_sqlite_ignores_pool_arguments() -> None:
    """SQLite (локальный режим) собирается на пуле без очереди."""
    repo = PgRepo("sqlite+pysqlite:///:memory:", pool_size=9, max_overflow=4)
    repo.engine.dispose()


def _limit_seen_during_lifespan(rest_workers: int) -> float:
    """Предел пула потоков, как его видит приложение на своём event loop.

    Замер идёт изнутри: `current_default_thread_limiter()` живёт в RunVar,
    то есть у каждого цикла событий он свой — ровно поэтому предел и
    выставляется в lifespan, а не на импорте модуля.
    """

    async def run() -> float:
        api = create_api(FakeIndex(), Settings(rest_workers=rest_workers))
        async with api.router.lifespan_context(api):
            return anyio.to_thread.current_default_thread_limiter().total_tokens

    return asyncio.run(run())


@pytest.mark.parametrize("workers", [1, 5, 16])
def test_rest_thread_pool_is_limited_by_settings(workers: int) -> None:
    """Дефолт anyio (40) здесь не пропускная способность, а конкуренция:
    горячий участок поиска CPU-bound, лишние потоки только дерутся за ядра."""
    assert _limit_seen_during_lifespan(workers) == workers


def test_app_still_serves_requests_with_the_limit_applied() -> None:
    with TestClient(create_api(FakeIndex(), Settings(rest_workers=2))) as client:
        assert client.get("/healthz").status_code == 200
