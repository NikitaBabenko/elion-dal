"""Пробы здоровья DAL (§2 docs/api/dal).

Liveness и readiness отвечают на разные вопросы: первый — «процесс жив»,
второй — «поиск может отвечать по существу». Их смешение приводит к двум
одинаково плохим исходам: перезапуску исправного процесса из-за недоступного
Qdrant либо трафику на пустой индекс.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from elion_dal.config import Settings
from elion_dal.service.rest_api import create_api


class FakeIndex:
    """Индекс, чьё состояние readiness задаётся тестом."""

    def __init__(self, **overrides) -> None:
        self.settings_store = None
        self._readiness = {
            "ok": True,
            "checks": {
                "qdrant": "ok",
                "postgres": "ok",
                "index": "ok",
                "schema": "ok",
            },
            "embedding_backend": "fake",
            "embedding_dim": 1024,
        }
        self._readiness["checks"].update(overrides)
        self._readiness["ok"] = all(
            value == "ok" for value in self._readiness["checks"].values()
        )

    def health(self) -> dict:
        return {"ok": True, "qdrant_ok": True, "postgres_ok": True}

    def readiness(self) -> dict:
        return self._readiness

    def live_top_k(self) -> int:  # pragma: no cover - не используется пробами
        return 5


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_api(FakeIndex(), Settings()))


def test_live_is_open_and_does_not_touch_backends():
    """Liveness обязан отвечать даже когда всё вокруг лежит."""
    broken = FakeIndex(qdrant="unavailable", postgres="unavailable")

    with TestClient(create_api(broken, Settings())) as client:
        response = client.get("/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "dal"


def test_ready_reports_checks_when_everything_is_up(client: TestClient):
    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["checks"] == {
        "qdrant": "ok",
        "postgres": "ok",
        "index": "ok",
        "schema": "ok",
    }
    assert payload["embedding_backend"] == "fake"


@pytest.mark.parametrize(
    "broken",
    [
        {"qdrant": "unavailable"},
        {"index": "missing"},
        {"index": "empty"},
        {"schema": "mismatch"},
    ],
)
def test_ready_returns_503_for_every_blocking_condition(broken):
    """Пустой индекс и чужая размерность — это «поиск не работает»."""
    with TestClient(create_api(FakeIndex(**broken), Settings())) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_probes_do_not_require_a_token(client: TestClient):
    """Пробы платформы не носят Bearer-токен и не должны его требовать."""
    assert client.get("/live").status_code == 200
    assert client.get("/ready").status_code == 200


def test_legacy_aliases_still_answer(client: TestClient):
    """Прежние пути остаются: на них настроены healthcheck'и стенда."""
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
