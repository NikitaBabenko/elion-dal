"""Smoke-тесты REST API через FastAPI TestClient на фейк-индексе."""

from __future__ import annotations

from fastapi.testclient import TestClient

from elion_dal.config import Settings
from elion_dal.service.rest_api import create_api
from elion_dal.service.sync import ParentHit
from elion_dal.store.pg_repo import SourceStats, StoreStats
from elion_dal.store.settings_store import SettingView


class FakeIndex:
    def __init__(self):
        self.deleted_sources = []
        self.deleted_docs = []
        self.processed = []
        self.updated_settings = None
        self.settings_store = None  # без override токена
        self._health = {
            "ok": True,
            "qdrant_ok": True,
            "postgres_ok": True,
            "embedding_backend": "fake",
        }

    def health(self):
        return self._health

    def get_stats(self):
        return StoreStats(2, 3, 9, [SourceStats("s1", "Источник 1", 1700000000, 2, 3, 9)])

    def list_sources(self):
        return self.get_stats().sources

    def search(self, query, top_k, source_ids, min_published_ts):
        return [
            ParentHit(
                parent_id="d1::0", doc_id="d1", source_id="s1", title="t", url="u",
                heading_path=["A"], text="parent", matched_child="child",
                score=0.5, dense_score=0.7,
            )
        ]

    def delete_source(self, source_id):
        self.deleted_sources.append(source_id)
        return 1, 3

    def delete_doc(self, doc_id):
        self.deleted_docs.append(doc_id)
        return 1, 3

    def process_document(self, doc, counts):
        self.processed.append(doc)
        counts.received += 1
        counts.indexed += 1

    def settings_view(self):
        return [
            SettingView("search_parent_fanout", "Fan-out", "live", "int", 5, False),
            SettingView("rerank_enabled", "Реранкер", "live", "bool", False, False),
        ]

    def update_settings(self, items):
        self.updated_settings = items


def app_open():
    return TestClient(create_api(FakeIndex(), Settings()))  # api_token=""


def app_with_token(token="secret"):
    return TestClient(create_api(FakeIndex(), Settings(api_token=token)))


# ---------- /healthz + /readyz ----------
def test_healthz_open():
    r = app_open().get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_ok_when_backends_healthy():
    r = app_open().get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["qdrant_ok"] is True
    assert body["postgres_ok"] is True


def test_readyz_503_when_backend_unhealthy():
    idx = FakeIndex()
    idx._health = {
        "ok": False, "qdrant_ok": False, "postgres_ok": True, "embedding_backend": "fake",
    }
    c = TestClient(create_api(idx, Settings()))
    r = c.get("/readyz")
    assert r.status_code == 503
    assert r.json()["qdrant_ok"] is False


def test_readyz_open_without_token():
    # /readyz должен быть доступен без Bearer даже когда токен задан (проба платформы).
    idx = FakeIndex()
    c = TestClient(create_api(idx, Settings(api_token="secret")))
    r = c.get("/readyz")
    assert r.status_code == 200


# ---------- auth ----------
def test_search_requires_token_when_set():
    c = app_with_token("secret")
    r = c.post("/api/v1/search", json={"query": "q"})
    assert r.status_code == 401


def test_search_ok_with_bearer():
    c = app_with_token("secret")
    r = c.post(
        "/api/v1/search",
        json={"query": "q", "top_k": 3},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200
    assert r.json()["hits"][0]["dense_score"] == 0.7


def test_search_open_when_no_token():
    r = app_open().post("/api/v1/search", json={"query": "q"})
    assert r.status_code == 200


# ---------- ручки ----------
def test_stats_and_sources():
    c = app_open()
    s = c.get("/api/v1/stats").json()
    assert s["total_documents"] == 2
    assert s["total_chunks"] == 9
    assert s["sources"][0]["source_id"] == "s1"
    src = c.get("/api/v1/sources").json()["sources"]
    assert len(src) == 1


def test_delete_source_and_doc():
    c = app_open()
    r1 = c.delete("/api/v1/sources/s1")
    assert r1.status_code == 200
    assert r1.json() == {"documents_deleted": 1, "chunks_deleted": 3}
    r2 = c.delete("/api/v1/documents/d1")
    assert r2.json()["documents_deleted"] == 1


def test_upsert_document_with_sections():
    c = app_open()
    payload = {
        "doc_id": "x1", "source_id": "kb",
        "sections": [{"section_id": "0", "heading_path": [], "url": "u", "text": "hello"}],
    }
    r = c.post("/api/v1/documents", json=payload)
    assert r.status_code == 200
    assert r.json()["indexed"] == 1


def test_upsert_document_fallback_text():
    # sections пустой + text => одна синтетическая секция
    api = create_api(FakeIndex(), Settings())
    c = TestClient(api)
    r = c.post(
        "/api/v1/documents",
        json={"doc_id": "x2", "source_id": "kb", "text": "плоский текст"},
    )
    assert r.status_code == 200


def test_get_and_update_settings():
    api = create_api(FakeIndex(), Settings())
    c = TestClient(api)
    fields = c.get("/api/v1/settings").json()["fields"]
    keys = [f["key"] for f in fields]
    assert "search_parent_fanout" in keys

    r = c.post("/api/v1/settings", json={"items": {"search_parent_fanout": "9"}})
    assert r.status_code == 200
    assert r.json()["fields"]  # снова отдаёт список


def test_admin_mounted_on_main_app():
    """Главный app = REST API + /admin/ — основная конфигурация деплоя."""
    from elion_dal.admin.web import create_app as create_admin_app

    idx = FakeIndex()
    app = create_api(idx, Settings())
    app.mount("/admin", create_admin_app(idx, Settings()))
    c = TestClient(app)

    # /healthz и REST в корне работают
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/v1/stats").status_code == 200

    # Админка отдаёт HTML под /admin/
    r = c.get("/admin/")
    assert r.status_code == 200
    assert "DAL Admin" in r.text
    # Формы используют относительные пути -> чтобы редиректы и POST-ы работали под mount
    assert "action='sources/" in r.text
    assert "action='upload'" in r.text
    assert "action='settings'" in r.text
    assert "fetch('api/search'" in r.text
