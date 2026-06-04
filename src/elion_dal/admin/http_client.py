"""HTTP-клиент локальной админки к серверу (REST API).

Совместимая «обёртка»: те же методы, что использовала web.py от IndexService /
GrpcAdminClient — get_stats/list_sources/search/delete_*/settings_view/
update_settings/process_document, но HTTP вместо gRPC.

(GrpcAdminClient оставлен в репозитории до момента, когда платформа научится
проксировать gRPC — см. ADR-006.)
"""

from __future__ import annotations

import httpx

from ..service.sync import ParentHit
from ..store.pg_repo import DocInput, SourceStats, StoreStats
from ..store.settings_store import SettingView


def _typed(field_type: str, raw):
    if field_type == "int":
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0
    if field_type == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return 0.0
    if field_type == "bool":
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return raw


class HttpAdminClient:
    def __init__(self, base_url: str, token: str = "", verify: bool = True) -> None:
        if not base_url:
            raise ValueError(
                "API_BASE_URL не задан: укажите URL сервера, напр. https://elion-dal.vibenest.net"
            )
        self.base_url = base_url.rstrip("/")
        self.token = token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=30.0, verify=verify
        )

    # ---------- статистика / источники ----------
    def get_stats(self) -> StoreStats:
        r = self._client.get("/api/v1/stats")
        r.raise_for_status()
        data = r.json()
        sources = [
            SourceStats(
                source_id=s["source_id"],
                name=s["name"],
                last_indexed_ts=s["last_indexed_ts"],
                document_count=s["document_count"],
                parent_count=s["parent_count"],
                chunk_count=s["chunk_count"],
            )
            for s in data.get("sources", [])
        ]
        return StoreStats(
            total_documents=data["total_documents"],
            total_parents=data["total_parents"],
            total_chunks=data["total_chunks"],
            sources=sources,
        )

    def list_sources(self) -> list[SourceStats]:
        return self.get_stats().sources

    # ---------- поиск ----------
    def search(
        self, query: str, top_k: int, source_ids: list[str], min_published_ts: int
    ) -> list[ParentHit]:
        r = self._client.post(
            "/api/v1/search",
            json={
                "query": query,
                "top_k": top_k,
                "source_ids": list(source_ids),
                "min_published_ts": min_published_ts,
            },
        )
        r.raise_for_status()
        return [
            ParentHit(
                parent_id=h["parent_id"],
                doc_id=h["doc_id"],
                source_id=h["source_id"],
                title=h["title"],
                url=h["url"],
                heading_path=list(h.get("heading_path", [])),
                text=h["text"],
                matched_child=h["matched_child"],
                score=h["score"],
                dense_score=h["dense_score"],
            )
            for h in r.json().get("hits", [])
        ]

    # ---------- удаление ----------
    def delete_source(self, source_id: str) -> tuple[int, int]:
        r = self._client.delete(f"/api/v1/sources/{source_id}")
        r.raise_for_status()
        d = r.json()
        return d["documents_deleted"], d["chunks_deleted"]

    def delete_doc(self, doc_id: str) -> tuple[int, int]:
        r = self._client.delete(f"/api/v1/documents/{doc_id}")
        r.raise_for_status()
        d = r.json()
        return d["documents_deleted"], d["chunks_deleted"]

    # ---------- просмотр документов и чанков ----------
    def list_documents(self, source_id: str = "") -> list[dict]:
        params = {"source_id": source_id} if source_id else {}
        r = self._client.get("/api/v1/documents", params=params)
        r.raise_for_status()
        return r.json().get("documents", [])

    def get_document_detail(self, doc_id: str) -> dict:
        r = self._client.get(f"/api/v1/documents/{doc_id}/detail")
        r.raise_for_status()
        return r.json()

    def preview_chunking(
        self,
        text: str,
        chunk_tokens: int | None = None,
        chunk_overlap: int | None = None,
        min_tokens: int | None = None,
        separator_mode: str | None = None,
    ) -> dict:
        r = self._client.post(
            "/api/v1/chunk-preview",
            json={
                "text": text,
                "chunk_tokens": chunk_tokens,
                "chunk_overlap": chunk_overlap,
                "min_tokens": min_tokens,
                "separator_mode": separator_mode,
            },
        )
        r.raise_for_status()
        return r.json()

    # ---------- настройки ----------
    def settings_view(self) -> list[SettingView]:
        r = self._client.get("/api/v1/settings")
        r.raise_for_status()
        return [
            SettingView(
                key=f["key"],
                label=f["label"],
                tier=f["tier"],
                type=f["type"],
                value=_typed(f["type"], f["value"]),
                is_override=f["is_override"],
            )
            for f in r.json().get("fields", [])
        ]

    def update_settings(self, items: dict[str, str]) -> None:
        r = self._client.post("/api/v1/settings", json={"items": items})
        r.raise_for_status()

    # ---------- индексация (upload) ----------
    def process_document(self, doc: DocInput, counts) -> None:  # counts оставлен для совместимости
        payload = {
            "doc_id": doc.doc_id,
            "source_id": doc.source_id,
            "url": doc.url,
            "title": doc.title,
            "lang": doc.lang,
            "published_ts": doc.published_ts,
            "content_hash": doc.content_hash,
            "index_in_rag": doc.index_in_rag,
            "sections": [
                {
                    "section_id": s.section_id,
                    "heading_path": list(s.heading_path),
                    "url": s.url,
                    "text": s.text,
                    "published_ts": s.published_ts,
                    "content_hash": s.content_hash,
                }
                for s in doc.sections
            ],
        }
        r = self._client.post("/api/v1/documents", json=payload)
        r.raise_for_status()

    def close(self) -> None:
        self._client.close()
