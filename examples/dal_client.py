"""Тонкий HTTP-клиент DAL REST API — общая часть демо-микросервисов.

Используется writer_service и reader_service. Скрывает httpx, Bearer-токен и
обработку ошибок; возвращает уже распарсенные dict-ы.

Конфиг через env (или явные аргументы конструктора):
- DAL_BASE_URL  (по умолчанию https://elion-dal.vibenest.net)
- DAL_API_TOKEN (по умолчанию пусто — ручки должны быть открыты на стороне DAL)
- DAL_TIMEOUT   (секунды, по умолчанию 30)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Section:
    """Одна секция документа (parent в parent-child retrieval).

    section_id уникален внутри документа; heading_path = крошки заголовка
    («Положения » «Глава 1»), которые отображаются в выдаче.
    """

    section_id: str
    text: str
    heading_path: list[str] | None = None
    url: str = ""
    published_ts: int = 0


class DalError(RuntimeError):
    """Любая ошибка взаимодействия с DAL (HTTP != 2xx или сеть)."""

    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class DalClient:
    """Минимальный sync-клиент DAL REST API.

    На каждый запрос подкладывает Authorization: Bearer <token>, если токен задан.
    Ошибки HTTP оборачиваются в DalError (с .status и .body для логов вызывающего).
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or os.environ.get("DAL_BASE_URL", "https://elion-dal.vibenest.net")).rstrip("/")
        self.token = token if token is not None else os.environ.get("DAL_API_TOKEN", "")
        timeout = timeout if timeout is not None else float(os.environ.get("DAL_TIMEOUT", "30"))
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    # --- lifecycle ---
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> DalClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- helpers ---
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise DalError(f"network: {e}") from e
        if not r.is_success:
            raise DalError(
                f"{method} {path} -> HTTP {r.status_code}",
                status=r.status_code,
                body=r.text[:500],
            )
        if r.status_code == 204 or not r.content:
            return None
        ctype = r.headers.get("content-type", "")
        return r.json() if "json" in ctype else r.text

    # --- эндпоинты ---
    def healthz(self) -> dict:
        return self._request("GET", "/healthz")

    def get_stats(self) -> dict:
        return self._request("GET", "/api/v1/stats")

    def list_sources(self) -> list[dict]:
        data = self._request("GET", "/api/v1/sources")
        return data.get("sources", [])

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_ids: list[str] | None = None,
        min_published_ts: int = 0,
    ) -> list[dict]:
        payload = {
            "query": query,
            "top_k": top_k,
            "source_ids": source_ids or [],
            "min_published_ts": min_published_ts,
        }
        data = self._request("POST", "/api/v1/search", json=payload)
        return data.get("hits", [])

    def upsert_document(
        self,
        *,
        doc_id: str,
        source_id: str,
        title: str = "",
        url: str = "",
        lang: str = "ru",
        published_ts: int = 0,
        content_hash: str = "",
        index_in_rag: bool = True,
        sections: list[Section] | None = None,
        text: str = "",
    ) -> dict:
        """Upsert одного документа.

        Если sections не задан, но передан text — на стороне DAL создастся одна
        синтетическая секция (fallback). Для богатого контекста выдачи лучше
        всегда отдавать sections с осмысленным heading_path.
        """
        payload = {
            "doc_id": doc_id,
            "source_id": source_id,
            "url": url,
            "title": title,
            "lang": lang,
            "published_ts": published_ts,
            "content_hash": content_hash,
            "index_in_rag": index_in_rag,
            "text": text,
            "sections": [
                {
                    "section_id": s.section_id,
                    "heading_path": list(s.heading_path or []),
                    "url": s.url or url,
                    "text": s.text,
                    "published_ts": s.published_ts or published_ts,
                }
                for s in (sections or [])
            ],
        }
        return self._request("POST", "/api/v1/documents", json=payload)

    def delete_source(self, source_id: str) -> dict:
        return self._request("DELETE", f"/api/v1/sources/{source_id}")

    def delete_doc(self, doc_id: str) -> dict:
        return self._request("DELETE", f"/api/v1/documents/{doc_id}")

    def get_settings(self) -> list[dict]:
        data = self._request("GET", "/api/v1/settings")
        return data.get("fields", [])

    def update_settings(self, items: dict[str, str]) -> list[dict]:
        data = self._request("POST", "/api/v1/settings", json={"items": items})
        return data.get("fields", [])
