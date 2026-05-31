"""Тонкий gRPC-клиент для локальной админки.

Экспортирует методы с теми же сигнатурами, что админский веб-UI ранее звал у
IndexService (get_stats / list_sources / search / delete_source / delete_doc /
settings_view / update_settings / process_document), но под капотом — gRPC.

Это позволяет запускать админку отдельным процессом (локально), указывая
GRPC_TARGET + API_TOKEN, без поднятия всей модели рядом.
"""

from __future__ import annotations

import grpc

from ..grpc_gen import vectorstore_pb2 as pb
from ..grpc_gen import vectorstore_pb2_grpc as pb_grpc
from ..service.sync import ParentHit
from ..store.pg_repo import DocInput, SourceStats, StoreStats
from ..store.settings_store import SettingView


def _channel(target: str, insecure: bool) -> grpc.Channel:
    if insecure or target.startswith(("localhost", "127.", "0.0.0.0")):
        return grpc.insecure_channel(target)
    return grpc.secure_channel(target, grpc.ssl_channel_credentials())


def _typed(field_type: str, raw: str):
    """Конвертация значения настройки из строки (как приходит по gRPC) в нужный тип."""
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


class GrpcAdminClient:
    """Совместимая «обёртка»: те же методы, что нужны admin/web.py от IndexService."""

    def __init__(self, target: str, token: str = "", insecure: bool = False) -> None:
        if not target:
            raise ValueError("GRPC_TARGET не задан: укажите host:port удалённого сервера")
        self._channel = _channel(target, insecure)
        self._stub = pb_grpc.VectorStoreStub(self._channel)
        self._meta = (("authorization", f"Bearer {token}"),) if token else ()
        self.target = target

    # --- статистика / источники ---
    def get_stats(self) -> StoreStats:
        resp = self._stub.GetStats(pb.StatsRequest(), metadata=self._meta)
        sources = [
            SourceStats(
                source_id=s.source_id,
                name=s.name,
                last_indexed_ts=s.last_indexed_ts,
                document_count=s.document_count,
                parent_count=s.parent_count,
                chunk_count=s.chunk_count,
            )
            for s in resp.sources
        ]
        return StoreStats(
            total_documents=resp.total_documents,
            total_parents=resp.total_parents,
            total_chunks=resp.total_chunks,
            sources=sources,
        )

    def list_sources(self) -> list[SourceStats]:
        return self.get_stats().sources

    # --- поиск ---
    def search(
        self, query: str, top_k: int, source_ids: list[str], min_published_ts: int
    ) -> list[ParentHit]:
        req = pb.SearchRequest(
            query=query,
            top_k=top_k,
            source_ids=list(source_ids),
            min_published_ts=min_published_ts,
        )
        resp = self._stub.Search(req, metadata=self._meta)
        return [
            ParentHit(
                parent_id=h.parent_id,
                doc_id=h.doc_id,
                source_id=h.source_id,
                title=h.title,
                url=h.url,
                heading_path=list(h.heading_path),
                text=h.text,
                matched_child=h.matched_child,
                score=h.score,
                dense_score=h.dense_score,
            )
            for h in resp.hits
        ]

    # --- удаление ---
    def delete_source(self, source_id: str) -> tuple[int, int]:
        r = self._stub.DeleteBySource(pb.SourceRef(source_id=source_id), metadata=self._meta)
        return r.documents_deleted, r.chunks_deleted

    def delete_doc(self, doc_id: str) -> tuple[int, int]:
        r = self._stub.DeleteByDoc(pb.DocRef(doc_id=doc_id), metadata=self._meta)
        return r.documents_deleted, r.chunks_deleted

    # --- настройки ---
    def settings_view(self) -> list[SettingView]:
        resp = self._stub.GetSettings(pb.StatsRequest(), metadata=self._meta)
        return [
            SettingView(
                key=f.key,
                label=f.label,
                tier=f.tier,
                type=f.type,
                value=_typed(f.type, f.value),
                is_override=f.is_override,
            )
            for f in resp.fields
        ]

    def update_settings(self, items: dict[str, str]) -> None:
        self._stub.UpdateSettings(pb.SettingsUpdate(items=items), metadata=self._meta)

    # --- индексация документа (для загрузки PDF/DOCX из админки) ---
    def process_document(self, doc: DocInput, counts) -> None:  # counts оставлен для совместимости
        def _gen():
            yield pb.Document(
                doc_id=doc.doc_id,
                source_id=doc.source_id,
                url=doc.url,
                title=doc.title,
                lang=doc.lang,
                published_ts=doc.published_ts,
                content_hash=doc.content_hash,
                index_in_rag=doc.index_in_rag,
                sections=[
                    pb.Section(
                        section_id=s.section_id,
                        heading_path=list(s.heading_path),
                        url=s.url,
                        text=s.text,
                        published_ts=s.published_ts,
                        content_hash=s.content_hash,
                    )
                    for s in doc.sections
                ],
            )

        self._stub.UpsertDocuments(_gen(), metadata=self._meta)

    def close(self) -> None:
        self._channel.close()
