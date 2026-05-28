"""Доступ к Postgres (source-of-truth): документы и чанки, дедуп по content_hash."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from ..chunking.chunker import Chunk as TextChunk
from .models import Base, Chunk, Document, Source, chunk_id


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DocInput:
    doc_id: str
    source_id: str
    url: str
    title: str
    text: str
    lang: str
    published_ts: int
    content_hash: str
    index_in_rag: bool


class PgRepo:
    def __init__(self, dsn: str) -> None:
        self.engine = create_engine(dsn, pool_pre_ping=True, future=True)
        # SQLite (локальный режим без Postgres) не включает внешние ключи по
        # умолчанию — без этого не сработает ON DELETE CASCADE для чанков.
        if dsn.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(self.engine, "connect")
            def _fk_on(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        self._sm: sessionmaker[Session] = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        """Создать схему напрямую (для тестов; в проде — alembic)."""
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def ensure_source(self, source_id: str, name: str | None = None) -> None:
        with self._sm.begin() as s:
            if s.get(Source, source_id) is None:
                s.add(Source(source_id=source_id, name=name or source_id))

    def get_content_hash(self, doc_id: str) -> str | None:
        with self._sm() as s:
            return s.execute(
                select(Document.content_hash).where(Document.doc_id == doc_id)
            ).scalar_one_or_none()

    def upsert_document(self, doc: DocInput) -> None:
        with self._sm.begin() as s:
            existing = s.get(Document, doc.doc_id)
            if existing is None:
                s.add(
                    Document(
                        doc_id=doc.doc_id,
                        source_id=doc.source_id,
                        url=doc.url,
                        title=doc.title,
                        lang=doc.lang,
                        published_ts=doc.published_ts,
                        content_hash=doc.content_hash,
                        raw_text=doc.text,
                        index_in_rag=doc.index_in_rag,
                    )
                )
            else:
                existing.source_id = doc.source_id
                existing.url = doc.url
                existing.title = doc.title
                existing.lang = doc.lang
                existing.published_ts = doc.published_ts
                existing.content_hash = doc.content_hash
                existing.raw_text = doc.text
                existing.index_in_rag = doc.index_in_rag

    def replace_chunks(self, doc_id: str, chunks: Sequence[TextChunk]) -> None:
        with self._sm.begin() as s:
            s.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
            for c in chunks:
                s.add(
                    Chunk(
                        chunk_id=chunk_id(doc_id, c.index),
                        doc_id=doc_id,
                        chunk_index=c.index,
                        text=c.text,
                        token_count=c.token_count,
                        content_hash=sha256(c.text),
                    )
                )

    def touch_source_indexed(self, source_id: str) -> None:
        with self._sm.begin() as s:
            src = s.get(Source, source_id)
            if src is not None:
                src.last_indexed_at = func.now()

    def delete_by_source(self, source_id: str) -> tuple[int, int]:
        """Удалить документы и чанки источника. Возвращает (docs, chunks)."""
        with self._sm.begin() as s:
            doc_ids = list(
                s.execute(select(Document.doc_id).where(Document.source_id == source_id)).scalars()
            )
            chunks_n = (
                s.execute(
                    select(func.count()).select_from(Chunk).where(Chunk.doc_id.in_(doc_ids))
                ).scalar_one()
                if doc_ids
                else 0
            )
            # ON DELETE CASCADE удалит чанки вместе с документами.
            s.execute(delete(Document).where(Document.source_id == source_id))
            return len(doc_ids), int(chunks_n)
