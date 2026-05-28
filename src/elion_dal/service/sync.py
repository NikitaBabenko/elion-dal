"""Оркестрация индексации: документ -> чанкинг -> эмбеддинг -> Postgres + Qdrant.

Не зависит от gRPC — чистая бизнес-логика, переиспользуется и сервисом, и
сид-утилитой, и CLI. Идемпотентно по content_hash.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..chunking.chunker import Chunker
from ..embedding.base import EmbeddingProvider
from ..store.pg_repo import DocInput, PgRepo, sha256
from ..store.qdrant_repo import PointInput, QdrantRepo, SearchHit


@dataclass(slots=True)
class UpsertCounts:
    received: int = 0
    indexed: int = 0
    skipped: int = 0
    blank: int = 0
    chunks_upserted: int = 0


class IndexService:
    def __init__(
        self,
        pg: PgRepo,
        qdrant: QdrantRepo,
        provider: EmbeddingProvider,
        chunker: Chunker,
    ) -> None:
        self.pg = pg
        self.qdrant = qdrant
        self.provider = provider
        self.chunker = chunker

    def process_document(self, doc: DocInput, counts: UpsertCounts) -> None:
        counts.received += 1
        self.pg.ensure_source(doc.source_id)
        if not doc.content_hash:
            doc.content_hash = sha256(doc.text)

        # Бланк-на-скачивание: храним в SoT, но не индексируем; чистим старые точки.
        if not doc.index_in_rag:
            self.pg.upsert_document(doc)
            self.pg.replace_chunks(doc.doc_id, [])
            self.qdrant.delete_by_doc(doc.doc_id)
            counts.blank += 1
            return

        prev_hash = self.pg.get_content_hash(doc.doc_id)
        if prev_hash is not None and prev_hash == doc.content_hash:
            counts.skipped += 1
            return

        self.pg.upsert_document(doc)
        chunks = self.chunker.split(doc.text)
        self.pg.replace_chunks(doc.doc_id, chunks)
        self.qdrant.delete_by_doc(doc.doc_id)

        if chunks:
            embeddings = self.provider.embed_documents([c.text for c in chunks])
            points = [
                PointInput(
                    doc_id=doc.doc_id,
                    chunk_index=c.index,
                    embedding=emb,
                    payload={
                        "chunk_id": f"{doc.doc_id}:{c.index}",
                        "doc_id": doc.doc_id,
                        "source_id": doc.source_id,
                        "url": doc.url,
                        "title": doc.title,
                        "text": c.text,
                        "published_ts": doc.published_ts,
                        "lang": doc.lang,
                    },
                )
                for c, emb in zip(chunks, embeddings, strict=True)
            ]
            counts.chunks_upserted += self.qdrant.upsert_chunks(points)

        self.pg.touch_source_indexed(doc.source_id)
        counts.indexed += 1

    def search(
        self, query: str, top_k: int, source_ids: list[str], min_published_ts: int
    ) -> list[SearchHit]:
        embedding = self.provider.embed_query(query)
        return self.qdrant.search(embedding, top_k, source_ids, min_published_ts)

    def delete_source(self, source_id: str) -> tuple[int, int]:
        docs, chunks = self.pg.delete_by_source(source_id)
        self.qdrant.delete_by_source(source_id)
        return docs, chunks

    def health(self) -> dict:
        qok = self.qdrant.ping()
        pok = self.pg.ping()
        return {
            "ok": qok and pok,
            "qdrant_ok": qok,
            "postgres_ok": pok,
            "embedding_backend": self.provider.name,
        }
