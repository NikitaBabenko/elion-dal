"""Оркестрация parent-child: документ -> секции(родители) -> дети -> PG + Qdrant.

Индексируем детей, ищем по детям, но возвращаем РОДИТЕЛЕЙ (схлопывая дублирующие
попадания) — точный матч + богатый контекст для генерации. Идемпотентно по хешу.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..chunking.chunker import Chunker
from ..embedding.base import EmbeddingProvider
from ..embedding.reranker import Reranker
from ..store.models import chunk_id, parent_pk
from ..store.pg_repo import DocInput, ParentBuild, PgRepo, SourceStats, StoreStats, sha256
from ..store.qdrant_repo import PointInput, QdrantRepo


@dataclass(slots=True)
class UpsertCounts:
    received: int = 0
    indexed: int = 0
    skipped: int = 0
    blank: int = 0
    failed: int = 0
    parents_upserted: int = 0
    chunks_upserted: int = 0


@dataclass(slots=True)
class ParentHit:
    parent_id: str
    doc_id: str
    source_id: str
    title: str
    url: str
    heading_path: list[str] = field(default_factory=list)
    text: str = ""
    matched_child: str = ""
    score: float = 0.0
    dense_score: float = 0.0  # raw cosine лучшего ребёнка — сигнал уверенности


class IndexService:
    def __init__(
        self,
        pg: PgRepo,
        qdrant: QdrantRepo,
        provider: EmbeddingProvider,
        chunker: Chunker,
        parent_fanout: int = 5,
        reranker: Reranker | None = None,
        recency_weight: float = 0.0,
        recency_halflife_days: float = 365.0,
    ) -> None:
        self.pg = pg
        self.qdrant = qdrant
        self.provider = provider
        self.chunker = chunker
        # Во сколько раз больше детей тянуть, чтобы схлопнуть в top_k уникальных родителей.
        self.parent_fanout = max(1, parent_fanout)
        self.reranker = reranker
        self.recency_weight = recency_weight
        self.recency_halflife_days = max(1.0, recency_halflife_days)

    def _recency_mult(self, published_ts: int) -> float:
        """Множитель к скору по свежести: до (1 + weight) для свежего, ~1 для старого.
        published_ts=0 (дата неизвестна) -> 1.0 (без штрафа/буста)."""
        if self.recency_weight <= 0 or published_ts <= 0:
            return 1.0
        age_days = max(0.0, (time.time() - published_ts) / 86400.0)
        decay = 0.5 ** (age_days / self.recency_halflife_days)  # (0, 1]
        return 1.0 + self.recency_weight * decay

    def process_document(self, doc: DocInput, counts: UpsertCounts) -> None:
        counts.received += 1
        self.pg.ensure_source(doc.source_id)

        raw_text = "\n\n".join(s.text for s in doc.sections)
        if not doc.content_hash:
            doc.content_hash = sha256(raw_text)

        # Бланк-на-скачивание: храним в SoT, но не индексируем; чистим старые точки.
        if not doc.index_in_rag:
            self.pg.upsert_document(doc, raw_text)
            self.pg.replace_parents_and_chunks(doc.doc_id, [])
            self.qdrant.delete_by_doc(doc.doc_id)
            self.pg.set_content_hash(doc.doc_id, doc.content_hash)
            counts.blank += 1
            return

        prev_hash = self.pg.get_content_hash(doc.doc_id)
        if prev_hash and prev_hash == doc.content_hash:
            counts.skipped += 1
            return

        self.pg.upsert_document(doc, raw_text)

        # Секция -> родитель, текст секции -> дети.
        parents: list[ParentBuild] = []
        for ordinal, section in enumerate(doc.sections):
            section_id = section.section_id or str(ordinal)
            pid = parent_pk(doc.doc_id, section_id)
            children = self.chunker.split(section.text)
            parents.append(
                ParentBuild(
                    parent_id=pid,
                    section_id=section_id,
                    heading_path=section.heading_path,
                    url=section.url or doc.url,
                    text=section.text,
                    token_count=sum(c.token_count for c in children),
                    ordinal=ordinal,
                    children=children,
                )
            )

        self.pg.replace_parents_and_chunks(doc.doc_id, parents)
        self.qdrant.delete_by_doc(doc.doc_id)

        # Эмбеддим всех детей одним батчем.
        texts = [c.text for p in parents for c in p.children]
        if texts:
            embeddings = self.provider.embed_documents(texts)
            points: list[PointInput] = []
            i = 0
            for p in parents:
                for c in p.children:
                    points.append(
                        PointInput(
                            parent_id=p.parent_id,
                            chunk_index=c.index,
                            embedding=embeddings[i],
                            payload={
                                "chunk_id": chunk_id(p.parent_id, c.index),
                                "parent_id": p.parent_id,
                                "doc_id": doc.doc_id,
                                "source_id": doc.source_id,
                                "url": p.url,
                                "title": doc.title,
                                "heading_path": p.heading_path,
                                "text": c.text,
                                "published_ts": doc.published_ts,
                                "lang": doc.lang,
                            },
                        )
                    )
                    i += 1
            counts.chunks_upserted += self.qdrant.upsert_chunks(points)

        # Commit point: фиксируем хеш только теперь — Qdrant уже обновлён.
        self.pg.set_content_hash(doc.doc_id, doc.content_hash)
        counts.parents_upserted += len(parents)
        self.pg.touch_source_indexed(doc.source_id)
        counts.indexed += 1

    def search(
        self, query: str, top_k: int, source_ids: list[str], min_published_ts: int
    ) -> list[ParentHit]:
        embedding = self.provider.embed_query(query)
        limit = top_k * self.parent_fanout
        child_hits = self.qdrant.search(
            embedding, limit=limit, source_ids=source_ids, min_published_ts=min_published_ts
        )

        # Схлопываем детей в уникальных родителей-кандидатов (порядок RRF сохраняем).
        ordered: list[str] = []
        best: dict[str, tuple[float, str, str]] = {}  # pid -> (rrf, child_text, child_chunk_id)
        for h in child_hits:
            if h.parent_id not in best:
                ordered.append(h.parent_id)
                best[h.parent_id] = (h.score, h.text, h.chunk_id)
        if not ordered:
            return []

        # Confidence: сырые dense-косинусы по chunk_id (RRF их не отдаёт).
        dense_map = self.qdrant.dense_scores(
            embedding, limit=limit, source_ids=source_ids, min_published_ts=min_published_ts
        )

        records = self.pg.get_parents(ordered)
        candidates: list[ParentHit] = []
        for pid in ordered:
            rec = records.get(pid)
            if rec is None:
                continue
            rrf, child_text, child_chunk_id = best[pid]
            candidates.append(
                ParentHit(
                    parent_id=rec.parent_id,
                    doc_id=rec.doc_id,
                    source_id=rec.source_id,
                    title=rec.title,
                    url=rec.url,
                    heading_path=rec.heading_path,
                    text=rec.text,
                    matched_child=child_text,
                    score=rrf,
                    dense_score=dense_map.get(child_chunk_id, 0.0),
                )
            )

        # Ранжирование: hybrid (RRF) -> опц. реранкер -> опц. recency -> top_k.
        rescored = False
        if self.reranker is not None:
            scores = self.reranker.rerank(query, [c.text for c in candidates])
            for c, s in zip(candidates, scores, strict=True):
                c.score = s
            rescored = True
        if self.recency_weight > 0:
            for c in candidates:
                c.score *= self._recency_mult(records[c.parent_id].published_ts)
            rescored = True
        if rescored:
            candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def delete_source(self, source_id: str) -> tuple[int, int]:
        docs, chunks = self.pg.delete_by_source(source_id)
        self.qdrant.delete_by_source(source_id)
        return docs, chunks

    def delete_doc(self, doc_id: str) -> tuple[int, int]:
        docs, chunks = self.pg.delete_by_doc(doc_id)
        self.qdrant.delete_by_doc(doc_id)
        return docs, chunks

    def list_sources(self) -> list[SourceStats]:
        return self.pg.list_sources()

    def get_stats(self) -> StoreStats:
        return self.pg.get_stats()

    def health(self) -> dict:
        qok = self.qdrant.ping()
        pok = self.pg.ping()
        return {
            "ok": qok and pok,
            "qdrant_ok": qok,
            "postgres_ok": pok,
            "embedding_backend": self.provider.name,
        }
