"""Оркестрация parent-child: документ -> секции(родители) -> дети -> PG + Qdrant.

Индексируем детей, ищем по детям, но возвращаем РОДИТЕЛЕЙ (схлопывая дублирующие
попадания) — точный матч + богатый контекст для генерации. Идемпотентно по хешу.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..chunking.chunker import Chunker
from ..embedding.base import EmbeddingProvider
from ..embedding.reranker import Reranker
from ..store.models import chunk_id, parent_pk
from ..store.pg_repo import DocInput, ParentBuild, PgRepo, SourceStats, StoreStats, sha256
from ..store.qdrant_repo import PointInput, QdrantRepo
from ..store.settings_store import SettingsStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocFailure:
    """Гранулярный отчёт о неудачной индексации одного документа."""

    doc_id: str
    stage: str  # "qdrant_upsert" | "qdrant_delete" | "embed" | "pg" | "rollback"
    written: int  # сколько дочерних точек успело записаться до сбоя
    total: int  # сколько всего должно было
    error: str  # type(e).__name__ + краткое сообщение
    rolled_back: bool  # удалось ли откатить частичную запись


@dataclass(slots=True)
class UpsertCounts:
    received: int = 0
    indexed: int = 0
    skipped: int = 0
    blank: int = 0
    failed: int = 0
    parents_upserted: int = 0
    chunks_upserted: int = 0
    failures: list[DocFailure] = field(default_factory=list)


@dataclass(slots=True)
class ReindexStats:
    docs: int = 0
    chunks: int = 0
    failed: int = 0


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
        settings_store: SettingsStore | None = None,
        reranker_factory: Callable[[], Reranker] | None = None,
        base_settings=None,
        upsert_batch_size: int = 256,
    ) -> None:
        self.pg = pg
        self.qdrant = qdrant
        self.provider = provider
        self.chunker = chunker
        self.settings_store = settings_store
        self._base = base_settings
        self.upsert_batch_size = max(1, upsert_batch_size)
        self._reranker = reranker
        self._reranker_factory = reranker_factory
        # Фолбэк-дефолты, если нет store/base (например, в тестах).
        self._d_parent_fanout = max(1, parent_fanout)
        self._d_recency_weight = recency_weight
        self._d_recency_halflife = max(1.0, recency_halflife_days)
        self._d_rerank_enabled = reranker is not None

    # --- живые настройки: override из БД -> .env -> фолбэк ---
    def _cfg(self, key: str, fallback):
        if self.settings_store is not None:
            v = self.settings_store.get(key)
            if v is not None:
                return v
        if self._base is not None:
            bv = getattr(self._base, key, None)
            if bv is not None:
                return bv
        return fallback

    def _live_parent_fanout(self) -> int:
        return max(1, int(self._cfg("search_parent_fanout", self._d_parent_fanout)))

    def _live_prefetch(self) -> int:
        return int(self._cfg("search_prefetch", getattr(self.qdrant, "prefetch", 20)))

    def _live_recency(self) -> tuple[float, float]:
        return (
            float(self._cfg("recency_weight", self._d_recency_weight)),
            max(1.0, float(self._cfg("recency_halflife_days", self._d_recency_halflife))),
        )

    def _live_rerank_enabled(self) -> bool:
        return bool(self._cfg("rerank_enabled", self._d_rerank_enabled))

    def _apply_live_chunk_params(self) -> None:
        tokens = int(self._cfg("chunk_tokens", self.chunker.chunk_tokens))
        overlap = int(self._cfg("chunk_overlap", self.chunker.chunk_overlap))
        self.chunker.chunk_tokens = tokens
        self.chunker.chunk_overlap = min(overlap, max(0, tokens - 1))  # защита от overlap>=tokens

    def _get_reranker(self) -> Reranker | None:
        if self._reranker is None and self._reranker_factory is not None:
            try:
                self._reranker = self._reranker_factory()
            except Exception as e:  # noqa: BLE001
                logger.warning("Не удалось загрузить реранкер: %s", e)
                self._reranker_factory = None
        return self._reranker

    @staticmethod
    def _recency_mult(published_ts: int, weight: float, halflife: float) -> float:
        """Множитель к скору по свежести: до (1+weight) для свежего, ~1 для старого.
        published_ts=0 (дата неизвестна) -> 1.0."""
        if weight <= 0 or published_ts <= 0:
            return 1.0
        age_days = max(0.0, (time.time() - published_ts) / 86400.0)
        return 1.0 + weight * (0.5 ** (age_days / halflife))

    def settings_view(self):
        return self.settings_store.view(self._base) if self.settings_store is not None else []

    def update_settings(self, items: dict[str, str]) -> None:
        if self.settings_store is not None:
            self.settings_store.set_many(items)

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
        self._apply_live_chunk_params()

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
                    token_count=self.chunker.count_tokens(section.text),
                    ordinal=ordinal,
                    children=children,
                )
            )

        self.pg.replace_parents_and_chunks(doc.doc_id, parents)

        # Запись в Qdrant — «всё или ничего» по документу. И первичная чистка старого
        # поколения (delete_by_doc), и батчи upsert обёрнуты в один try: при сбое (после
        # ретраев внутри методов qdrant) откатываем частичную запись и НЕ ставим
        # content_hash → самозалечивание на следующем ingest.
        flat = [(p, c) for p in parents for c in p.children]
        total = len(flat)
        written = 0
        stage = "qdrant_delete"
        try:
            self.qdrant.delete_by_doc(doc.doc_id)  # снести старое поколение точек
            stage = "qdrant_upsert"
            for start in range(0, total, self.upsert_batch_size):
                window = flat[start : start + self.upsert_batch_size]
                embeddings = self.provider.embed_documents([c.text for (_p, c) in window])
                points = [
                    PointInput(
                        parent_id=p.parent_id,
                        chunk_index=c.index,
                        embedding=emb,
                        payload=self._chunk_payload(
                            parent_id=p.parent_id,
                            chunk_index=c.index,
                            text=c.text,
                            doc_id=doc.doc_id,
                            source_id=doc.source_id,
                            url=p.url,
                            title=doc.title,
                            heading_path=p.heading_path,
                            published_ts=doc.published_ts,
                            lang=doc.lang,
                        ),
                    )
                    for (p, c), emb in zip(window, embeddings, strict=True)
                ]
                written += self.qdrant.upsert_chunks(points)
        except Exception as e:  # noqa: BLE001 — изоляция сбоя по документу
            rolled_back = self._rollback_partial(doc.doc_id)
            counts.failed += 1
            counts.failures.append(
                DocFailure(
                    doc_id=doc.doc_id,
                    stage=stage,
                    written=written,
                    total=total,
                    error=f"{type(e).__name__}: {str(e)[:200]}",
                    rolled_back=rolled_back,
                )
            )
            logger.exception(
                "Индексация прервана doc_id=%s на стадии %s (записано %d/%d, откат=%s)",
                doc.doc_id, stage, written, total, rolled_back,
            )
            # content_hash НЕ фиксируем — документ останется pending.
            return
        counts.chunks_upserted += written

        # Commit point: фиксируем хеш только теперь — Qdrant уже обновлён.
        self.pg.set_content_hash(doc.doc_id, doc.content_hash)
        counts.parents_upserted += len(parents)
        self.pg.touch_source_indexed(doc.source_id)
        counts.indexed += 1

    @staticmethod
    def _chunk_payload(
        *,
        parent_id: str,
        chunk_index: int,
        text: str,
        doc_id: str,
        source_id: str,
        url: str,
        title: str,
        heading_path: list[str],
        published_ts: int,
        lang: str,
    ) -> dict:
        """Единая схема payload точки Qdrant (для индексации и reindex)."""
        return {
            "chunk_id": chunk_id(parent_id, chunk_index),
            "parent_id": parent_id,
            "doc_id": doc_id,
            "source_id": source_id,
            "url": url,
            "title": title,
            "heading_path": heading_path,
            "text": text,
            "published_ts": published_ts,
            "lang": lang,
        }

    def _rollback_partial(self, doc_id: str) -> bool:
        """Снести точки документа из Qdrant после неудачной частичной записи.

        delete_by_doc идемпотентен (по фильтру doc_id) и сам с ретраями. Если откат
        тоже не удался — точки могут остаться, но content_hash остаётся pending, и
        повторный ingest их перезапишет (point_id детерминирован)."""
        try:
            self.qdrant.delete_by_doc(doc_id)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Откат частичной записи не удался doc_id=%s", doc_id)
            return False

    def search(
        self, query: str, top_k: int, source_ids: list[str], min_published_ts: int
    ) -> list[ParentHit]:
        embedding = self.provider.embed_query(query)
        limit = top_k * self._live_parent_fanout()
        child_hits = self.qdrant.search(
            embedding,
            limit=limit,
            source_ids=source_ids,
            min_published_ts=min_published_ts,
            prefetch_limit=self._live_prefetch(),
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
        if self._live_rerank_enabled():
            reranker = self._get_reranker()
            if reranker is not None:
                scores = reranker.rerank(query, [c.text for c in candidates])
                for c, s in zip(candidates, scores, strict=True):
                    c.score = s
                rescored = True
        weight, halflife = self._live_recency()
        if weight > 0:
            for c in candidates:
                c.score *= self._recency_mult(records[c.parent_id].published_ts, weight, halflife)
            rescored = True
        if rescored:
            candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def reindex_recreate_collection(self) -> None:
        """Пересоздать коллекцию Qdrant с нуля (для чистого восстановления после
        битого storage — старые повреждённые сегменты сносятся целиком)."""
        self.qdrant.recreate_collection()

    def reindex_from_pg(
        self, source_id: str | None = None, batch: int = 256, dry_run: bool = False
    ) -> ReindexStats:
        """Disaster recovery: пересобрать индекс Qdrant из Postgres (source-of-truth).

        Переэмбеддивает ГОТОВЫЕ чанки из PG (не перенарезает) → те же детерминированные
        point_id → точное восстановление. PG и content_hash НЕ трогаются. Идемпотентно:
        повторный прогон перезаписывает существующие точки, не плодя дублей.
        """
        stats = ReindexStats()
        for row in self.pg.iter_documents_for_reindex(source_id, batch):
            if not row.chunks:
                continue
            try:
                for start in range(0, len(row.chunks), self.upsert_batch_size):
                    window = row.chunks[start : start + self.upsert_batch_size]
                    embeddings = self.provider.embed_documents([c.text for c in window])
                    points = []
                    for c, emb in zip(window, embeddings, strict=True):
                        par = row.parents.get(c.parent_id)
                        points.append(
                            PointInput(
                                parent_id=c.parent_id,
                                chunk_index=c.chunk_index,
                                embedding=emb,
                                payload=self._chunk_payload(
                                    parent_id=c.parent_id,
                                    chunk_index=c.chunk_index,
                                    text=c.text,
                                    doc_id=row.doc_id,
                                    source_id=row.source_id,
                                    url=par.url if par else "",
                                    title=row.title,
                                    heading_path=par.heading_path if par else [],
                                    published_ts=row.published_ts,
                                    lang=row.lang,
                                ),
                            )
                        )
                    stats.chunks += (
                        len(points) if dry_run else self.qdrant.upsert_chunks(points)
                    )
            except Exception:  # noqa: BLE001 — изоляция сбоя по документу
                stats.failed += 1
                logger.exception("reindex прерван на doc_id=%s", row.doc_id)
            stats.docs += 1
        return stats

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
