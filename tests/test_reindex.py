"""Reindex round-trip: реальный SQLite PgRepo + фейковые Qdrant/provider/chunker.

Проверяем disaster recovery: после «потери» Qdrant пересобрать индекс из PG (SoT)
с теми же point_id/chunk_id, без обращения к ETL.
"""

from __future__ import annotations

from elion_dal.chunking.chunker import Chunk
from elion_dal.embedding.base import Embedding, SparseVector
from elion_dal.service.sync import IndexService, UpsertCounts
from elion_dal.store.models import chunk_id
from elion_dal.store.pg_repo import DocInput, PgRepo, SectionInput


class FakeQdrant:
    def __init__(self):
        self.points: dict[str, list] = {}

    def delete_by_doc(self, doc_id):
        self.points.pop(doc_id, None)

    def upsert_chunks(self, points):
        for p in points:
            self.points.setdefault(p.payload["doc_id"], []).append(p)
        return len(points)

    def delete_by_source(self, source_id):
        pass

    def search(
        self,
        embedding,
        limit,
        source_ids=(),
        min_published_ts=0,
        prefetch_limit=None,
        academic_year=None,
        is_active=None,
    ):
        """Заглушка для поиска — возвращает пустой список."""
        return []

class FakeProvider:
    name = "fake"
    dim = 4
    sparse_uses_idf = False

    def embed_documents(self, texts):
        return [Embedding(dense=[0.0] * 4, sparse=SparseVector([1], [1.0])) for _ in texts]

    def embed_query(self, text):
        return Embedding(dense=[0.0] * 4, sparse=SparseVector([1], [1.0]))


class FakeChunker:
    chunk_tokens = 10
    chunk_overlap = 2

    def split(self, text):
        parts = [p for p in text.split("|") if p]
        return [Chunk(index=i, text=p, token_count=len(p)) for i, p in enumerate(parts)]

    def count_tokens(self, text):
        return len(text)


def make_service(pg, qdrant, batch=256):
    return IndexService(pg, qdrant, FakeProvider(), FakeChunker(), upsert_batch_size=batch)


def make_repo(tmp_path):
    repo = PgRepo(f"sqlite:///{(tmp_path / 'reindex.db').as_posix()}")
    repo.create_all()
    return repo


def make_doc(doc_id="d1", text="a|b|c", h="h1", index=True, source="s1"):
    return DocInput(
        doc_id=doc_id, source_id=source, url="u", title="T", lang="ru",
        published_ts=0, content_hash=h, index_in_rag=index,
        sections=[SectionInput(section_id="0", heading_path=["A"], url="u", text=text)],
    )


def test_reindex_restores_after_qdrant_loss(tmp_path):
    repo = make_repo(tmp_path)
    svc = make_service(repo, FakeQdrant())
    svc.process_document(make_doc(), UpsertCounts())
    assert len(svc.qdrant.points["d1"]) == 3  # три ребёнка "a|b|c"

    # «Теряем» Qdrant: подменяем на пустой.
    svc.qdrant = FakeQdrant()
    stats = svc.reindex_from_pg()
    assert stats.docs == 1
    assert stats.chunks == 3
    assert stats.failed == 0

    # Точки восстановлены с теми же chunk_id (=> те же детерминированные point_id).
    restored = svc.qdrant.points["d1"]
    assert len(restored) == 3
    ids = {p.payload["chunk_id"] for p in restored}
    assert ids == {chunk_id("d1::0", 0), chunk_id("d1::0", 1), chunk_id("d1::0", 2)}
    # Payload-поля родителя/документа на месте.
    assert restored[0].payload["source_id"] == "s1"
    assert restored[0].payload["heading_path"] == ["A"]


def test_reindex_skips_pending(tmp_path):
    repo = make_repo(tmp_path)
    qd = FakeQdrant()
    svc = make_service(repo, qd)
    # Документ записан в PG как pending (хеш не зафиксирован) — эмулируем «полузапись».
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(h="x"), raw_text="a|b|c")  # content_hash="" в PG
    # parents без commit hash
    from elion_dal.store.pg_repo import ParentBuild

    repo.replace_parents_and_chunks(
        "d1",
        [ParentBuild("d1::0", "0", ["A"], "u", "a|b|c", 3, 0, FakeChunker().split("a|b|c"))],
    )
    stats = svc.reindex_from_pg()
    assert stats.docs == 0  # pending (hash="") пропущен


def test_reindex_filter_by_source(tmp_path):
    repo = make_repo(tmp_path)
    svc = make_service(repo, FakeQdrant())
    svc.process_document(make_doc(doc_id="d1", source="s1"), UpsertCounts())
    svc.process_document(make_doc(doc_id="d2", source="s2"), UpsertCounts())

    svc.qdrant = FakeQdrant()
    stats = svc.reindex_from_pg(source_id="s1")
    assert stats.docs == 1
    assert "d1" in svc.qdrant.points
    assert "d2" not in svc.qdrant.points


def test_reindex_dry_run_writes_nothing(tmp_path):
    repo = make_repo(tmp_path)
    svc = make_service(repo, FakeQdrant())
    svc.process_document(make_doc(), UpsertCounts())

    svc.qdrant = FakeQdrant()
    stats = svc.reindex_from_pg(dry_run=True)
    assert stats.chunks == 3  # посчитал
    assert svc.qdrant.points == {}  # но ничего не записал
