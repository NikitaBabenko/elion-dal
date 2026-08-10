"""Unit-тесты PgRepo на временном SQLite (offline, без модели и без Postgres-сервера).

Проверяем дедуп по хешу, запись родителей/детей, get_parents с join документа и
каскадное удаление по источнику.
"""

from __future__ import annotations

from elion_dal.chunking.chunker import Chunk
from elion_dal.store.pg_repo import DocInput, ParentBuild, PgRepo, SectionInput


def make_repo(tmp_path):
    repo = PgRepo(f"sqlite:///{(tmp_path / 'elion_test.db').as_posix()}")
    repo.create_all()
    return repo


def make_doc(
    doc_id="d1",
    content_hash="h1",
    canonical_doc_id="",
    academic_year=0,
    is_active=True,
):
    return DocInput(
        doc_id=doc_id,
        source_id="s1",
        url="u",
        title="Заголовок",
        lang="ru",
        published_ts=0,
        content_hash=content_hash,
        index_in_rag=True,
        canonical_doc_id=canonical_doc_id,
        academic_year=academic_year,
        is_active=is_active,
        sections=[SectionInput(section_id="0", heading_path=["A"], url="u", text="секция")],
    )


def make_parent(parent_id="d1::0"):
    return ParentBuild(
        parent_id=parent_id,
        section_id="0",
        heading_path=["A", "A.1"],
        url="u",
        text="текст родителя",
        token_count=2,
        ordinal=0,
        children=[Chunk(0, "ребёнок1", 1), Chunk(1, "ребёнок2", 1)],
    )


def test_upsert_and_get_content_hash(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    assert repo.get_content_hash("d1") is None
    # upsert_document пишет хеш как "" (pending) — фиксируется отдельно.
    repo.upsert_document(make_doc(content_hash="abc"), raw_text="секция")
    assert repo.get_content_hash("d1") == ""
    repo.set_content_hash("d1", "abc")
    assert repo.get_content_hash("d1") == "abc"


def test_resolve_document_uses_canonical_or_strict_doc_id(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    stored = make_doc(canonical_doc_id="canonical-d1")
    repo.commit_document_version(stored, raw_text="секция", parents=[make_parent()])

    by_doc_id = repo.resolve_document("d1")
    assert by_doc_id is not None
    assert by_doc_id.doc_id == "d1"
    assert by_doc_id.content_hash == "h1"
    assert by_doc_id.index_in_rag is True

    by_canonical = repo.resolve_document("another-id", "canonical-d1")
    assert by_canonical == by_doc_id
    assert repo.resolve_document("missing") is None


def test_commit_document_version_replaces_canonical_doc_atomically(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    old = make_doc(doc_id="d1", content_hash="v1", canonical_doc_id="canonical")
    repo.commit_document_version(old, raw_text="old", parents=[make_parent("d1::0")])

    new = make_doc(doc_id="d2", content_hash="v2", canonical_doc_id="canonical")
    repo.commit_document_version(
        new,
        raw_text="new",
        parents=[make_parent("d2::0")],
        previous_doc_id="d1",
    )

    assert repo.get_content_hash("d1") is None
    assert repo.get_parents(["d1::0"]) == {}
    assert repo.get_content_hash("d2") == "v2"
    assert repo.get_parents(["d2::0"])["d2::0"].doc_id == "d2"
    assert repo.get_doc_id_by_canonical("canonical") == "d2"


def test_parents_and_children_with_join(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])

    recs = repo.get_parents(["d1::0"])
    assert "d1::0" in recs
    rec = recs["d1::0"]
    assert rec.text == "текст родителя"
    assert rec.source_id == "s1"  # join с documents
    assert rec.title == "Заголовок"
    assert rec.heading_path == ["A", "A.1"]


def test_document_metadata_persists_for_search_and_reindex(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(
        make_doc(canonical_doc_id="canonical-d1", academic_year=2026, is_active=False),
        raw_text="секция",
    )
    repo.replace_parents_and_chunks("d1", [make_parent()])
    repo.set_content_hash("d1", "h1")

    assert repo.get_doc_id_by_canonical("canonical-d1") == "d1"
    parent = repo.get_parents(["d1::0"])["d1::0"]
    assert parent.academic_year == 2026
    assert parent.is_active is False

    rows = list(repo.iter_documents_for_reindex())
    assert len(rows) == 1
    assert rows[0].academic_year == 2026
    assert rows[0].is_active is False


def test_replace_parents_is_idempotent(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])
    # Повторная запись (меньше детей) не плодит дубликаты.
    p = make_parent()
    p.children = [Chunk(0, "только один", 1)]
    repo.replace_parents_and_chunks("d1", [p])
    docs, chunks = repo.delete_by_source("s1")
    assert docs == 1
    assert chunks == 1


def test_delete_by_source_cascades(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])

    docs, chunks = repo.delete_by_source("s1")
    assert docs == 1
    assert chunks == 2
    assert repo.get_parents(["d1::0"]) == {}  # каскад снёс родителей и детей


def test_delete_by_doc(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])

    docs, chunks = repo.delete_by_doc("d1")
    assert docs == 1
    assert chunks == 2
    assert repo.get_parents(["d1::0"]) == {}
    # Удаление несуществующего документа — без ошибок и нулевые счётчики.
    assert repo.delete_by_doc("nope") == (0, 0)


def test_delete_all_cascades_sources_docs_and_chunks(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])
    repo.ensure_source("s2")
    repo.upsert_document(make_doc(doc_id="d2"), raw_text="секция")
    repo.replace_parents_and_chunks("d2", [make_parent("d2::0")])

    assert repo.delete_all() == (2, 2, 4)
    stats = repo.get_stats()
    assert stats.total_documents == 0
    assert stats.total_parents == 0
    assert stats.total_chunks == 0
    assert stats.sources == []


def test_list_documents_and_detail(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])
    repo.set_content_hash("d1", "abc")  # закоммичен -> indexed=True

    docs = repo.list_documents()
    assert len(docs) == 1
    d = docs[0]
    assert d.doc_id == "d1"
    assert d.parent_count == 1
    assert d.chunk_count == 2
    assert d.indexed is True
    assert d.index_in_rag is True

    # фильтр по источнику
    assert len(repo.list_documents("s1")) == 1
    assert repo.list_documents("nope") == []

    detail = repo.get_document_detail("d1")
    assert detail is not None
    assert detail.title == "Заголовок"
    assert detail.indexed is True
    assert len(detail.parents) == 1
    p = detail.parents[0]
    assert p.parent_id == "d1::0"
    assert p.heading_path == ["A", "A.1"]
    assert [c.chunk_index for c in p.chunks] == [0, 1]
    assert p.chunks[0].text == "ребёнок1"
    assert p.chunks[0].chunk_id == "d1::0#0"
    assert p.chunks[0].token_count == 1
    # отсутствующий документ
    assert repo.get_document_detail("nope") is None


def test_export_chunks_includes_full_context(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])
    repo.set_content_hash("d1", "abc")

    export = repo.export_chunks("s1")
    assert export["schema"] == "elion-dal.chunks-export.v1"
    assert export["source_id"] == "s1"
    assert export["counts"]["chunks"] == 2
    first = export["chunks"][0]
    assert first["source"]["source_id"] == "s1"
    assert first["document"]["doc_id"] == "d1"
    assert first["document"]["indexed"] is True
    assert first["parent"]["parent_id"] == "d1::0"
    assert first["parent"]["heading_path"] == ["A", "A.1"]
    assert first["chunk"]["chunk_id"] == "d1::0#0"
    assert first["chunk"]["point_id"]
    assert first["chunk"]["text"] == "ребёнок1"


def test_list_documents_pending_flag(tmp_path):
    # Документ записан, но content_hash ещё не зафиксирован -> indexed=False (pending).
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    docs = repo.list_documents()
    assert docs[0].indexed is False


def test_list_sources_and_stats(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])  # 1 родитель, 2 ребёнка

    sources = repo.list_sources()
    assert len(sources) == 1
    s = sources[0]
    assert s.source_id == "s1"
    assert s.document_count == 1
    assert s.parent_count == 1
    assert s.chunk_count == 2

    stats = repo.get_stats()
    assert stats.total_documents == 1
    assert stats.total_parents == 1
    assert stats.total_chunks == 2
    assert len(stats.sources) == 1
