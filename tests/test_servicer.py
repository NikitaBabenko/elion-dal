"""Unit-тесты gRPC-servicer: маппинг proto<->DTO, в т.ч. fallback «text -> один родитель»."""

from __future__ import annotations

from elion_dal.config import Settings
from elion_dal.grpc_gen import vectorstore_pb2 as pb
from elion_dal.service.servicer import VectorStoreServicer
from elion_dal.service.sync import ParentHit
from elion_dal.store.pg_repo import SourceStats, StoreStats
from elion_dal.store.settings_store import SettingView


class FakeIndex:
    def __init__(self):
        self.docs = []
        self._hits = []
        self.deleted_docs = []
        self.updated_settings = None

    def process_document(self, doc, counts):
        self.docs.append(doc)
        counts.received += 1
        counts.indexed += 1
        counts.parents_upserted += len(doc.sections)
        counts.chunks_upserted += sum(1 for _ in doc.sections)

    def set_hits(self, hits):
        self._hits = hits

    def search(self, query, top_k, source_ids, min_published_ts):
        return self._hits

    def delete_doc(self, doc_id):
        self.deleted_docs.append(doc_id)
        return 1, 3

    def list_sources(self):
        return [SourceStats("s1", "Источник 1", 1700000000, 2, 4, 9)]

    def get_stats(self):
        return StoreStats(2, 4, 9, self.list_sources())

    def settings_view(self):
        return [
            SettingView("search_parent_fanout", "Fan-out", "live", "int", 5, False),
            SettingView("rerank_enabled", "Реранкер", "live", "bool", False, False),
            SettingView("embedding_backend", "Бэкенд", "restart", "str", "fastembed", False),
        ]

    def update_settings(self, items):
        self.updated_settings = items


def make_servicer():
    return VectorStoreServicer(FakeIndex(), Settings())


def test_upsert_maps_sections():
    svc = make_servicer()
    d = pb.Document(doc_id="d1", source_id="s1", index_in_rag=True)
    sec = d.sections.add()
    sec.section_id = "4.9"
    sec.text = "текст секции"
    sec.heading_path.append("4. Регистрация")

    result = svc.UpsertDocuments(iter([d]), None)
    assert result.documents_received == 1
    mapped = svc.index.docs[0]
    assert len(mapped.sections) == 1
    assert mapped.sections[0].section_id == "4.9"
    assert mapped.sections[0].heading_path == ["4. Регистрация"]


def test_upsert_fallback_text_becomes_single_parent():
    svc = make_servicer()
    d = pb.Document(doc_id="d2", source_id="s1", text="плоский текст", index_in_rag=True)

    svc.UpsertDocuments(iter([d]), None)
    mapped = svc.index.docs[0]
    assert len(mapped.sections) == 1
    assert mapped.sections[0].section_id == "0"
    assert mapped.sections[0].text == "плоский текст"


def test_search_maps_parent_hits():
    svc = make_servicer()
    svc.index.set_hits(
        [
            ParentHit(
                parent_id="d1::0",
                doc_id="d1",
                source_id="s1",
                title="t",
                url="u",
                heading_path=["A"],
                text="родитель",
                matched_child="ребёнок",
                score=0.42,
                dense_score=0.77,
            )
        ]
    )
    resp = svc.Search(pb.SearchRequest(query="q", top_k=1), None)
    assert len(resp.hits) == 1
    h = resp.hits[0]
    assert h.parent_id == "d1::0"
    assert list(h.heading_path) == ["A"]
    assert h.text == "родитель"
    assert h.matched_child == "ребёнок"
    assert abs(h.score - 0.42) < 1e-6
    assert abs(h.dense_score - 0.77) < 1e-6


def test_delete_by_doc():
    svc = make_servicer()
    resp = svc.DeleteByDoc(pb.DocRef(doc_id="d1"), None)
    assert svc.index.deleted_docs == ["d1"]
    assert resp.documents_deleted == 1
    assert resp.chunks_deleted == 3


def test_list_sources():
    svc = make_servicer()
    resp = svc.ListSources(pb.StatsRequest(), None)
    assert len(resp.sources) == 1
    s = resp.sources[0]
    assert s.source_id == "s1"
    assert s.document_count == 2
    assert s.chunk_count == 9
    assert s.last_indexed_ts == 1700000000


def test_get_stats():
    svc = make_servicer()
    resp = svc.GetStats(pb.StatsRequest(), None)
    assert resp.total_documents == 2
    assert resp.total_parents == 4
    assert resp.total_chunks == 9
    assert len(resp.sources) == 1


def test_get_settings_returns_fields():
    svc = make_servicer()
    resp = svc.GetSettings(pb.StatsRequest(), None)
    keys = [f.key for f in resp.fields]
    assert "search_parent_fanout" in keys
    sf = next(f for f in resp.fields if f.key == "search_parent_fanout")
    assert sf.tier == "live"
    assert sf.value == "5"  # значение сериализовано в строку


def test_update_settings_applies_and_returns_current():
    svc = make_servicer()
    resp = svc.UpdateSettings(
        pb.SettingsUpdate(items={"search_parent_fanout": "9", "rerank_enabled": "true"}),
        None,
    )
    assert svc.index.updated_settings == {
        "search_parent_fanout": "9",
        "rerank_enabled": "true",
    }
    # ответ — текущий список настроек (для UI)
    assert any(f.key == "search_parent_fanout" for f in resp.fields)


def test_search_uses_config_top_k_when_zero():
    svc = make_servicer()
    captured = {}
    orig = svc.index.search

    def spy(query, top_k, source_ids, min_published_ts):
        captured["top_k"] = top_k
        return orig(query, top_k, source_ids, min_published_ts)

    svc.index.search = spy
    svc.Search(pb.SearchRequest(query="q", top_k=0), None)
    assert captured["top_k"] == Settings().search_top_k
