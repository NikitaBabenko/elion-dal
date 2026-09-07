"""Lifecycle updates with unchanged text must reach both stores and REST search."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from elion_dal.config import Settings
from elion_dal.service.rest_api import create_api
from elion_dal.service.sync import IndexService, UpsertCounts
from elion_dal.store.qdrant_repo import QdrantRepo

from .test_reindex import FakeChunker, FakeProvider, make_doc, make_repo
from .test_sync import FakeQdrant


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "before,after",
    [
        ({"academic_year": 2025, "is_active": False}, {"is_active": True}),
        ({"academic_year": 2025, "is_active": True}, {"is_active": False}),
        ({"academic_year": None}, {"academic_year": 2026}),
        ({"academic_year": 2025}, {"academic_year": 2026}),
        ({"academic_year": 2026}, {"academic_year": None}),
        ({"published_ts": 1}, {"published_ts": 2}),
        ({"metadata_fingerprint": "v1"}, {"metadata_fingerprint": "v2"}),
    ],
)
def test_metadata_update_with_same_text(tmp_path, canonical, before, after):
    pg = make_repo(tmp_path)
    qd = QdrantRepo(":memory:", "lifecycle", dim=4, sparse_uses_idf=False)
    qd.ensure_collection()
    svc = IndexService(pg, qd, FakeProvider(), FakeChunker())
    old = replace(make_doc(doc_id="articles/156"), **before)
    if canonical:
        old.canonical_doc_id = "article-156"
    new = replace(old, **after)
    if canonical:
        new.doc_id = "articles/156-new"
    try:
        svc.process_document(old, UpsertCounts())
        counts = UpsertCounts()
        svc.process_document(new, counts)
        assert counts.indexed == 1
        assert counts.skipped == counts.failed == 0

        detail = pg.get_document_detail(new.doc_id)
        year = new.academic_year or 0
        active = True if new.is_active is None else new.is_active
        assert detail.academic_year == year
        assert detail.is_active is active
        assert detail.metadata_fingerprint == new.metadata_fingerprint
        assert detail.published_ts == new.published_ts
        points, _ = qd.client.scroll(qd.collection, limit=100)
        assert len(points) == 3
        assert all(p.payload["doc_id"] == new.doc_id for p in points)
        assert all(p.payload["academic_year"] == year for p in points)
        assert all(p.payload["is_active"] is active for p in points)
        assert all(p.payload["published_ts"] == new.published_ts for p in points)
        if canonical:
            assert pg.get_document_detail(old.doc_id) is None

        client = TestClient(create_api(svc, Settings(api_token="")))
        response = client.post("/api/v1/search", json={
            "query": "biology", "academic_year": year, "is_active": active,
        })
        assert response.status_code == 200
        hits = response.json()["hits"]
        assert hits and all(h["doc_id"] == new.doc_id for h in hits)
        assert all(h["academic_year"] == year and h["is_active"] is active for h in hits)
        opposite = client.post("/api/v1/search", json={
            "query": "biology", "is_active": not active,
        })
        assert opposite.json()["hits"] == []

        repeated = UpsertCounts()
        svc.process_document(replace(new), repeated)
        assert repeated.skipped == 1
    finally:
        qd.client.close()
        pg.engine.dispose()


def test_failed_lifecycle_update_can_be_retried(tmp_path):
    pg = make_repo(tmp_path)
    qd = FakeQdrant()
    svc = IndexService(pg, qd, FakeProvider(), FakeChunker())
    old = make_doc(academic_year=2025, is_active=False)
    new = replace(old, academic_year=2026, is_active=True)
    try:
        svc.process_document(old, UpsertCounts())
        qd.fail_upserts = 1
        failed = UpsertCounts()
        svc.process_document(new, failed)
        assert failed.failed == 1
        assert pg.get_document_detail(old.doc_id).is_active is False
        assert pg.get_document_detail(old.doc_id).academic_year == 2025

        retried = UpsertCounts()
        svc.process_document(new, retried)
        assert retried.indexed == 1
        assert retried.skipped == retried.failed == 0
        assert pg.get_document_detail(new.doc_id).is_active is True
        assert all(p.payload["academic_year"] == 2026 for p in qd.points[new.doc_id])
    finally:
        pg.engine.dispose()
