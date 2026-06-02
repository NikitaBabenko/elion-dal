"""Ретраи QdrantRepo на транзиентных сбоях (fake-client, без сети)."""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from elion_dal.embedding.base import Embedding, SparseVector
from elion_dal.store.qdrant_repo import QdrantRepo


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class _FakePoint:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class _FlakyClient:
    """Эмулирует Qdrant-клиент: query_points падает первые fail_n раз, потом успех."""

    def __init__(self, fail_n: int, exc: Exception):
        self.fail_n = fail_n
        self.exc = exc
        self.calls = 0

    def query_points(self, **_kw):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise self.exc
        return _FakeQueryResult(
            [_FakePoint({"chunk_id": "p::0#0", "parent_id": "p::0", "doc_id": "d",
                         "source_id": "s", "text": "t"}, 0.9)]
        )


def _repo(fail_n: int, exc: Exception, attempts: int) -> QdrantRepo:
    # url=http -> _make_client не подключается на старте; сразу подменяем client.
    repo = QdrantRepo(
        url="http://localhost:6333",
        collection="c",
        dim=4,
        sparse_uses_idf=False,
        retry_attempts=attempts,
        retry_base_delay_s=0.01,
        sleep=lambda _d: None,  # без реальных задержек
    )
    repo.client = _FlakyClient(fail_n, exc)
    return repo


def _query() -> Embedding:
    return Embedding(dense=[0.0] * 4, sparse=SparseVector([1], [1.0]))


def test_search_retries_then_succeeds():
    repo = _repo(2, UnexpectedResponse(500, "ISE", b"rocksdb io", {}), attempts=3)
    hits = repo.search(_query(), limit=5)
    assert len(hits) == 1
    assert repo.client.calls == 3  # 2 падения + 1 успех


def test_search_exhausts_and_raises():
    repo = _repo(99, UnexpectedResponse(500, "ISE", b"rocksdb io", {}), attempts=3)
    with pytest.raises(UnexpectedResponse):
        repo.search(_query(), limit=5)
    assert repo.client.calls == 3


def test_dense_scores_retries_transient():
    repo = _repo(1, ResponseHandlingException(TimeoutError("read")), attempts=2)
    out = repo.dense_scores(_query(), limit=5)
    assert out == {"p::0#0": 0.9}
    assert repo.client.calls == 2


def test_no_retry_when_attempts_one():
    repo = _repo(99, UnexpectedResponse(503, "SU", b"x", {}), attempts=1)
    with pytest.raises(UnexpectedResponse):
        repo.search(_query(), limit=5)
    assert repo.client.calls == 1
