"""Unit-тесты проверки API-токена (чистые функции + servicer-abort)."""

from __future__ import annotations

import pytest

from elion_dal.config import Settings
from elion_dal.grpc_gen import vectorstore_pb2 as pb
from elion_dal.service.auth import extract_token, token_ok
from elion_dal.service.servicer import VectorStoreServicer


def test_extract_token_bearer():
    assert extract_token([("authorization", "Bearer abc123")]) == "abc123"


def test_extract_token_x_api_token():
    assert extract_token([("x-api-token", "tok42")]) == "tok42"


def test_extract_token_absent():
    assert extract_token([]) == ""
    assert extract_token(None) == ""


def test_token_ok_disabled_when_expected_empty():
    # Токен не настроен -> доступ открыт, метаданные не важны.
    assert token_ok(None, "") is True


def test_token_ok_match_and_mismatch():
    md = [("authorization", "Bearer secret")]
    assert token_ok(md, "secret") is True
    assert token_ok(md, "other") is False
    assert token_ok([], "secret") is False


class FakeContext:
    """Мини-замена gRPC-context для теста авторизации."""

    class Aborted(Exception):
        pass

    def __init__(self, metadata):
        self._md = metadata

    def invocation_metadata(self):
        return self._md

    def abort(self, code, details):
        raise FakeContext.Aborted(f"{code}: {details}")


class _Idx:
    settings_store = None  # токен берётся из settings


def test_search_aborts_without_token_when_required():
    svc = VectorStoreServicer(_Idx(), Settings(api_token="secret"))
    ctx = FakeContext([])  # токен не передан
    with pytest.raises(FakeContext.Aborted):
        svc.Search(pb.SearchRequest(query="q", top_k=1), ctx)


def test_search_aborts_with_wrong_token():
    svc = VectorStoreServicer(_Idx(), Settings(api_token="secret"))
    ctx = FakeContext([("authorization", "Bearer nope")])
    with pytest.raises(FakeContext.Aborted):
        svc.Search(pb.SearchRequest(query="q", top_k=1), ctx)
