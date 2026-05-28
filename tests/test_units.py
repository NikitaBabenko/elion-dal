"""Лёгкие unit-тесты без инфраструктуры и без загрузки моделей."""

from __future__ import annotations

import pytest

from elion_dal.config import Settings
from elion_dal.embedding.factory import build_provider
from elion_dal.store.models import chunk_id, point_id
from elion_dal.store.pg_repo import sha256


def test_chunk_id_format():
    assert chunk_id("doc-1", 3) == "doc-1:3"


def test_point_id_deterministic():
    a = point_id("doc-1", 0)
    b = point_id("doc-1", 0)
    c = point_id("doc-1", 1)
    assert a == b
    assert a != c
    assert len(a) == 36  # UUID


def test_sha256_stable():
    assert sha256("текст") == sha256("текст")
    assert sha256("a") != sha256("b")


def test_factory_unknown_backend():
    with pytest.raises(ValueError):
        build_provider(Settings(embedding_backend="nope"))
