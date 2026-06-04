"""Лёгкие unit-тесты без инфраструктуры и без загрузки моделей."""

from __future__ import annotations

import pytest

from elion_dal.config import Settings
from elion_dal.embedding.factory import build_provider
from elion_dal.store.models import chunk_id, parent_pk, point_id
from elion_dal.store.pg_repo import sha256


def test_parent_pk_format():
    assert parent_pk("doc-1", "4.9") == "doc-1::4.9"


def test_chunk_id_format():
    pid = parent_pk("doc-1", "0")
    assert chunk_id(pid, 3) == "doc-1::0#3"


def test_point_id_deterministic():
    pid = parent_pk("doc-1", "0")
    a = point_id(pid, 0)
    b = point_id(pid, 0)
    c = point_id(pid, 1)
    assert a == b
    assert a != c
    assert len(a) == 36  # UUID


def test_sha256_stable():
    assert sha256("текст") == sha256("текст")
    assert sha256("a") != sha256("b")


def test_factory_unknown_backend():
    with pytest.raises(ValueError):
        build_provider(Settings(embedding_backend="nope"))


def test_quantize_default_off():
    # По замерам torch dynamic int8 для BGE-M3 не снижает RSS -> default OFF (ADR-004).
    assert Settings().embedding_quantize is False


def test_preview_chunking_dry_run_offline():
    # Превью нарезки строит отдельный Chunker, переиспользуя length_fn боевого (offline).
    from elion_dal.chunking.chunker import Chunker
    from elion_dal.service.sync import IndexService

    words = lambda s: len(s.split())  # noqa: E731
    chunker = Chunker(chunk_tokens=10, chunk_overlap=2, length_fn=words)
    svc = IndexService(None, None, None, chunker)

    text = " ".join(f"слово{i}" for i in range(45))
    out = svc.preview_chunking(text)
    assert out["summary"]["count"] >= 4
    assert out["summary"]["count"] == len(out["chunks"])
    assert [c["index"] for c in out["chunks"]] == list(range(len(out["chunks"])))
    assert out["summary"]["dropped"] == 0  # без min_tokens ничего не отсеяно

    # Параметры-оверрайды: явный min_tokens отсеивает короткие хвосты.
    out2 = svc.preview_chunking(text, chunk_tokens=10, chunk_overlap=2, min_tokens=8)
    assert all(c["token_count"] >= 8 for c in out2["chunks"])
    assert out2["summary"]["min_tokens"] == 8
    # перенумерация после дропа
    assert [c["index"] for c in out2["chunks"]] == list(range(len(out2["chunks"])))
