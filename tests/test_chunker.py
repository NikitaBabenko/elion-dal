"""Unit-тесты чанкера (offline: длина считается через length_fn, без токенайзера)."""

from __future__ import annotations

import pytest

from elion_dal.chunking.chunker import Chunker

# Длина в "словах" — детерминированно и без загрузки модели.
WORDS = lambda s: len(s.split())  # noqa: E731


def make_chunker(tokens=10, overlap=2):
    return Chunker(chunk_tokens=tokens, chunk_overlap=overlap, length_fn=WORDS)


def test_empty_text_returns_no_chunks():
    assert make_chunker().split("") == []
    assert make_chunker().split("   \n  ") == []


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunk_tokens=10, chunk_overlap=10, length_fn=WORDS)


def test_splits_long_text_into_sequential_chunks():
    text = " ".join(f"слово{i}" for i in range(45))  # 45 "токенов" при tokens=10
    chunks = make_chunker(tokens=10, overlap=2).split(text)
    assert len(chunks) >= 4
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Каждый чанк не превышает лимит (текст дробится по пробелам).
    assert all(c.token_count <= 10 for c in chunks)
    assert all(c.text for c in chunks)


def test_short_text_is_single_chunk():
    chunks = make_chunker(tokens=10).split("одно два три")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].token_count == 3


def test_min_tokens_drops_short_and_renumbers():
    text = " ".join(f"слово{i}" for i in range(45))
    base = make_chunker(tokens=10, overlap=2).split(text)
    filtered = Chunker(chunk_tokens=10, chunk_overlap=2, min_tokens=5, length_fn=WORDS).split(text)
    # все оставшиеся не короче порога
    assert all(c.token_count >= 5 for c in filtered)
    assert len(filtered) <= len(base)
    # индексы непрерывны после дропа
    assert [c.index for c in filtered] == list(range(len(filtered)))


def test_min_tokens_zero_is_backward_compatible():
    text = " ".join(f"w{i}" for i in range(30))
    a = Chunker(chunk_tokens=8, chunk_overlap=2, min_tokens=0, length_fn=WORDS).split(text)
    b = make_chunker(tokens=8, overlap=2).split(text)
    assert [c.text for c in a] == [c.text for c in b]


def test_separator_mode_stored_and_validated():
    assert Chunker(chunk_tokens=10, chunk_overlap=2, length_fn=WORDS).separator_mode == "structured"
    tok = Chunker(chunk_tokens=10, chunk_overlap=2, separator_mode="token", length_fn=WORDS)
    assert tok.separator_mode == "token"
    # неизвестный режим тихо откатывается на structured
    bad = Chunker(chunk_tokens=10, chunk_overlap=2, separator_mode="???", length_fn=WORDS)
    assert bad.separator_mode == "structured"


def test_separator_modes_both_produce_chunks():
    text = "Первое предложение. Второе предложение! Третье; четвёртое, пятое слово тут."
    structured = Chunker(
        chunk_tokens=4, chunk_overlap=0, separator_mode="structured", length_fn=WORDS
    ).split(text)
    token = Chunker(
        chunk_tokens=4, chunk_overlap=0, separator_mode="token", length_fn=WORDS
    ).split(text)
    assert structured and token
    assert all(c.token_count <= 4 for c in structured)
    assert all(c.token_count <= 4 for c in token)
