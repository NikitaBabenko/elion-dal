"""Контракт эмбеддинг-провайдера.

Любой провайдер отдаёт пару (dense, sparse). Чем именно является sparse —
learned-веса BGE-M3 (FlagEmbedding, вариант A) или BM25 (FastEmbed) — скрыто за
интерфейсом; различие в обработке на стороне Qdrant выражается флагом
`sparse_uses_idf` (BM25 требует IDF-модификатор коллекции, learned-веса — нет).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(slots=True)
class Embedding:
    dense: list[float]
    sparse: SparseVector


class EmbeddingProvider(ABC):
    name: str
    dim: int
    #: True, если sparse — это BM25-частоты и Qdrant должен применять IDF-модификатор.
    sparse_uses_idf: bool

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Эмбеддинг батча документов (для индексации)."""

    @abstractmethod
    def embed_query(self, text: str) -> Embedding:
        """Эмбеддинг одного запроса (для поиска)."""
