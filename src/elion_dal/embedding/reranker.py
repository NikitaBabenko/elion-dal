"""Опциональный cross-encoder реранкер (по умолчанию ВЫКЛЮЧЕН).

Включается RERANK_ENABLED=true. Переупорядочивает схлопнутых родителей по
релевантности (query, parent_text) — главный рычаг под метрику качества ТЗ.
Требует доп. зависимость `.[flag]` (torch) и грузит модель ~600 МБ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        """Вернуть скоры релевантности, выровненные по docs (больше = релевантнее)."""


class FlagRerankerProvider(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(model_name, use_fp16=False)

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        docs = list(docs)
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        scores = self._model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):  # один документ -> скаляр
            scores = [scores]
        return [float(s) for s in scores]
