"""Фабрика эмбеддинг-провайдера по конфигу (EMBEDDING_BACKEND)."""

from __future__ import annotations

from ..config import Settings
from .base import EmbeddingProvider


def build_provider(settings: Settings) -> EmbeddingProvider:
    backend = settings.embedding_backend.lower()
    model = settings.embedding_model.strip()
    if backend == "fastembed":
        from .fastembed_provider import FastEmbedProvider

        kwargs: dict = {"dim": settings.embedding_dim}
        if model:
            kwargs["dense_model"] = model
        return FastEmbedProvider(**kwargs)
    if backend == "flag":
        from .flag_provider import FlagProvider

        kwargs = {"dim": settings.embedding_dim}
        if model:
            kwargs["model_name"] = model
        return FlagProvider(**kwargs)
    raise ValueError(f"Неизвестный EMBEDDING_BACKEND: {settings.embedding_backend!r}")
