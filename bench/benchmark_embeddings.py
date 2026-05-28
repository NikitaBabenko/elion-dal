"""Бенч эмбеддинг-провайдеров на CPU: латентность запроса, throughput индексации,
наличие sparse-выхода. Помогает выбрать EMBEDDING_BACKEND (fastembed vs flag).

Запуск:
    python -m bench.benchmark_embeddings
    python -m bench.benchmark_embeddings --backends fastembed flag
"""

from __future__ import annotations

import argparse
import statistics
import time

from elion_dal.config import get_settings
from elion_dal.embedding.base import EmbeddingProvider

SAMPLE_QUERIES = [
    "когда олимпиада Физтех по биологии",
    "как получить справку для налогового вычета",
    "правила приёма на бакалавриат биомед",
    "стоимость летней проектной школы по биоинформатике",
    "какие олимпиады засчитываются при поступлении",
]

SAMPLE_DOCS = [
    "Олимпиада Физтех по биологии проводится в два этапа: отборочный онлайн и заключительный.",
    "Для налогового вычета необходимо заявление на предоставление справки о стоимости обучения.",
    "Правила приёма на бакалавриат определяют минимальные баллы ЕГЭ и перечень вступительных.",
    "Летняя проектная школа по биоинформатике включает лекции, практикумы и защиту проектов.",
    "Засчитываемые олимпиады при поступлении перечислены в соответствующем положении.",
] * 10  # 50 документов


def build(backend: str) -> EmbeddingProvider:
    from elion_dal.config import Settings
    from elion_dal.embedding.factory import build_provider

    s = get_settings()
    return build_provider(
        Settings(
            embedding_backend=backend,
            embedding_model=s.embedding_model,
            embedding_dim=s.embedding_dim,
        )
    )


def bench_backend(backend: str) -> None:
    print(f"\n=== backend: {backend} ===")
    try:
        provider = build(backend)
    except Exception as e:  # noqa: BLE001
        print(f"  недоступен: {e}")
        return

    # warm-up
    q0 = provider.embed_query(SAMPLE_QUERIES[0])
    print(
        f"  dense_dim={len(q0.dense)}  sparse_terms={len(q0.sparse.indices)}  "
        f"sparse_uses_idf={provider.sparse_uses_idf}"
    )

    # латентность одиночного запроса
    lat: list[float] = []
    for q in SAMPLE_QUERIES * 4:
        t = time.perf_counter()
        provider.embed_query(q)
        lat.append((time.perf_counter() - t) * 1000)
    print(
        f"  query latency: p50={statistics.median(lat):.1f}ms  "
        f"mean={statistics.mean(lat):.1f}ms  max={max(lat):.1f}ms"
    )

    # throughput индексации (батч документов)
    t = time.perf_counter()
    provider.embed_documents(SAMPLE_DOCS)
    dt = time.perf_counter() - t
    print(
        f"  index throughput: {len(SAMPLE_DOCS)} docs за {dt:.2f}s "
        f"= {len(SAMPLE_DOCS) / dt:.1f} docs/s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["fastembed", "flag"])
    args = parser.parse_args(argv)
    for b in args.backends:
        bench_backend(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
