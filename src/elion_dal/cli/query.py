"""Ручной поиск из консоли — для верификации без gRPC-клиента.

Запуск:
    python -m elion_dal.cli.query "когда олимпиада Физтех по биологии"
    python -m elion_dal.cli.query "налоговый вычет справка" --top-k 5
"""

from __future__ import annotations

import argparse
import sys

from ..service.bootstrap import build_index_service


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Гибридный поиск по индексу Элион-DAL")
    parser.add_argument("query", help="текст запроса")
    parser.add_argument(
        "--top-k", type=int, default=0, help="сколько чанков вернуть (0 = из конфига)"
    )
    parser.add_argument(
        "--source", action="append", default=[], help="фильтр по source_id (можно несколько)"
    )
    args = parser.parse_args(argv[1:])

    index = build_index_service(ensure=False)
    hits = index.search(
        query=args.query,
        top_k=args.top_k or 3,
        source_ids=args.source,
        min_published_ts=0,
    )

    if not hits:
        print("Ничего не найдено.")
        return 0

    print(f"Запрос: {args.query!r}\n")
    for i, h in enumerate(hits, 1):
        snippet = h.text.replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        print(f"#{i}  score={h.score:.4f}  source={h.source_id}")
        print(f"    {h.title}  <{h.url}>")
        print(f"    {snippet}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
