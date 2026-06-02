"""Пересборка индекса Qdrant из Postgres (source-of-truth) — disaster recovery.

Восстанавливает векторный индекс после потери/повреждения данных Qdrant, не
обращаясь к ETL/краулеру: читает готовые чанки из PG, переэмбеддивает, заливает
обратно. Идемпотентно по детерминированным point_id.

Запуск:
    python -m elion_dal.cli.reindex                      # все источники
    python -m elion_dal.cli.reindex --source biomed-total-base
    python -m elion_dal.cli.reindex --recreate           # снести коллекцию и собрать с нуля
    python -m elion_dal.cli.reindex --dry-run            # посчитать, ничего не писать

ВНИМАНИЕ: при повреждённом storage Qdrant сначала почините инфру (диск/volume),
иначе reindex зальёт данные в то же битое хранилище. `--recreate` сносит старые
сегменты целиком — полезно после повреждения.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..logging_setup import setup_logging
from ..service.bootstrap import build_index_service


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Reindex Qdrant из Postgres (SoT)")
    parser.add_argument("--source", default=None, help="только один source_id (по умолчанию все)")
    parser.add_argument(
        "--recreate", action="store_true", help="пересоздать коллекцию с нуля (drop + create)"
    )
    parser.add_argument("--batch", type=int, default=256, help="размер выборки документов из PG")
    parser.add_argument(
        "--dry-run", action="store_true", help="посчитать объём, ничего не писать в Qdrant"
    )
    args = parser.parse_args(argv[1:])

    setup_logging("INFO")
    log = logging.getLogger("elion_dal.reindex")

    # ensure=True гарантирует существование коллекции (если не --recreate).
    index = build_index_service(ensure=not args.recreate)
    if args.recreate and not args.dry_run:
        log.info("Пересоздаю коллекцию Qdrant с нуля...")
        index.reindex_recreate_collection()

    log.info(
        "Reindex source=%s batch=%d dry_run=%s ...",
        args.source or "(все)", args.batch, args.dry_run,
    )
    stats = index.reindex_from_pg(source_id=args.source, batch=args.batch, dry_run=args.dry_run)
    log.info(
        "Готово: документов=%d чанков=%d ошибок=%d", stats.docs, stats.chunks, stats.failed
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
