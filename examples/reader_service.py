"""Демо «читатель»: микросервис, который **запрашивает** данные у DAL.

В реальной системе это RAG-ядро: получает вопрос пользователя → дёргает
`/api/v1/search` → склеивает текст родителей в контекст для LLM. Здесь —
тонкий CLI, который печатает результаты в удобном для глаза виде.

Команды:
    python -m examples.reader_service search "налоговый вычет" --top-k 3
    python -m examples.reader_service search "олимпиада" --source kb-demo
    python -m examples.reader_service stats
    python -m examples.reader_service sources
    python -m examples.reader_service settings
    python -m examples.reader_service health

Конфиг через env: DAL_BASE_URL, DAL_API_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dal_client import DalClient, DalError  # noqa: E402


def _logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s reader: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("reader")


def _print_hit(idx: int, h: dict) -> None:
    crumbs = " › ".join(h.get("heading_path") or [])
    text = (h.get("text") or "").strip().replace("\n", " ")
    matched = (h.get("matched_child") or "").strip().replace("\n", " ")
    score = h.get("score", 0.0)
    dense = h.get("dense_score", 0.0)
    print(f"\n[{idx}] {h.get('title') or h.get('parent_id')}")
    print(f"    источник: {h.get('source_id')}   url: {h.get('url') or '—'}")
    if crumbs:
        print(f"    {crumbs}")
    print(f"    score={score:.4f}  dense={dense:.4f}  (confidence-сигнал для fallback)")
    print(f"    нашли по: {matched[:200]}")
    print(f"    контекст: {text[:300]}{'…' if len(text) > 300 else ''}")


def cmd_search(args: argparse.Namespace) -> int:
    log = _logger()
    sources = [s.strip() for s in (args.source or "").split(",") if s.strip()]
    min_ts = 0
    if args.since:
        # YYYY-MM-DD → unix ts UTC midnight
        dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)
        min_ts = int(dt.timestamp())

    with DalClient() as client:
        log.info(
            "search: query=%r top_k=%d sources=%s since=%s",
            args.query, args.top_k, sources or "(any)", args.since or "(any)",
        )
        hits = client.search(
            query=args.query,
            top_k=args.top_k,
            source_ids=sources,
            min_published_ts=min_ts,
        )
        if not hits:
            print("\n(no-hit) — поднимай порог fallback на стороне RAG-ядра.")
            return 0
        print(f"\nНайдено хитов: {len(hits)}")
        for i, h in enumerate(hits, 1):
            _print_hit(i, h)
        # Доп. подсказка: confidence-сигнал. Для BGE-M3 cosine редко падает
        # ниже 0.7 на одном языке, поэтому ориентир «уверенного» матча — 0.85+.
        # Реальный порог fallback подбирает RAG-ядро под свой кейс.
        top_dense = hits[0].get("dense_score", 0.0)
        if top_dense < 0.85:
            print(
                f"\n⚠ top dense_score={top_dense:.3f} ниже 0.85 — "
                f"RAG-ядру стоит зайти в fallback."
            )
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    with DalClient() as client:
        stats = client.get_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_sources(_: argparse.Namespace) -> int:
    with DalClient() as client:
        for s in client.list_sources():
            ts = s.get("last_indexed_ts", 0)
            when = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC") if ts else "—"
            print(
                f"{s['source_id']:<16} docs={s['document_count']:<4} "
                f"parents={s['parent_count']:<4} chunks={s['chunk_count']:<4} "
                f"last={when}  name={s.get('name', '')}"
            )
    return 0


def cmd_settings(_: argparse.Namespace) -> int:
    with DalClient() as client:
        for f in client.get_settings():
            ovr = " *" if f.get("is_override") else ""
            tier = f.get("tier", "?")
            label = f.get("label", "")
            print(f"  {f['key']:<28} = {f.get('value', ''):<10} ({tier}{ovr})  — {label}")
    return 0


def cmd_health(_: argparse.Namespace) -> int:
    with DalClient() as client:
        print(json.dumps(client.healthz(), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Демо-микросервис reader: имитация RAG-ядра.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("search", help="hybrid-search (dense+sparse, RRF) с печатью хитов")
    ps.add_argument("query")
    ps.add_argument("--top-k", type=int, default=5)
    ps.add_argument("--source", default="", help="csv source_id для фильтра")
    ps.add_argument("--since", default="", help="YYYY-MM-DD, отсечь старше")
    ps.set_defaults(func=cmd_search)

    sub.add_parser("stats", help="агрегат: docs/parents/chunks").set_defaults(func=cmd_stats)
    sub.add_parser("sources", help="список источников с объёмами").set_defaults(func=cmd_sources)
    sub.add_parser("settings", help="управляемые настройки DAL").set_defaults(func=cmd_settings)
    sub.add_parser("health", help="health-проба").set_defaults(func=cmd_health)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DalError as e:
        print(f"DAL error: {e} (status={e.status}) body={e.body}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
