"""Демо «писатель»: микросервис, который **собирает и записывает** данные в DAL.

В реальной системе это ETL/crawler: он принимает документы (PDF/HTML/Markdown
после очистки), режет их на секции, и кладёт в DAL через `POST /api/v1/documents`.
Здесь — синтетический набор документов про ФБМФ МФТИ; полезен для дымового
прогона задеплоенного сервиса.

Команды:
    python -m examples.writer_service ingest              # залить весь набор
    python -m examples.writer_service ingest --source kb-demo
    python -m examples.writer_service status              # список источников + объёмы
    python -m examples.writer_service delete --source kb-demo  # удалить тестовый источник
    python -m examples.writer_service delete-doc --doc-id <id>

Конфиг через env: DAL_BASE_URL (по умолчанию прод), DAL_API_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Папка с этим скриптом — на sys.path, чтобы импорт dal_client работал и при
# запуске из корня репо, и при `python -m examples.writer_service`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dal_client import DalClient, DalError, Section  # noqa: E402

DEFAULT_SOURCE = "kb-demo"

# Маленький синтетический набор. В реальной жизни — выход краулера/ETL после
# очистки от меню/footer и разбиения на секции по заголовкам Markdown.
SAMPLE_DOCS: list[dict] = [
    {
        "doc_id": "kb-demo-admission",
        "title": "Правила приёма ФБМФ МФТИ 2026",
        "url": "https://example.org/fbmf/admission-2026",
        "lang": "ru",
        "sections": [
            {
                "section_id": "0",
                "heading_path": ["Правила приёма", "Общие положения"],
                "text": (
                    "Приём абитуриентов на физтех-школу биологической и медицинской физики "
                    "(ФБМФ) МФТИ ведётся по результатам ЕГЭ, олимпиад и собеседования. "
                    "Срок подачи документов — с 20 июня по 25 июля 2026 года. "
                    "Программы: «Биофизика», «Медицинская физика», «Биоинженерия»."
                ),
            },
            {
                "section_id": "1",
                "heading_path": ["Правила приёма", "Олимпиадники"],
                "text": (
                    "Победители и призёры Всероссийской олимпиады школьников по биологии, "
                    "химии и физике, а также олимпиады Физтех имеют право поступления без "
                    "вступительных испытаний при наличии результата ЕГЭ не ниже 75 баллов."
                ),
            },
        ],
    },
    {
        "doc_id": "kb-demo-olympiad",
        "title": "Олимпиада Физтех по биологии",
        "url": "https://example.org/fbmf/olympiad-biology",
        "lang": "ru",
        "sections": [
            {
                "section_id": "0",
                "heading_path": ["Олимпиада Физтех", "Биология"],
                "text": (
                    "Олимпиада Физтех по биологии — двухэтапная: отборочный (онлайн) и "
                    "заключительный (очный) этапы. Отборочный проходит в ноябре–декабре, "
                    "заключительный — в феврале на площадке МФТИ. Задания включают "
                    "молекулярную биологию, генетику и физиологию."
                ),
            },
            {
                "section_id": "1",
                "heading_path": ["Олимпиада Физтех", "Регистрация"],
                "text": (
                    "Регистрация участников открывается на сайте olymp.mipt.ru в сентябре. "
                    "Участвовать могут школьники 8–11 классов. По итогам победители "
                    "получают льготы при поступлении в МФТИ."
                ),
            },
        ],
    },
    {
        "doc_id": "kb-demo-tax",
        "title": "Справка для налогового вычета (бланк)",
        "url": "https://example.org/fbmf/tax-deduction-form",
        "lang": "ru",
        "sections": [
            {
                "section_id": "0",
                "heading_path": ["Справки", "Налоговый вычет"],
                "text": (
                    "Для оформления социального налогового вычета за обучение студент или "
                    "родитель подаёт заявление в учебный отдел ФБМФ. К заявлению прилагаются: "
                    "копия договора об обучении, копии платёжных документов, копия ИНН. "
                    "Справка выдаётся в течение 10 рабочих дней."
                ),
            },
        ],
    },
    {
        "doc_id": "kb-demo-dorm",
        "title": "Заселение в общежитие первокурсников",
        "url": "https://example.org/fbmf/dorm",
        "lang": "ru",
        "sections": [
            {
                "section_id": "0",
                "heading_path": ["Общежитие", "Заселение"],
                "text": (
                    "Заселение первокурсников ФБМФ в общежития МФТИ проходит в Долгопрудном "
                    "с 24 по 30 августа. При себе иметь паспорт, флюорографию, копию приказа "
                    "о зачислении и оригинал ЕГЭ. Распределение по комнатам — по заявкам."
                ),
            },
        ],
    },
]


def _logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s writer: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("writer")


def _doc_to_payload(d: dict, source_id: str, published_ts: int) -> dict:
    return {
        "doc_id": d["doc_id"],
        "source_id": source_id,
        "title": d["title"],
        "url": d.get("url", ""),
        "lang": d.get("lang", "ru"),
        "published_ts": published_ts,
        "sections": [
            Section(
                section_id=s["section_id"],
                heading_path=s.get("heading_path", []),
                text=s["text"],
                url=d.get("url", ""),
            )
            for s in d["sections"]
        ],
    }


def cmd_ingest(args: argparse.Namespace) -> int:
    log = _logger()
    log.info("DAL: ingest %d документов в source=%r", len(SAMPLE_DOCS), args.source)
    published_ts = int(datetime.now(tz=UTC).timestamp())

    with DalClient() as client:
        log.info("health: %s", client.healthz())
        total = {"received": 0, "indexed": 0, "skipped": 0, "blank": 0, "failed": 0}
        for d in SAMPLE_DOCS:
            payload = _doc_to_payload(d, source_id=args.source, published_ts=published_ts)
            try:
                t0 = time.perf_counter()
                resp = client.upsert_document(**payload)
                dt = (time.perf_counter() - t0) * 1000
            except DalError as e:
                log.error("doc %s: %s body=%s", d["doc_id"], e, e.body)
                total["failed"] += 1
                continue
            log.info(
                "  %s — %dms — indexed=%d parents=%d chunks=%d",
                d["doc_id"],
                dt,
                resp.get("indexed", 0),
                resp.get("parents_upserted", 0),
                resp.get("chunks_upserted", 0),
            )
            for k in total:
                total[k] += resp.get(k, 0)
        log.info("итого: %s", total)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    log = _logger()
    with DalClient() as client:
        log.info("health: %s", client.healthz())
        stats = client.get_stats()
        log.info(
            "stats: docs=%d parents=%d chunks=%d sources=%d",
            stats.get("total_documents", 0),
            stats.get("total_parents", 0),
            stats.get("total_chunks", 0),
            len(stats.get("sources", [])),
        )
        for s in stats.get("sources", []):
            ts = s.get("last_indexed_ts", 0)
            when = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC") if ts else "—"
            log.info(
                "  %-12s docs=%d parents=%d chunks=%d last=%s name=%s",
                s["source_id"],
                s["document_count"],
                s["parent_count"],
                s["chunk_count"],
                when,
                s.get("name", ""),
            )
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    log = _logger()
    with DalClient() as client:
        resp = client.delete_source(args.source)
        log.info(
            "удалён source=%s: documents=%d chunks=%d",
            args.source,
            resp.get("documents_deleted", 0),
            resp.get("chunks_deleted", 0),
        )
    return 0


def cmd_delete_doc(args: argparse.Namespace) -> int:
    log = _logger()
    with DalClient() as client:
        resp = client.delete_doc(args.doc_id)
        log.info(
            "удалён doc=%s: documents=%d chunks=%d",
            args.doc_id,
            resp.get("documents_deleted", 0),
            resp.get("chunks_deleted", 0),
        )
    return 0


def cmd_dump_samples(_: argparse.Namespace) -> int:
    """Печатает sample-набор JSON-ом (для интеграционных тестов)."""
    print(json.dumps(SAMPLE_DOCS, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Демо-микросервис writer: ETL-импитация для DAL.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="загрузить набор документов в DAL")
    pi.add_argument("--source", default=DEFAULT_SOURCE, help="source_id (по умолчанию kb-demo)")
    pi.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("status", help="показать stats/sources")
    ps.set_defaults(func=cmd_status)

    pd = sub.add_parser("delete", help="удалить источник целиком")
    pd.add_argument("--source", default=DEFAULT_SOURCE)
    pd.set_defaults(func=cmd_delete)

    pdd = sub.add_parser("delete-doc", help="удалить один документ")
    pdd.add_argument("--doc-id", required=True)
    pdd.set_defaults(func=cmd_delete_doc)

    pds = sub.add_parser("dump-samples", help="показать sample-набор JSON")
    pds.set_defaults(func=cmd_dump_samples)

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
