# examples/ — демо-микросервисы для DAL

Две тонкие Python-программы, иллюстрирующие, **как другие части системы**
(ETL и RAG-ядро) **должны общаться с DAL** через публичный REST API.

| Файл | Кого имитирует | Что делает |
|---|---|---|
| `writer_service.py` | ETL/краулер | Собирает документы → режет на секции → `POST /api/v1/documents` |
| `reader_service.py` | RAG-ядро | По вопросу пользователя → `POST /api/v1/search` → форматирует контекст для LLM |
| `dal_client.py` | (общий) | Тонкий `httpx`-клиент с Bearer-токеном и обработкой ошибок |

Это разные процессы и разные обязанности — ETL и RAG разрабатываются и
деплоятся независимо, у DAL единственный публичный контракт — REST.

## Запуск

```bash
# 1. Креденшалы прода (либо локалки)
set DAL_BASE_URL=https://elion-dal.vibenest.net
set DAL_API_TOKEN=<API_TOKEN>     # значение из VibeNest Environment

# 2. Заливка sample-набора через writer
python -m examples.writer_service ingest --source kb-demo
python -m examples.writer_service status

# 3. Поиск через reader
python -m examples.reader_service search "налоговый вычет" --top-k 3
python -m examples.reader_service search "олимпиада физтех биология" --source kb-demo
python -m examples.reader_service stats
python -m examples.reader_service sources

# 4. Уборка тестовых данных
python -m examples.writer_service delete --source kb-demo
```

Без `DAL_API_TOKEN` запросы пойдут без `Authorization`-заголовка — это
сработает только если DAL запущен в dev-режиме (пустой `API_TOKEN` на сервере).
В проде с включённым auth получишь 401.

## sample-набор

`writer_service.py` содержит 4 синтетических документа про ФБМФ МФТИ
(правила приёма, олимпиада, налоговый вычет, общежитие). Каждый —
несколько секций с `heading_path` — это и есть «родители» в parent-child
retrieval. Поиск на стороне DAL идёт по дочерним чанкам, в выдачу
возвращаются эти родители целиком.

Чтобы посмотреть набор как JSON (например, для собственного интеграционного
теста):

```bash
python -m examples.writer_service dump-samples
```

## dal_client.py — что внутри

- `DalClient(base_url, token, timeout)` — sync-клиент, контекст-менеджер.
- `Section(section_id, text, heading_path=..., url=..., published_ts=...)` —
  один parent для `upsert_document`.
- `DalError(message, status, body)` — единое исключение для HTTP-ошибок;
  у вызывающего есть `.status` и `.body[:500]` для логов.
- Эндпоинты: `healthz / get_stats / list_sources / search /
  upsert_document / delete_source / delete_doc / get_settings /
  update_settings`.

Async-аналог не делаю — для демо хватает sync-режима; реальное RAG-ядро
вряд ли упрётся в задержку DAL (p50 ≈ 150 ms на CPU), а если упрётся —
заменить `httpx.Client` на `httpx.AsyncClient` тривиально.
