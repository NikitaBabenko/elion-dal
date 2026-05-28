# elion-dal — Векторизация и Хранение (Data Access Layer «Элиона»)

Микросервис Этапа 2 ТЗ: принимает документы (чистый текст + метаданные), внутри
**чанкует**, **эмбеддит** (BGE-M3: dense + sparse), хранит в **Postgres**
(source-of-truth) и **Qdrant** (производный индекс), отдаёт результаты
**гибридного поиска** (dense + sparse, fusion = RRF) по **gRPC**.

Сервис не вызывает LLM и не решает про fallback — он лишь ищет и возвращает чанки
с источниками и скорами. Confidence/роутинг/карточки — на стороне RAG-ядра.

## Архитектура

```
ETL / сидер ──UpsertDocuments──► [chunk → embed(dense+sparse)] ──► Postgres (SoT)
                                                               └──► Qdrant (index)
RAG-ядро ──────Search──────────► embed(query) → Qdrant hybrid (RRF) → top-k чанков
```

- **Эмбеддинги за интерфейсом** `EmbeddingProvider` (`src/elion_dal/embedding/`):
  - `fastembed` — BGE-M3 dense (ONNX, CPU) + BM25 sparse (IDF-модификатор Qdrant);
  - `flag` — настоящий BGE-M3 dense + learned sparse (вариант A; `pip install -e ".[flag]"`).
  Выбор — по итогам `bench/benchmark_embeddings.py` (всё на CPU, GPU не нужен).
- **Qdrant**: коллекция `elion_chunks`, named-векторы `dense`(1024, Cosine) + `sparse`,
  payload-индексы `source_id` / `doc_id` / `published_ts`. `point_id` детерминирован →
  идемпотентный upsert.
- **Postgres**: `sources` / `documents` / `chunks`; дедуп по `content_hash`, полная
  пересборка индекса без перекраулинга.

## Быстрый старт (локально)

```bash
# 1. Окружение
python -m venv .venv && . .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Сгенерировать gRPC-код из proto
python scripts/gen_proto.py

# 3. Поднять инфраструктуру и накатить схему
docker compose up -d qdrant postgres
copy .env.example .env          # при необходимости поправить
alembic upgrade head

# 4. (опц.) Сравнить эмбеддинг-провайдеры на CPU и выбрать EMBEDDING_BACKEND
python -m bench.benchmark_embeddings

# 5. Засидить локальную «Базу знаний» (PDF/DOCX из ../База знаний)
python -m elion_dal.ingestion.seed_knowledge_base

# 6. Проверить поиск из консоли
python -m elion_dal.cli.query "как получить справку для налогового вычета"
python -m elion_dal.cli.query "когда олимпиада Физтех по биологии"

# 7. Поднять gRPC-сервер
python -m elion_dal.service.server
```

### Полностью в Docker

```bash
docker compose --profile full up --build
```

### Локальный запуск без Docker (embedded-бэкенды)

Тот же код умеет работать без серверов — через embedded-режим Qdrant и SQLite.
Бэкенды выбираются конфигом, код сервиса не меняется:

```bash
# Qdrant — встроенный on-disk режим, Postgres -> SQLite-файл
set QDRANT_URL=./qdrant_local        # или ":memory:" для эфемерного
set PG_DSN=sqlite:///./elion_dev.db

python -m elion_dal.ingestion.seed_knowledge_base
python -m elion_dal.cli.query "когда олимпиада Физтех по биологии"
```

`QDRANT_URL` интерпретируется так: `http(s)://…` — внешний сервер; `:memory:` —
эфемерный embedded; иначе — путь к локальному on-disk-хранилищу. Прод остаётся на
Qdrant-сервере + Postgres, embedded-режим — для dev/CI без Docker.

## gRPC API (`proto/vectorstore.proto`)

| RPC | Назначение |
|---|---|
| `UpsertDocuments(stream Document)` | индексация (чанкинг+эмбеддинг внутри, идемпотентно по хешу) |
| `Search(SearchRequest)` | гибридный поиск (RRF), Топ-k чанков + источники + скоры |
| `DeleteBySource(SourceRef)` | удалить источник из PG и Qdrant (переиндексация) |
| `HealthCheck` | живость + доступность Qdrant/Postgres |

Проверка через `grpcurl` (включена reflection):
```bash
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext -d '{"query":"налоговый вычет","top_k":3}' \
  localhost:50051 elion.vectorstore.v1.VectorStore/Search
```

## Тесты

```bash
pytest                 # unit (быстрые, без инфраструктуры)
pytest -m integration  # round-trip на поднятых Qdrant+Postgres (+скачивание модели)
```

## Конфигурация (`.env`)

См. `.env.example`: `GRPC_*`, `QDRANT_URL`, `PG_DSN`, `EMBEDDING_BACKEND`
(`fastembed`|`flag`), `CHUNK_TOKENS`/`CHUNK_OVERLAP`, `SEARCH_TOP_K`/`SEARCH_PREFETCH`.

## За рамками сервиса

Веб-краулинг/ETL, RAG-ядро и LLM, роутинг интентов, виджет, админ-панель — отдельные
части системы. Документы сюда приходят уже очищенными (от ETL) либо из сид-утилиты.
