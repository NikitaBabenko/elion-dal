"""Залить корпус (JSONL) в локально запущенный DAL — для экспериментов.

    python scripts/load_local.py <путь_к.jsonl> [source_id] [limit]

Поддерживает два формата:
1. Старый: поля doc_id и text на верхнем уровне.
2. Новый (docs_filtered.jsonl): текст в поле text.normalized.

Сервер должен быть запущен (python -m elion_dal.service.server, порт 8080).
"""

import json
import sys
import time
import logging
import urllib.request

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE = "http://localhost:8080"


def extract_text(doc: dict) -> str:
    """Извлекает текст из документа, поддерживая оба формата."""
    # Новый формат: text.normalized
    if "text" in doc and isinstance(doc["text"], dict):
        raw_text = doc["text"].get("normalized", "")
    else:
        # Старый формат: text на верхнем уровне
        raw_text = doc.get("text", "")

    # Санация текста с замером времени
    if raw_text and isinstance(raw_text, str):
        try:
            from elion_dal.sanitizers.factory import sanitize_text_with_config

            start = time.time()
            sanitized = sanitize_text_with_config(raw_text)
            elapsed = time.time() - start

            # Логируем только если замена заняла время или были изменения
            if sanitized != raw_text:
                logger.debug(f"Санация текста: заменено символов (длина {len(raw_text)}) за {elapsed:.4f}с")
            elif elapsed > 0.01:
                logger.debug(f"Проверка текста заняла {elapsed:.4f}с (без изменений)")

            return sanitized
        except ImportError as e:
            logger.warning(f"Модуль санации не найден: {e}. Пропускаем санацию.")
            return raw_text
        except Exception as e:
            logger.error(f"Ошибка при санации: {e}. Используем исходный текст.")
            return raw_text

    return raw_text or ""


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/load_local.py <path.jsonl> [source_id] [limit]")
        return 1

    path = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "kb-local"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

    logger.info(f"Загрузка файла: {path}")
    logger.info(f"Source ID: {source}")
    logger.info(f"Лимит записей: {limit}")

    ok = fail = 0
    total_start = time.time()

    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    # Парсим JSON
                    d = json.loads(line)

                    # Извлекаем текст с санацией
                    text = extract_text(d)
                    if not text.strip():
                        continue

                    # Извлечение метаданных для нового формата
                    doc_id = d.get("doc_id", "")
                    title = d.get("document", {}).get("title", "") or d.get("title", "")
                    url = d.get("source", {}).get("display_url", "") or d.get("url", "")

                    # Извлечение lifecycle данных
                    lifecycle = d.get("lifecycle", {})
                    published_ts = lifecycle.get("published_ts", 0) or d.get("published_ts", 0) or 0
                    academic_year = lifecycle.get("academic_year")  # может быть None
                    is_active = lifecycle.get("is_active", True)

                    content_hash = d.get("hashes", {}).get("normalized_text_sha256", "") or d.get("content_hash", "")

                    # Формируем базовый payload
                    payload = {
                        "doc_id": doc_id,
                        "source_id": source,
                        "url": url,
                        "title": title,
                        "lang": d.get("lang", "ru"),
                        "published_ts": published_ts,
                        "content_hash": content_hash,
                        "index_in_rag": d.get("index_in_rag", True),
                        "text": text,
                    }

                    # Добавляем academic_year только если он есть (чтобы не сломать API)
                    if academic_year is not None:
                        payload["academic_year"] = academic_year

                    # Добавляем is_active (всегда есть, но на всякий случай)
                    if is_active is not None:
                        payload["is_active"] = is_active

                    # Отправляем запрос
                    req = urllib.request.Request(
                        BASE + "/api/v1/documents",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )

                    try:
                        response = urllib.request.urlopen(req, timeout=180)
                        r = json.loads(response.read())
                        ok += r.get("indexed", 0)
                        fail += r.get("failed", 0)
                        chunks = r.get("chunks_upserted", 0)

                        # Показываем academic_year и is_active в выводе для контроля
                        year_str = f"year={academic_year}" if academic_year else "year=N/A"
                        active_str = f"active={is_active}"
                        print(f"[{i}] {title[:40]:40} {year_str} {active_str} chunks={chunks}")

                    except Exception as e:
                        fail += 1
                        logger.error(f"[{i}] Ошибка при отправке: {e}")
                        print(f"[{i}] FAIL {e}")

                except json.JSONDecodeError as e:
                    fail += 1
                    logger.error(f"[{i}] Ошибка парсинга JSON: {e}")
                    print(f"[{i}] FAIL JSON: {e}")
                except Exception as e:
                    fail += 1
                    logger.error(f"[{i}] Неожиданная ошибка: {e}")
                    print(f"[{i}] FAIL {e}")

    except FileNotFoundError:
        logger.error(f"Файл не найден: {path}")
        return 1
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1

    total_elapsed = time.time() - total_start
    logger.info(f"\nГотово за {total_elapsed:.2f}с")
    logger.info(f"source={source!r} indexed={ok} failed={fail}")
    print(f"\nГотово: source={source!r} indexed={ok} failed={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())