"""Очистка текста от проблемных Unicode-символов."""

import logging
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# Карта замен для проблемных символов
DEFAULT_REPLACEMENTS = {
    # Умные кавычки
    "\u2018": "'",  # левая одинарная
    "\u2019": "'",  # правая одинарная
    "\u201a": "'",  # нижняя одинарная
    "\u201b": "'",  # верхняя обратная одинарная
    "\u201c": '"',  # левая двойная
    "\u201d": '"',  # правая двойная
    "\u201e": '"',  # нижняя двойная
    "\u201f": '"',  # верхняя обратная двойная
    # Дефисы и тире
    "\u2010": "-",  # дефис
    "\u2011": "-",  # неразрывный дефис
    "\u2012": "-",  # цифровое тире
    "\u2013": "-",  # en-dash
    "\u2014": "-",  # em-dash
    "\u2015": "-",  # горизонтальная черта
    # Пробелы
    "\u00a0": " ",  # неразрывный пробел
    "\u2000": " ",  # en quad
    "\u2001": " ",  # em quad
    "\u2002": " ",  # en space
    "\u2003": " ",  # em space
    "\u2004": " ",  # three-per-em space
    "\u2005": " ",  # four-per-em space
    "\u2006": " ",  # six-per-em space
    "\u2007": " ",  # figure space
    "\u2008": " ",  # punctuation space
    "\u2009": " ",  # thin space
    "\u200a": " ",  # hair space
    "\u202f": " ",  # narrow no-break space
    # Разделители строк
    "\u2028": "\n",  # Line Separator
    "\u2029": "\n",  # Paragraph Separator
}


def sanitize_text(
    text: str,
    normalize_form: str = "NFKC",
    extra_replacements: dict[str, str] | None = None,
) -> str:
    """
    Очистка строки от проблемных Unicode-символов.

    Args:
        text: Исходная строка
        normalize_form: Форма нормализации Unicode (NFKC, NFC, NFD, NFKD)
        extra_replacements: Дополнительные замены (словарь {старый: новый})

    Returns:
        Очищенная строка
    """
    if not isinstance(text, str):
        return text

    # NFKC нормализация
    text = unicodedata.normalize(normalize_form, text)

    # Базовые замены
    replacements = DEFAULT_REPLACEMENTS.copy()
    if extra_replacements:
        replacements.update(extra_replacements)

    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)

    return text


def sanitize_record(
    record: dict[str, Any],
    normalize_form: str = "NFKC",
    extra_replacements: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Рекурсивная очистка всех строковых полей в JSON-записи.

    Args:
        record: Исходный словарь
        normalize_form: Форма нормализации Unicode
        extra_replacements: Дополнительные замены

    Returns:
        Очищенный словарь
    """
    if isinstance(record, dict):
        return {
            k: sanitize_record(v, normalize_form, extra_replacements)
            for k, v in record.items()
        }
    elif isinstance(record, list):
        return [
            sanitize_record(item, normalize_form, extra_replacements)
            for item in record
        ]
    elif isinstance(record, str):
        return sanitize_text(record, normalize_form, extra_replacements)
    else:
        return record


def sanitize_jsonl_file(
    input_path: str,
    output_path: str | None = None,
    normalize_form: str = "NFKC",
    extra_replacements: dict[str, str] | None = None,
    strict_mode: bool = False,
) -> int:
    """
    Очистка JSONL-файла на уровне строк и полей.

    Args:
        input_path: Путь к исходному файлу
        output_path: Путь к выходному файлу (если None — перезаписывает исходный)
        normalize_form: Форма нормализации Unicode
        extra_replacements: Дополнительные замены
        strict_mode: Если True — падать при ошибках, иначе пропускать строки

    Returns:
        Количество обработанных записей
    """
    import json

    output_path = output_path or input_path
    processed = 0
    errors = []

    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    sanitized_lines = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
            sanitized = sanitize_record(record, normalize_form, extra_replacements)
            sanitized_lines.append(json.dumps(sanitized, ensure_ascii=False))
            processed += 1
        except json.JSONDecodeError as e:
            # Пробуем очистить саму строку
            sanitized_line = sanitize_text(line, normalize_form, extra_replacements)
            try:
                record = json.loads(sanitized_line)
                sanitized = sanitize_record(record, normalize_form, extra_replacements)
                sanitized_lines.append(json.dumps(sanitized, ensure_ascii=False))
                processed += 1
                logger.warning(f"Строка {i} была исправлена: {e}")
            except json.JSONDecodeError as e2:
                if strict_mode:
                    raise ValueError(f"Не удалось исправить строку {i}: {e2}") from e2
                else:
                    logger.error(f"Невалидный JSON в строке {i}, строка пропущена: {e2}")
                    errors.append({"line": i, "error": str(e2)})

    # Запись результата
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sanitized_lines))
        if sanitized_lines:
            f.write("\n")

    if errors:
        logger.warning(f"Пропущено {len(errors)} строк с ошибками")

    return processed