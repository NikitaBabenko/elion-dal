"""Модуль санации данных — очистка от проблемных Unicode-символов."""

from .factory import (
    get_sanitizer_config,
    sanitize_jsonl_file_with_config,
    sanitize_record_with_config,
    sanitize_text_with_config,
)
from .unicode import sanitize_jsonl_file, sanitize_record, sanitize_text

__all__ = [
    "sanitize_text",
    "sanitize_record",
    "sanitize_jsonl_file",
    "get_sanitizer_config",
    "sanitize_text_with_config",
    "sanitize_record_with_config",
    "sanitize_jsonl_file_with_config",
]