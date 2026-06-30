"""Фабрика для создания санитайзеров с настройками из config."""

from typing import Optional, Dict, Any
from .unicode import sanitize_text, sanitize_record, sanitize_jsonl_file


def get_sanitizer_config() -> Dict[str, Any]:
    """
    Получить настройки санации из глобальной конфигурации.

    Returns:
        Словарь с настройками
    """
    try:
        from ..config import get_settings
        settings = get_settings()
        return {
            "enabled": getattr(settings, "sanitize_enabled", True),
            "normalize_form": getattr(settings, "sanitize_normalize_form", "NFKC"),
            "strict_mode": getattr(settings, "sanitize_strict_mode", False),
            "log_warnings": getattr(settings, "sanitize_log_warnings", True),
        }
    except ImportError:
        # Если конфиг недоступен — возвращаем настройки по умолчанию
        return {
            "enabled": True,
            "normalize_form": "NFKC",
            "strict_mode": False,
            "log_warnings": True,
        }


def get_sanitizer() -> Dict[str, Any]:
    """
    Получить санитайзер с настройками.
    Алиас для get_sanitizer_config() для обратной совместимости.
    """
    return get_sanitizer_config()


def sanitize_text_with_config(text: str) -> str:
    """Очистка текста с настройками из config."""
    config = get_sanitizer_config()
    if not config["enabled"]:
        return text
    return sanitize_text(text, normalize_form=config["normalize_form"])


def sanitize_record_with_config(record: Dict) -> Dict:
    """Очистка записи с настройками из config."""
    config = get_sanitizer_config()
    if not config["enabled"]:
        return record
    return sanitize_record(record, normalize_form=config["normalize_form"])


def sanitize_jsonl_file_with_config(input_path: str, output_path: Optional[str] = None) -> int:
    """Очистка JSONL-файла с настройками из config."""
    config = get_sanitizer_config()
    if not config["enabled"]:
        # Если санация выключена — просто копируем файл
        import shutil
        shutil.copy2(input_path, output_path or input_path)
        # Подсчет строк
        with open(input_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    return sanitize_jsonl_file(
        input_path,
        output_path,
        normalize_form=config["normalize_form"],
        strict_mode=config["strict_mode"],
    )