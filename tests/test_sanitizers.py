"""Тесты для модуля санации данных."""

import json
from pathlib import Path

import pytest

from elion_dal.sanitizers import (
    sanitize_jsonl_file,
    sanitize_record,
    sanitize_text,
)
from elion_dal.sanitizers.factory import (
    sanitize_record_with_config,
    sanitize_text_with_config,
)


class TestSanitizeText:
    """Тесты для функции sanitize_text."""

    def test_sanitize_text_smart_quotes(self):
        """Тест: умные кавычки должны заменяться на обычные."""
        # Двойные умные кавычки (U+201C, U+201D) -> обычные
        assert sanitize_text('“Привет”') == '"Привет"'
        assert sanitize_text('„Привет“') == '"Привет"'

        # Одинарные умные кавычки (U+2018, U+2019) -> обычные
        assert sanitize_text('‘Привет’') == "'Привет'"
        assert sanitize_text('‚Привет‘') == "'Привет'"

        # Смешанные
        assert sanitize_text('“Привет” и ‘мир’') == '"Привет" и \'мир\''

        # Кавычки-елочки НЕ заменяем (это валидные символы)
        # Просто проверяем, что они не меняются
        assert sanitize_text('«Привет»') == '«Привет»'

    def test_sanitize_text_line_separators(self):
        """Тест: нестандартные разделители строк должны заменяться на \n."""
        text = 'line1\u2028line2\u2029line3'
        expected = 'line1\nline2\nline3'
        assert sanitize_text(text) == expected

    def test_sanitize_text_non_breaking_space(self):
        """Тест: неразрывные пробелы должны заменяться на обычные."""
        assert sanitize_text('Hello\u00a0World') == 'Hello World'
        assert sanitize_text('Hello\u2009World') == 'Hello World'

    def test_sanitize_text_dashes(self):
        """Тест: длинные тире должны заменяться на дефис."""
        assert sanitize_text('en–dash') == 'en-dash'
        assert sanitize_text('em—dash') == 'em-dash'
        assert sanitize_text('non‑breaking‑hyphen') == 'non-breaking-hyphen'

    def test_sanitize_text_mixed(self):
        """Тест: смешанные проблемные символы."""
        text = '“Hello”\u2028World\u00a0!—test'
        expected = '"Hello"\nWorld !-test'
        assert sanitize_text(text) == expected

    def test_sanitize_text_non_string(self):
        """Тест: не-строки должны возвращаться без изменений."""
        assert sanitize_text(None) is None
        assert sanitize_text(123) == 123
        assert sanitize_text(3.14) == 3.14
        assert sanitize_text(True) is True

    def test_sanitize_text_empty(self):
        """Тест: пустая строка."""
        assert sanitize_text("") == ""
        assert sanitize_text("   ") == "   "


class TestSanitizeRecord:
    """Тесты для функции sanitize_record."""

    def test_sanitize_record_flat(self):
        """Тест: плоский словарь."""
        record = {
            "title": "“Hello”\u2028World",
            "description": "Test\u00a0with\u2029dash—test",
        }
        expected = {
            "title": '"Hello"\nWorld',
            "description": "Test with\ndash-test",
        }
        assert sanitize_record(record) == expected

    def test_sanitize_record_nested(self):
        """Тест: вложенный словарь."""
        record = {
            "title": "Hello\u2028World",
            "content": {
                "text": "This\u00a0is\u2028a\u2029test",
                "meta": {"tag": "“important”"}
            },
            "tags": ["tag1\u2028tag2", "tag3"]
        }
        expected = {
            "title": "Hello\nWorld",
            "content": {
                "text": "This is\na\ntest",
                "meta": {"tag": '"important"'}
            },
            "tags": ["tag1\ntag2", "tag3"]
        }
        assert sanitize_record(record) == expected

    def test_sanitize_record_with_lists(self):
        """Тест: записи со списками внутри."""
        record = {
            "items": [
                {"name": "Item\u00a01"},
                {"name": "Item\u20282"},
                "simple\u2029string"
            ]
        }
        expected = {
            "items": [
                {"name": "Item 1"},
                {"name": "Item\n2"},
                "simple\nstring"
            ]
        }
        assert sanitize_record(record) == expected

    def test_sanitize_record_non_dict(self):
        """Тест: не-словари должны возвращаться без изменений."""
        assert sanitize_record([1, 2, 3]) == [1, 2, 3]
        assert sanitize_record("test") == "test"
        assert sanitize_record(None) is None


class TestSanitizeJSONL:
    """Тесты для функции sanitize_jsonl_file."""

    def test_sanitize_jsonl_basic(self, tmp_path):
        """Тест: базовая очистка JSONL-файла."""
        # Создаем тестовый файл с проблемными символами
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"id": 1, "text": "Hello\u2028World"}\n'
            '{"id": 2, "text": "“Test” with\u00a0space"}\n',
            encoding="utf-8"
        )

        # Очищаем
        output_file = tmp_path / "output.jsonl"
        count = sanitize_jsonl_file(str(input_file), str(output_file))

        # Проверяем результат
        assert count == 2

        with open(output_file, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        data1 = json.loads(lines[0])
        assert data1["text"] == "Hello\nWorld"

        data2 = json.loads(lines[1])
        assert data2["text"] == '"Test" with space'

    def test_sanitize_jsonl_inplace(self, tmp_path):
        """Тест: очистка файла на месте (перезапись)."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"id": 1, "text": "Hello\u2028World"}\n',
            encoding="utf-8"
        )

        # Очищаем на месте
        count = sanitize_jsonl_file(str(input_file))

        assert count == 1

        with open(input_file, encoding="utf-8") as f:
            data = json.loads(f.readline())

        assert data["text"] == "Hello\nWorld"

    def test_sanitize_jsonl_invalid_json(self, tmp_path):
        """Тест: обработка невалидного JSON."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"id": 1, "text": "Hello World"}\n'
            'invalid json line\n'
            '{"id": 3, "text": "Test"}\n',
            encoding="utf-8"
        )

        output_file = tmp_path / "output.jsonl"
        count = sanitize_jsonl_file(str(input_file), str(output_file), strict_mode=False)

        # Должны обработаться только валидные строки
        assert count == 2

        with open(output_file, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == 1
        assert json.loads(lines[1])["id"] == 3

    def test_sanitize_jsonl_strict_mode(self, tmp_path):
        """Тест: strict_mode=True должен вызывать ошибку."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"id": 1, "text": "Hello"}\n'
            'invalid json\n',
            encoding="utf-8"
        )

        with pytest.raises(ValueError):
            sanitize_jsonl_file(str(input_file), strict_mode=True)


class TestSanitizerFactory:
    """Тесты для фабрики санитайзеров с конфигом."""

    def test_sanitize_text_with_config(self):
        """Тест: sanitize_text_with_config использует настройки из config."""
        # Должно работать с дефолтными настройками
        result = sanitize_text_with_config("Hello\u2028World")
        assert result == "Hello\nWorld"

    def test_sanitize_record_with_config(self):
        """Тест: sanitize_record_with_config использует настройки из config."""
        record = {"text": "Hello\u2028World"}
        result = sanitize_record_with_config(record)
        assert result["text"] == "Hello\nWorld"


# Интеграционный тест (опционально, можно пропустить если нет файла)
@pytest.mark.skip(reason="Требует реального файла ready_for_indexing.jsonl")
class TestRealFile:
    """Тесты на реальном файле ready_for_indexing.jsonl."""

    def test_real_file_sanitization(self):
        """Проверка, что реальный файл очищается без ошибок."""
        input_path = Path("ready_for_indexing.jsonl")
        if not input_path.exists():
            pytest.skip("Файл ready_for_indexing.jsonl не найден")

        output_path = Path("ready_for_indexing_sanitized.jsonl")
        count = sanitize_jsonl_file(str(input_path), str(output_path))

        assert count > 0
        assert output_path.exists()

        # Проверяем, что все строки валидны
        with open(output_path, encoding="utf-8") as f:
            for _i, line in enumerate(f, 1):
                if line.strip():
                    data = json.loads(line)
                    assert isinstance(data, dict)
                    # Проверяем, что нет проблемных символов
                    text = str(data)
                    assert "\u2028" not in text
                    assert "\u2029" not in text
                    assert "\u201c" not in text
                    assert "\u201d" not in text
                    assert "\u00a0" not in text