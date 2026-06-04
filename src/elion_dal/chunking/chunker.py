"""Token-aware чанкинг под токенайзер BGE-M3.

Рекурсивно режем по естественным границам (абзацы -> строки -> предложения ->
слова), измеряя длину в токенах BGE-M3, с перекрытием. Токенайзер грузится лениво
(один раз) и не требует GPU. Для тестов длину можно подменить через length_fn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    token_count: int


# Пресеты границ нарезки (передаются в RecursiveCharacterTextSplitter):
#   structured — режем по естественным границам (абзацы → строки → предложения → слова);
#   token      — жёстко: абзац/строка/слово/символ, без учёта границ предложений.
SEPARATOR_PRESETS: dict[str, list[str]] = {
    "structured": ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    "token": ["\n\n", "\n", " ", ""],
}
DEFAULT_SEPARATOR_MODE = "structured"


@lru_cache(maxsize=1)
def _tokenizer(model_name: str = "BAAI/bge-m3"):
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    # Глушим предупреждение "sequence length is longer than max": мы используем
    # токенайзер только для подсчёта длины, без прогона через модель.
    hf_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(model_name)


class Chunker:
    def __init__(
        self,
        chunk_tokens: int = 400,
        chunk_overlap: int = 64,
        model_name: str = "BAAI/bge-m3",
        min_tokens: int = 0,
        separator_mode: str = DEFAULT_SEPARATOR_MODE,
        length_fn: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_tokens:
            raise ValueError("chunk_overlap должен быть меньше chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self._model_name = model_name
        # Фильтр мусора: чанки короче min_tokens дропаются (0 = выкл).
        self.min_tokens = max(0, min_tokens)
        # Неизвестный режим тихо откатываем на structured (валидация на входе).
        self.separator_mode = (
            separator_mode if separator_mode in SEPARATOR_PRESETS else DEFAULT_SEPARATOR_MODE
        )
        # length_fn задаётся в тестах (offline); иначе считаем токенами BGE-M3.
        self._length_fn = length_fn

    def _count(self, text: str) -> int:
        if self._length_fn is not None:
            return self._length_fn(text)
        return len(_tokenizer(self._model_name).encode(text, add_special_tokens=False))

    def count_tokens(self, text: str) -> int:
        """Длина текста в токенах (без двойного счёта overlap)."""
        return self._count(text)

    def split(self, text: str) -> list[Chunk]:
        """Нарезать текст на чанки.

        Куски короче min_tokens отбрасываются (фильтр мусора), а оставшиеся
        перенумеровываются подряд — chunk_id = parent_id#index остаётся
        детерминированным и непрерывным. ВНИМАНИЕ: при агрессивном min_tokens у
        короткой секции могут отсеяться все дети, тогда parent окажется без точек
        в Qdrant и не найдётся поиском — это и есть смысл фильтра. При min_tokens=0
        поведение полностью совпадает с прежним (обратная совместимость).
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text = (text or "").strip()
        if not text:
            return []

        separators = SEPARATOR_PRESETS.get(
            self.separator_mode, SEPARATOR_PRESETS[DEFAULT_SEPARATOR_MODE]
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count,
            separators=separators,
            keep_separator=True,
        )
        chunks: list[Chunk] = []
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            tc = self._count(piece)
            if self.min_tokens and tc < self.min_tokens:
                continue  # фильтр мусора
            chunks.append(Chunk(index=len(chunks), text=piece, token_count=tc))
        return chunks
