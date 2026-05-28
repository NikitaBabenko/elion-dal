"""Token-aware чанкинг под токенайзер BGE-M3.

Рекурсивно режем по естественным границам (абзацы -> строки -> предложения ->
слова), измеряя длину в токенах BGE-M3, с перекрытием. Токенайзер грузится один
раз; быстрый и не требует GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    token_count: int


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
    ) -> None:
        if chunk_overlap >= chunk_tokens:
            raise ValueError("chunk_overlap должен быть меньше chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self._tok = _tokenizer(model_name)

    def _count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def split(self, text: str) -> list[Chunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text = (text or "").strip()
        if not text:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator=True,
        )
        pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]
        return [Chunk(index=i, text=p, token_count=self._count(p)) for i, p in enumerate(pieces)]
