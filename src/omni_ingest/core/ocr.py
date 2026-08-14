"""OCR engine abstraction and factory wiring."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeAlias

from .model import ByteContent, ResolvedResource

if TYPE_CHECKING:
    from .pipeline import IngestionContext

Ocr: TypeAlias = Callable[[ByteContent], Awaitable[ByteContent]]

OcrEngineBuilder: TypeAlias = Callable[["IngestionContext[ResolvedResource]", str | None, str | None], Ocr]
OcrFactory: TypeAlias = Callable[[str, "IngestionContext[ResolvedResource]", str | None, str | None], Ocr]


def build_simple_factory(mapping: dict[str, OcrEngineBuilder]) -> OcrFactory:
    """Build an `OcrFactory` that dispatches by engine name to a fixed mapping."""

    def factory(engine: str, ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
        if engine not in mapping:
            raise ValueError(f"Unknown OCR engine: {engine}")
        return mapping[engine](ctx, src_lang, dst_lang)

    return factory
