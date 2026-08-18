"""OCR engine and factory definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias

from .model import ByteContent, ResolvedResource

if TYPE_CHECKING:
    from .pipeline import IngestionContext


class OcrOutputFormat(StrEnum):
    # TODO: This ought to be a proper data class instead of this hacky enum. It stays enum
    # because making this a proper extendable reegistry confuses the implementation boundry
    # a bit. In future, each output format must also have their own post-validator function.

    HTML = "html", "Standalone HTML", "text/html"
    JSON = "json", "JSON", "application/json"
    MARKDOWN = "markdown", "Markdown", "text/markdown"
    TEXT = "text", "plain text", "text/plain"

    label: str  # meant to be human-readable, passed in certain llm prompts
    mime_type: str  # used when reinventing ByteContent items

    def __new__(cls, value: str, label: str, mime_type: str):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.label = label
        obj.mime_type = mime_type
        return obj


Ocr: TypeAlias = Callable[[ByteContent], Awaitable[ByteContent]]
OcrEngineBuilder: TypeAlias = Callable[["IngestionContext[ResolvedResource]", OcrOutputFormat, str | None, str | None], Ocr]
OcrFactory: TypeAlias = Callable[[str, "IngestionContext[ResolvedResource]", OcrOutputFormat, str | None, str | None], Ocr]


def simple(mapping: dict[str, OcrEngineBuilder]) -> OcrFactory:
    """
    Build a factory that dispatches by engine name to a fixed mapping.

    Args:
        mapping: Engines indexed by their name.

    Example:
        ```python
        from omni_ingest.core.ocr import simple

        def llm_ocr_builder(ctx, out_fmt, src_lang, dst_lang) -> Ocr:
            async def ocr(item: ByteContent) -> ByteContent:
                data = await item.decode(ctx)
                ocr = await llm("Transcribe to markdown", data)
                return ByteContent.raw(ocr.encode("utf-8"), "text/markdown")
            return ocr

        factory = simple({"llm": llm_ocr_builder})
        ```
    """

    def factory(engine: str, ctx: IngestionContext[ResolvedResource], out_fmt: OcrOutputFormat, src_lang: str | None, dst_lang: str | None) -> Ocr:
        if engine not in mapping:
            raise ValueError(f"Unknown OCR engine: {engine}")
        return mapping[engine](ctx, out_fmt, src_lang, dst_lang)

    return factory
