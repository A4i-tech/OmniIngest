"""OCR, builder, and factory definitions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from pydantic import BaseModel

from .model import ByteContent, ResolvedResource

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext


class Figure(BaseModel):
    """A figure extracted from a document by an OCR engine, ready to become a `KnowledgeItem`."""

    id: str | None = None
    page: int | None = None
    caption: str | None = None
    bbox: list[int | None] | None = None
    image: str  # base64-encoded
    content_type: str = "image/png"


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


@dataclass(kw_only=True)
class OcrBuilderParams:
    ctx: IngestionContext[ResolvedResource]
    format: OcrOutputFormat
    concurrency: int
    src_lang: str | None = None
    dst_lang: str | None = None


T = TypeVar("T", bound=ByteContent)
Ocr: TypeAlias = Callable[[AsyncIterator[T]], AsyncIterator[tuple[T, ByteContent]]]
OcrBuilder: TypeAlias = Callable[[OcrBuilderParams], Ocr]
OcrFactory: TypeAlias = Callable[[str, OcrBuilderParams], Ocr]


def simple(mapping: dict[str, OcrBuilder]) -> OcrFactory:
    """
    Build a factory that dispatches by some name to a fixed mapping.

    Args:
        mapping: Builders indexed by their name.

    Example:
        ```python
        from omni_ingest.core.ocr import simple

        def llm_ocr_builder(params: OcrBuilderParams) -> Ocr:
            async def ocr(items):
                async for item in items:
                    data = await item.decode(ctx)
                    ocr = await llm("Transcribe to markdown", data)
                    yield item, ByteContent.from_bytes(ocr.encode("utf-8"), "text/markdown")
            return ocr

        factory = simple({"llm": llm_ocr_builder})
        ```
    """

    def factory(name: str, params: OcrBuilderParams) -> Ocr:
        if name not in mapping:
            raise ValueError(f"Unknown OCR builder: {name}")
        return mapping[name](params)

    return factory
