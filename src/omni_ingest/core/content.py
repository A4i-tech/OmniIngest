"""Content URI resolution for virtual URIs."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, urlparse

from pypdf import PdfReader, PdfWriter

from omni_ingest.core.model import ResolvedResource

if TYPE_CHECKING:
    from .pipeline import IngestionContext


def resource_pages_uri(pages: list[int]) -> str:
    return f"omni-ingest://resource/pages/{','.join(map(str, pages))}"


async def resource_pages(parsed: ParseResult, ctx: IngestionContext[ResolvedResource]) -> bytes | None:
    if parsed.scheme != "omni-ingest" or parsed.netloc != "resource":
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 2 or parts[0] != "pages":
        return None

    pages = [int(page) for page in parts[1].split(",")]
    if any(page < 1 for page in pages):
        raise ValueError(f"Invalid omni-ingest page numbers: {pages}")

    source = io.BytesIO(await ctx.resource.content(ctx))
    with source, PdfReader(source) as reader, io.BytesIO() as output, PdfWriter() as writer:
        for page in pages:
            writer.add_page(reader.pages[page - 1])
        for pdf_page in writer.pages:
            pdf_page.compress_content_streams()
        writer.write(output)
        return output.getvalue()


class ContentResolver(ABC):
    """Resolve virtual content URIs into bytes during ingestion.

    `ByteContent.content(ctx)` delegates to `ctx.content_resolver` whenever an
    item has `content_uri` instead of inline `raw_content`. Implementations can
    use full ingestion context to resolve routes against the source resource,
    object stores, caches, or other pipeline-scoped state.
    """

    @abstractmethod
    async def resolve(self, uri: str, ctx: IngestionContext[ResolvedResource]) -> bytes:
        """
        Return the resolved bytes for supported URIs, or raise `ValueError`
        when no route can handle the URI.
        """
        ...


class CompositeContentResolver(ContentResolver):

    def __init__(self, resolvers: list[Callable[[ParseResult, IngestionContext[ResolvedResource]], Awaitable[bytes | None]]]):
        self.parsers = resolvers


    async def resolve(self, uri: str, ctx: IngestionContext[ResolvedResource]) -> bytes:
        parsed = urlparse(uri)
        for p in self.parsers:
            if result := await p(parsed, ctx):
                return result
        raise ValueError(f"Unsupported content URI: {uri}")


def default():
    return CompositeContentResolver(resolvers=[resource_pages])
