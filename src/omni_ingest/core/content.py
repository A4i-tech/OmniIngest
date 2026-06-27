"""Content URI construction for virtual content."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from urllib.parse import ParseResult

from pypdf import PdfReader, PdfWriter

from .model import ResolvedResource

if TYPE_CHECKING:
    from .pipeline import IngestionContext


def resource_pages_uri(pages: list[int]) -> str:
    return f"omni-ingest://resource/pages/{','.join(map(str, pages))}"


async def resource_page(parsed: ParseResult, ctx: IngestionContext[ResolvedResource]) -> bytes | None:
    if parsed.scheme != "omni-ingest" or parsed.netloc != "resource":
        return None

    parts = [part for part in parsed.path.split("/") if part]
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
