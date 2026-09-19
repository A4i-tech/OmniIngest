"""Content URI construction for virtual content."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, parse_qs, quote

from pypdf import PdfReader, PdfWriter

from .model import ResolvedResource

if TYPE_CHECKING:
    from .pipeline import IngestionContext


def resource_pages_uri(pages: list[int]) -> str:
    return f"omni-ingest://resource/pages/{','.join(map(str, pages))}"


def content_chain_uri(operation: str, inputs: list[str]) -> str:
    recipe = json.dumps({"op": operation, "inputs": inputs}, separators=(",", ":"), sort_keys=True)
    return f"omni-ingest://chain?recipe={quote(recipe, safe='')}"


async def resource_page(parsed: ParseResult, ctx: IngestionContext[ResolvedResource]) -> bytes | None:
    if parsed.scheme != "omni-ingest" or parsed.netloc != "resource":
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0] != "pages":
        return None

    pages = [int(page) for page in parts[1].split(",")]
    if any(page < 1 for page in pages):
        raise ValueError(f"Invalid omni-ingest page numbers: {pages}")

    source = io.BytesIO(await ctx.resource.content.read(ctx))
    with source, PdfReader(source) as reader, io.BytesIO() as output, PdfWriter() as writer:
        for page in pages:
            writer.add_page(reader.pages[page - 1])
        for pdf_page in writer.pages:
            pdf_page.compress_content_streams()
        writer.write(output)
        return output.getvalue()


async def content_chain(parsed: ParseResult, ctx: IngestionContext[ResolvedResource]) -> bytes | None:
    if parsed.scheme != "omni-ingest" or parsed.netloc != "chain":
        return None

    recipe = json.loads(parse_qs(parsed.query)["recipe"][0])
    if recipe.get("op") != "pdf.concat" or not isinstance(recipe.get("inputs"), list):
        raise ValueError(f"Unsupported content chain recipe: {recipe}")

    with io.BytesIO() as output, PdfWriter() as writer:
        for uri in recipe["inputs"]:
            with io.BytesIO(await ctx.content_resolver.resolve(uri, ctx)) as pdf, PdfReader(pdf) as reader:
                writer.append(reader)
        for page in writer.pages:
            page.compress_content_streams()
        writer.write(output)
        return output.getvalue()
