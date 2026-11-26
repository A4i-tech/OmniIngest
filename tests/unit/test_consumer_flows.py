"""Pipelines shaped like common downstream YAML, run through the public API.

One flow runs `page_chunking` in pdf mode only; the other runs `page_chunking` then `rasterize` at 180 dpi
for PDFs whose text layer is unreadable (legacy font encodings). These tests pin what those flows see (item count, order, page metadata, content type, what survives
rasterizing) so a PDF-backend swap cannot break them silently.
"""

import io
from pathlib import Path
from uuid import uuid4

import pytest
from pypdf import PdfReader

from omni_ingest.core.model import ByteContent, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, create_pipeline_from_config
from omni_ingest.port.metadata_store import NullMetadataStore

from ._pdf_helpers import PDFS

CHUNK_ONLY = """
pipeline_id: chunk_only
steps:
- agent: page_chunking
  config:
    mode: pdf
"""

CHUNK_RASTERIZE = """
pipeline_id: chunk_rasterize
steps:
- agent: page_chunking
  config:
    mode: pdf
- agent: rasterize
  config:
    foreach: .items[]
    input: '[.id]'
    dpi: 180
"""


async def _run(tmp_path: Path, yaml_text: str, pdf: Path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml_text)
    runner = create_pipeline_from_config(config)
    ctx = IngestionContext(
        tenant_id=uuid4(),
        store=NullMetadataStore(),
        resource=ResolvedResource(uri=pdf.name, content=ByteContent.from_bytes(pdf.read_bytes(), content_type="application/pdf")),
    )
    results = await runner.run(ctx)
    return ctx, results


def _image_sizes(page) -> list[tuple[int, int]]:
    xobjects = page["/Resources"].get("/XObject", {})
    return [(int(x["/Width"]), int(x["/Height"])) for x in (v.get_object() for v in xobjects.values()) if x.get("/Subtype") == "/Image"]


async def _pdf_bytes(ctx, item) -> bytes:
    return await item.content.read(ctx)


@pytest.mark.asyncio
async def test_chunk_only_page_chunking_yields_one_pdf_item_per_page(tmp_path):
    ctx, results = await _run(tmp_path, CHUNK_ONLY, PDFS / "multi_page.pdf")
    assert results[-1].status == StepStatus.SUCCESS, results[-1].error
    assert len(ctx.items) == 3
    assert [i.metadata["page"] for i in ctx.items] == [1, 2, 3]
    for item in ctx.items:
        assert await item.content.content_type(ctx) == "application/pdf"
        assert len(PdfReader(io.BytesIO(await _pdf_bytes(ctx, item))).pages) == 1


@pytest.mark.asyncio
async def test_chunk_rasterize_keeps_items_and_drops_text_layer(tmp_path):
    source = PDFS / "metadata_rich.pdf"
    before, _ = await _run(tmp_path, CHUNK_ONLY, source)
    ctx, results = await _run(tmp_path, CHUNK_RASTERIZE, source)
    assert results[-1].status == StepStatus.SUCCESS, results[-1].error
    assert [i.metadata["page"] for i in ctx.items] == [i.metadata["page"] for i in before.items] == [1, 2, 3]
    for item, original in zip(ctx.items, before.items, strict=True):
        assert await item.content.content_type(ctx) == "application/pdf"
        page = PdfReader(io.BytesIO(await _pdf_bytes(ctx, item))).pages[0]
        reference = PdfReader(io.BytesIO(await _pdf_bytes(before, original))).pages[0]
        assert page.extract_text().strip() == ""  # text layer gone, what the vision model must read is pixels
        assert reference.extract_text().strip() != ""
        assert (float(page.mediabox.width), float(page.mediabox.height)) == pytest.approx((float(reference.mediabox.width), float(reference.mediabox.height)), abs=1.0)
        assert len(_image_sizes(page)) == 1  # one full-page image


@pytest.mark.asyncio
async def test_chunk_rasterize_resolution_follows_dpi(tmp_path):
    ctx, _ = await _run(tmp_path, CHUNK_RASTERIZE, PDFS / "one_image.pdf")
    page = PdfReader(io.BytesIO(await _pdf_bytes(ctx, ctx.items[0]))).pages[0]
    # 400x200 pt page at 180 dpi is 1000x500 px
    assert _image_sizes(page) == [(1000, 500)]
