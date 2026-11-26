"""Characterization tests for PDF image extraction, rasterization and image output.

These pin the behaviour of the PDF backend (pypdfium2) so a backend change cannot alter results silently.
Results are inspected with pypdf and the standard library only, so the same file runs unchanged against any backend.

Geometry uses a 1pt tolerance, pixel sizes 2px.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pytest
from pypdf import PdfReader

from omni_ingest.agent.document import ImageExtractionAgent, RasterizeAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import MarkdownDirOutput
from omni_ingest.core.pipeline import IngestionContext

from ._pdf_helpers import PDFS, RESOURCES, TOL, assert_bbox, extract, image_bytes, load, make_parent, page_image_sizes, page_sizes, page_texts, png_size

# --------------------------------------------------------------------------------------
# ImageExtractionAgent: embedded images
# --------------------------------------------------------------------------------------


def test_single_embedded_image_bbox_page_and_pixels():
    data = load(PDFS / "one_image.pdf")
    _, items = extract(data, include_vector=False, padding=0, dpi=72)

    [item] = items
    assert item.metadata["kind"] == "image"
    assert item.metadata["page"] == 1
    assert_bbox(item.metadata["bbox"], (40, 60, 360, 140))
    assert item.metadata["parent_id"]


@pytest.mark.asyncio
async def test_extracted_image_is_png_sized_to_region_at_dpi():
    data = load(PDFS / "one_image.pdf")
    _, [at_72] = extract(data, include_vector=False, padding=0, dpi=72)
    _, [at_144] = extract(data, include_vector=False, padding=0, dpi=144)

    w72, h72 = png_size(await image_bytes(at_72))
    w144, h144 = png_size(await image_bytes(at_144))
    assert abs(w72 - 320) <= 2 and abs(h72 - 80) <= 2
    assert abs(w144 - 640) <= 2 and abs(h144 - 160) <= 2


def test_padding_grows_bbox_and_is_clamped_to_page():
    data = load(PDFS / "one_image.pdf")
    _, [padded] = extract(data, include_vector=False, padding=10, dpi=72)
    assert_bbox(padded.metadata["bbox"], (30, 50, 370, 150))

    _, [clamped] = extract(data, include_vector=False, padding=500, dpi=72)
    assert_bbox(clamped.metadata["bbox"], (0, 0, 400, 200))


def test_multi_page_images_in_page_order_with_page_numbers():
    data = load(PDFS / "multi_page.pdf")
    _, items = extract(data, include_vector=False, padding=0, dpi=72, deduplicate=False)

    pages = [item.metadata["page"] for item in items]
    assert pages == [1, 1, 2, 2]
    assert_bbox(items[0].metadata["bbox"], (20, 20, 180, 120))
    assert_bbox(items[1].metadata["bbox"], (220, 150, 380, 270))


def test_deduplicate_removes_identical_images_on_same_page_only():
    data = load(PDFS / "multi_page.pdf")
    _, deduped = extract(data, include_vector=False, padding=0, dpi=72, deduplicate=True)
    _, all_items = extract(data, include_vector=False, padding=0, dpi=72, deduplicate=False)

    # Page 2 holds two identical images (same pixels, same size): one survives.
    assert [i.metadata["page"] for i in all_items] == [1, 1, 2, 2]
    assert [i.metadata["page"] for i in deduped] == [1, 1, 2]


def test_regions_outside_area_thresholds_are_skipped():
    """Page 3 has a ~0.08% image (below the 0.1% floor) and a ~95% image (above the 80% ceiling)."""
    data = load(PDFS / "multi_page.pdf")
    _, items = extract(data, include_vector=False, padding=0, dpi=72, deduplicate=False)
    assert 3 not in [item.metadata["page"] for item in items]


def test_item_ids_are_unique_and_parent_linked():
    data = load(PDFS / "multi_page.pdf")
    parent, items = extract(data, include_vector=False, padding=0, dpi=72, deduplicate=False)
    assert len({item.id for item in items}) == len(items)
    assert {item.metadata["parent_id"] for item in items} == {str(parent.id)}


def test_non_pdf_bytes_raise():
    with pytest.raises(Exception):  # noqa: B017 - exception type is backend specific
        extract(b"not a pdf", include_vector=False)


# --------------------------------------------------------------------------------------
# ImageExtractionAgent: vector drawings and tables
# --------------------------------------------------------------------------------------


def test_vector_figure_found_and_table_excluded_by_default():
    data = load(PDFS / "vector_shapes.pdf")
    _, items = extract(data, include_vector=True, include_tables=False, padding=0, dpi=72)

    [figure] = items
    assert figure.metadata["page"] == 1
    assert_bbox(figure.metadata["bbox"], (50, 60, 200, 160))


def test_include_tables_keeps_table_region():
    data = load(PDFS / "vector_shapes.pdf")
    _, items = extract(data, include_vector=True, include_tables=True, padding=0, dpi=72)

    boxes = sorted(item.metadata["bbox"] for item in items)
    assert len(boxes) == 2
    assert_bbox(boxes[0], (50, 60, 200, 160))
    assert_bbox(boxes[1], (50, 240, 410, 330))


def test_include_vector_false_ignores_drawings():
    data = load(PDFS / "vector_shapes.pdf")
    _, items = extract(data, include_vector=False, padding=0, dpi=72)
    assert items == []


def test_real_pdf_with_figure():
    data = load(RESOURCES / "produce-report-with-figure.pdf")
    _, items = extract(data, include_vector=False, padding=0, dpi=72)

    [item] = items
    assert item.metadata["page"] == 1
    assert_bbox(item.metadata["bbox"], (72, 149.6, 400, 390.4))


def test_real_pdf_table_region_excluded_by_default_and_kept_when_requested():
    data = load(RESOURCES / "when-tomorrow-was-yesterday.pdf")

    _, default_items = extract(data, include_vector=True, include_tables=False, padding=0, dpi=72)
    assert default_items == []

    _, with_tables = extract(data, include_vector=True, include_tables=True, padding=0, dpi=72)
    [item] = with_tables
    assert item.metadata["page"] == 4
    assert_bbox(item.metadata["bbox"], (72, 244, 540, 362))


# --------------------------------------------------------------------------------------
# ImageExtractionAgent: parent PDF modes
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preserve_leaves_parent_bytes_untouched():
    data = load(PDFS / "multi_page.pdf")
    parent, _ = extract(data, include_vector=False, padding=0, dpi=72, mode="preserve")
    assert await image_bytes(parent) == data


@pytest.mark.asyncio
async def test_redact_removes_images_and_writes_markers_for_every_item():
    data = load(PDFS / "multi_page.pdf")
    parent, items = extract(data, include_vector=False, padding=0, dpi=72, mode="redact", deduplicate=False)

    redacted = await image_bytes(parent)
    assert redacted != data
    assert page_sizes(redacted) == page_sizes(data)
    assert page_image_sizes(redacted)[0] == [] and page_image_sizes(redacted)[1] == []
    text = "\n".join(page_texts(redacted))
    for item in items:
        assert f'<image id="{item.id}"/>' in text


@pytest.mark.asyncio
async def test_caption_keeps_images_and_adds_markers():
    data = load(PDFS / "multi_page.pdf")
    parent, items = extract(data, include_vector=False, padding=0, dpi=72, mode="caption", deduplicate=False)

    captioned = await image_bytes(parent)
    assert captioned != data
    assert page_sizes(captioned) == page_sizes(data)
    assert [len(p) for p in page_image_sizes(captioned)] == [len(p) for p in page_image_sizes(data)]
    text = "\n".join(page_texts(captioned))
    for item in items:
        assert f'<image id="{item.id}"/>' in text


@pytest.mark.asyncio
async def test_redact_and_caption_preserve_existing_text():
    data = load(PDFS / "vector_shapes.pdf")
    assert "Quarterly report" in page_texts(data)[0]
    for mode in ("redact", "caption"):
        parent, _ = extract(data, include_vector=True, include_tables=False, padding=0, dpi=72, mode=mode)
        text = page_texts(await image_bytes(parent))[0]
        assert "Quarterly report" in text, mode
        assert "r1c1" in text, mode  # table text is outside the redacted figure region


@pytest.mark.asyncio
async def test_parent_content_type_stays_pdf_after_modification():
    data = load(PDFS / "one_image.pdf")
    parent = make_parent(data)
    ImageExtractionAgent(mode="redact", include_vector=False, padding=0, dpi=72)._extract_pdf(parent, data)
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[parent])
    assert await parent.content.content_type(ctx) == "application/pdf"


# --------------------------------------------------------------------------------------
# RasterizeAgent
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["one_image.pdf", "multi_page.pdf", "vector_shapes.pdf"])
async def test_rasterize_keeps_page_count_and_size_and_drops_text_layer(name):
    data = load(PDFS / name)
    item = make_parent(data)
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)

    out = await image_bytes(item)
    assert out != data
    assert len(PdfReader(io.BytesIO(out)).pages) == len(PdfReader(io.BytesIO(data)).pages)
    for (w, h), (ow, oh) in zip(page_sizes(out), page_sizes(data), strict=True):
        assert abs(w - ow) <= TOL and abs(h - oh) <= TOL
    assert all(text.strip() == "" for text in page_texts(out))
    assert all(len(images) == 1 for images in page_image_sizes(out))


@pytest.mark.asyncio
async def test_rasterize_removes_text_layer_that_existed():
    data = load(PDFS / "vector_shapes.pdf")
    assert "Quarterly report" in page_texts(data)[0]
    item = make_parent(data)
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    assert page_texts(await image_bytes(item))[0].strip() == ""


@pytest.mark.asyncio
async def test_rasterize_dpi_controls_pixel_size():
    data = load(PDFS / "vector_shapes.pdf")  # 500x500pt
    sizes = {}
    for dpi in (72, 144):
        item = make_parent(data)
        RasterizeAgent(dpi=dpi)._rasterize_pdf(item, data)
        [[(w, h)]] = page_image_sizes(await image_bytes(item))
        sizes[dpi] = (w, h)
    assert abs(sizes[72][0] - 500) <= 2 and abs(sizes[72][1] - 500) <= 2
    assert abs(sizes[144][0] - 1000) <= 2 and abs(sizes[144][1] - 1000) <= 2


@pytest.mark.asyncio
async def test_rasterize_keeps_content_type_pdf():
    data = load(PDFS / "one_image.pdf")
    item = make_parent(data)
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[item])
    assert await item.content.content_type(ctx) == "application/pdf"


# --------------------------------------------------------------------------------------
# MarkdownDirOutput: image normalisation to PNG
# --------------------------------------------------------------------------------------


def bmp_2x2() -> bytes:
    """Uncompressed 24-bit 2x2 BMP, bottom-up, rows padded to 4 bytes."""
    rows = (b"\x00\x00\xff\x00\xff\x00" + b"\x00\x00") + (b"\xff\x00\x00\xff\xff\xff" + b"\x00\x00")
    header = b"BM" + struct.pack("<IHHI", 14 + 40 + len(rows), 0, 0, 14 + 40)
    info = struct.pack("<IiiHHIIiiII", 40, 2, 2, 1, 24, 0, len(rows), 2835, 2835, 0, 0)
    return header + info + rows


async def write_image(tmp_path: Path, raw: bytes, content_type: str) -> tuple[list[str], dict]:
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")))
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(raw, content_type=content_type, content_encoding="binary"), metadata={"page": 1, "caption": "c"})]
    result = await MarkdownDirOutput(str(tmp_path)).write(ctx)
    assert result.status == StepStatus.SUCCESS
    sidecar = json.loads((tmp_path / "doc_metadata.json").read_text())
    return sorted(p.name for p in (tmp_path / "out").iterdir()), sidecar


@pytest.mark.asyncio
async def test_markdown_output_converts_bmp_to_png(tmp_path):
    names, sidecar = await write_image(tmp_path, bmp_2x2(), "image/bmp")
    [name] = names
    assert name.endswith(".png")
    data = (tmp_path / "out" / name).read_bytes()
    assert png_size(data) == (2, 2)
    assert sidecar["images"][0]["file"].endswith(".png")


@pytest.mark.asyncio
async def test_markdown_output_png_stays_png_and_valid(tmp_path):
    png_1x1 = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415478da6360606060000000050001a5f645400000000049454e44ae426082"
    )
    names, _ = await write_image(tmp_path, png_1x1, "image/png")
    [name] = names
    assert name.endswith(".png")
    assert png_size((tmp_path / "out" / name).read_bytes()) == (1, 1)


@pytest.mark.asyncio
async def test_markdown_output_keeps_original_bytes_and_extension_when_undecodable(tmp_path):
    garbage = b"definitely not an image"
    names, _ = await write_image(tmp_path, garbage, "image/webp")
    [name] = names
    assert name.endswith(".webp")
    assert (tmp_path / "out" / name).read_bytes() == garbage
