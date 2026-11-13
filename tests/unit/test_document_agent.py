import pymupdf
import pytest

from omni_ingest.agent.document import ImageExtractionAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource
from omni_ingest.core.pipeline import IngestionContext


def pdf_with_image() -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16), False)
        pixmap.clear_with(0x336699)
        page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=pixmap.tobytes("png"))
        return bytes(pdf.tobytes())


@pytest.mark.asyncio
async def test_image_extraction_preserves_parent_by_default():
    original = pdf_with_image()
    parent = KnowledgeItem(content=ByteContent.from_bytes(original))
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[parent])
    extracted = ImageExtractionAgent(include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    assert await parent.content.read(ctx) == original


@pytest.mark.asyncio
async def test_image_extraction_redacts_parent_with_extracted_item_id():
    original = pdf_with_image()
    parent = KnowledgeItem(content=ByteContent.from_bytes(original))
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[parent])
    extracted = ImageExtractionAgent(mode="redact", include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    content = await parent.content.read(ctx)
    assert content != original
    with pymupdf.open(stream=content, filetype="pdf") as redacted:
        assert f'<image id="{extracted[0].id}"/>' in redacted[0].get_text()
        assert redacted[0].get_image_info() == []


@pytest.mark.asyncio
async def test_image_extraction_captions_parent_with_extracted_item_id():
    original = pdf_with_image()
    parent = KnowledgeItem(content=ByteContent.from_bytes(original))
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[parent])
    extracted = ImageExtractionAgent(mode="caption", include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    content = await parent.content.read(ctx)
    assert content != original
    with pymupdf.open(stream=content, filetype="pdf") as captioned:
        assert f'<image id="{extracted[0].id}"/>' in captioned[0].get_text()
        assert len(captioned[0].get_image_info()) == 1


def pdf_with_rects(*rects: tuple[float, float, float, float]) -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=400)
        for rect in rects:
            page.draw_rect(pymupdf.Rect(rect), fill=(0, 0, 1))
        return bytes(pdf.tobytes())


def extract(pdf: bytes, **config):
    parent = KnowledgeItem(content=ByteContent.from_bytes(pdf))
    return ImageExtractionAgent(padding=0, dpi=72, **config)._extract_pdf(parent, pdf)


def test_image_extraction_merges_regions_within_gap():
    pdf = pdf_with_rects((50, 100, 150, 200), (155, 100, 250, 200), (300, 300, 350, 350))
    assert len(extract(pdf, merge_gap=0)) == 3
    assert len(extract(pdf, merge_gap=6)) == 2


def test_image_extraction_skips_thin_full_width_band_without_marker():
    pdf = pdf_with_rects((10, 10, 390, 40), (100, 150, 200, 250))
    items = extract(pdf, mode="redact", min_width=1)
    assert len(items) == 1 and items[0].metadata["bbox"][1] == 150


def test_image_extraction_min_size_and_max_px():
    pdf = pdf_with_rects((50, 50, 60, 60), (100, 100, 300, 300))
    items = extract(pdf, min_width=20, min_height=20, max_image_px=50)
    assert len(items) == 1
    with pymupdf.open(stream=items[0].content.data, filetype="png") as image:
        assert max(image[0].rect.width, image[0].rect.height) <= 50
