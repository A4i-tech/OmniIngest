import io

import pypdf
import pytest
from PIL import Image
from pypdf.generic import DecodedStreamObject, NameObject

from omni_ingest.agent.document import ImageExtractionAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource
from omni_ingest.core.pipeline import IngestionContext


@pytest.fixture
def original(resources_dir) -> bytes:
    return (resources_dir / "pdf" / "single_image.pdf").read_bytes()



def page_of(content: bytes) -> pypdf.PageObject:
    return pypdf.PdfReader(io.BytesIO(content)).pages[0]


@pytest.mark.asyncio
async def test_image_extraction_preserves_parent_by_default(original):
    parent = KnowledgeItem(content=ByteContent.from_bytes(original))
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[parent])
    extracted = ImageExtractionAgent(include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    assert await parent.content.read(ctx) == original


@pytest.mark.asyncio
async def test_image_extraction_redacts_parent_with_extracted_item_id(original):
    parent = KnowledgeItem(content=ByteContent.from_bytes(original))
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[parent])
    extracted = ImageExtractionAgent(mode="redact", include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    content = await parent.content.read(ctx)
    assert content != original
    page = page_of(content)
    assert f'<image id="{extracted[0].id}"/>' in page.extract_text()
    assert len(page.images) == 0


@pytest.mark.asyncio
async def test_image_extraction_captions_parent_with_extracted_item_id(original):
    parent = KnowledgeItem(content=ByteContent.from_bytes(original))
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[parent])
    extracted = ImageExtractionAgent(mode="caption", include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    content = await parent.content.read(ctx)
    assert content != original
    page = page_of(content)
    assert f'<image id="{extracted[0].id}"/>' in page.extract_text()
    assert len(page.images) == 1


def pdf_with_rects(*rects: tuple[float, float, float, float]) -> bytes:
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(400, 400)
    stream = DecodedStreamObject()
    stream.set_data("".join(f"0 0 1 rg {x0} {400 - y1} {x1 - x0} {y1 - y0} re f\n" for x0, y0, x1, y1 in rects).encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with io.BytesIO() as out:
        writer.write(out)
        return out.getvalue()


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
    assert max(Image.open(io.BytesIO(items[0].content.data)).size) <= 50
