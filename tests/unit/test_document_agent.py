import io

import pypdf
import pytest

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
