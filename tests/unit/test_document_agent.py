import pymupdf

from omni_ingest.agent.document import ImageExtractionAgent
from omni_ingest.core.model import KnowledgeItem


def pdf_with_image() -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16), False)
        pixmap.clear_with(0x336699)
        page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=pixmap.tobytes("png"))
        return bytes(pdf.tobytes())


def test_image_extraction_preserves_parent_by_default():
    original = pdf_with_image()
    parent = KnowledgeItem(raw_content=original)
    extracted = ImageExtractionAgent(include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    assert parent.raw_content == original


def test_image_extraction_redacts_parent_with_extracted_item_id():
    original = pdf_with_image()
    parent = KnowledgeItem(raw_content=original)
    extracted = ImageExtractionAgent(mode="redact", include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    assert parent.raw_content != original
    with pymupdf.open(stream=parent.raw_content, filetype="pdf") as redacted:
        assert f'<image id="{extracted[0].id}"/>' in redacted[0].get_text()
        assert redacted[0].get_image_info() == []


def test_image_extraction_captions_parent_with_extracted_item_id():
    original = pdf_with_image()
    parent = KnowledgeItem(raw_content=original)
    extracted = ImageExtractionAgent(mode="caption", include_vector=False, padding=0, dpi=72)._extract_pdf(parent, original)

    assert len(extracted) == 1
    assert parent.raw_content != original
    with pymupdf.open(stream=parent.raw_content, filetype="pdf") as captioned:
        assert f'<image id="{extracted[0].id}"/>' in captioned[0].get_text()
        assert len(captioned[0].get_image_info()) == 1
