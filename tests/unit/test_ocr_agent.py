import asyncio

import pymupdf
import pytest

from omni_ingest.agent.document import OcrAgent
from omni_ingest.core.config import LlamaCppOcrConfig
from omni_ingest.core.model import KnowledgeItem, ResolvedResource
from omni_ingest.core.pipeline import IngestionContext


def text_native_pdf() -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        page.insert_text((20, 20), "Hello structured world")
        return bytes(pdf.tobytes())


def scanned_pdf() -> bytes:
    with pymupdf.open() as pdf:
        pdf.new_page(width=400, height=200)
        return bytes(pdf.tobytes())


class FakeOcrEngine:
    def __init__(self, markdown: str = "# OCR page"):
        self.markdown = markdown
        self.calls = 0

    async def extract(self, image: bytes) -> str:
        self.calls += 1
        return self.markdown


def test_render_pages_skips_text_native_page():
    text, needs_ocr, image = OcrAgent()._render_pages(text_native_pdf())[0]

    assert "Hello structured world" in text
    assert needs_ocr is False
    assert image is None


def test_render_pages_flags_scanned_page():
    text, needs_ocr, image = OcrAgent()._render_pages(scanned_pdf())[0]

    assert text == ""
    assert needs_ocr is True
    assert isinstance(image, bytes) and image


@pytest.mark.asyncio
async def test_extract_uses_ocr_engine_only_for_scanned_pages():
    with pymupdf.open() as source, pymupdf.open() as pdf:
        native = source.new_page(width=400, height=200)
        native.insert_text((20, 20), "Native text")
        pdf.insert_pdf(source)
        pdf.new_page(width=400, height=200)
        content = bytes(pdf.tobytes())

    item = KnowledgeItem(raw_content=content, metadata={"content_type": "application/pdf"})
    engine = FakeOcrEngine("# OCR page")

    pages, chars, ocr_pages = await OcrAgent()._extract(item, content, engine, asyncio.Semaphore(2))

    assert pages == 2
    assert ocr_pages == 1
    assert engine.calls == 1
    assert "Native text" in item.raw_content.decode("utf-8")
    assert "# OCR page" in item.raw_content.decode("utf-8")
    assert item.content_encoding == "utf-8"
    assert item.metadata["content_type"] == "text/markdown"
    assert chars == len(item.raw_content.decode("utf-8"))


@pytest.mark.asyncio
async def test_run_fails_without_configured_engine(monkeypatch):
    import omni_ingest.agent.document as document_module

    monkeypatch.setattr(document_module.settings, "ocr_engine", None)
    item = KnowledgeItem(raw_content=scanned_pdf(), metadata={"content_type": "application/pdf"})
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""), items=[item])

    with pytest.raises(ValueError, match="No OCR engine configured"):
        await OcrAgent().run(ctx)


@pytest.mark.asyncio
async def test_run_uses_engine_override_over_settings(monkeypatch):
    import omni_ingest.agent.document as document_module

    monkeypatch.setattr(document_module.settings, "ocr_engine", None)
    item = KnowledgeItem(raw_content=text_native_pdf(), metadata={"content_type": "application/pdf"})
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""), items=[item])

    result = await OcrAgent(engine=LlamaCppOcrConfig()).run(ctx)

    assert result.metadata["ocr_used"] is False
    assert "Hello structured world" in item.raw_content.decode("utf-8")
