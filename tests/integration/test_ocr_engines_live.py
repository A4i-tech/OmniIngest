"""Live OCR-engine tests against real services.

Credentials come from settings/.env (see .env.example): SARVAM_API_KEY,
AZURE_DI_OCR_ENDPOINT/KEY, AZURE_ML_OCR_SCORING_URI/KEY, and OPENAI_API_KEY for
the LLM engine.
"""

from pathlib import Path

import pymupdf

from omni_ingest.agent.document import OcrAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner
from omni_ingest.port.ocr import azure_di_ocr_builder, azure_ml_ocr_builder, sarvam_ocr_builder

SCANNED_PDF = Path("tests/resources/textbook-scan-middle-pages.pdf")


def middle_page_image() -> bytes:
    with pymupdf.open(SCANNED_PDF) as pdf:
        return pdf[0].get_pixmap(dpi=200, alpha=False).tobytes("png")


def dummy_ctx(metadata_store) -> IngestionContext:
    return IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", raw_content=b""))


async def test_sarvam_ocr_engine_extracts_real_page(metadata_store):
    async with sarvam_ocr_builder(dummy_ctx(metadata_store)) as engine:
        markdown = await engine.extract(middle_page_image())

    assert len(markdown) > 0


async def test_azure_di_ocr_engine_extracts_real_page(metadata_store):
    async with azure_di_ocr_builder(dummy_ctx(metadata_store)) as engine:
        markdown = await engine.extract(middle_page_image())

    assert "<table>" in markdown


async def test_azure_ml_ocr_engine_extracts_real_page(metadata_store):
    async with azure_ml_ocr_builder(dummy_ctx(metadata_store)) as engine:
        markdown = await engine.extract(middle_page_image())

    assert len(markdown) > 0


async def test_llm_ocr_engine_extracts_real_page(metadata_store):
    item = KnowledgeItem(raw_content=SCANNED_PDF.read_bytes(), metadata={"content_type": "application/pdf"})
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", raw_content=b""))

    result = (await PipelineRunner([OcrAgent(engine="llm", workers=2)]).run(ctx))[0]

    assert result.status == StepStatus.SUCCESS, f"Step failed: {result.error}"
    assert item.metadata["content_type"] == "text/markdown"
    assert len(item.raw_content) > 0
