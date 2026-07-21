"""Live OCR-engine tests against real services.

Credentials come from settings/.env (see .env.example): SARVAM_API_KEY,
AZURE_DI_OCR_ENDPOINT/KEY, AZURE_ML_OCR_SCORING_URI/KEY, and OPENAI_API_KEY for
the LLM engine.
"""

from pathlib import Path

import pymupdf

from omni_ingest.agent.document import OcrAgent
from omni_ingest.core.config import AzureDocIntelOcrConfig, AzureMlOcrConfig, LLMOcrConfig, SarvamOcrConfig, settings
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner

SCANNED_PDF = Path("tests/resources/textbook-scan-middle-pages.pdf")


def middle_page_image() -> bytes:
    with pymupdf.open(SCANNED_PDF) as pdf:
        return pdf[0].get_pixmap(dpi=200, alpha=False).tobytes("png")


async def test_sarvam_ocr_engine_extracts_real_page():
    async with SarvamOcrConfig(api_key=settings.sarvam_api_key, language="hi-IN").build() as engine:
        markdown = await engine.extract(middle_page_image())

    assert len(markdown) > 0


async def test_azure_di_ocr_engine_extracts_real_page():
    config = AzureDocIntelOcrConfig(endpoint=settings.azure_di_ocr_endpoint, api_key=settings.azure_di_ocr_key)
    async with config.build() as engine:
        markdown = await engine.extract(middle_page_image())

    assert "<table>" in markdown


async def test_azure_ml_ocr_engine_extracts_real_page():
    config = AzureMlOcrConfig(scoring_uri=settings.azure_ml_ocr_scoring_uri, api_key=settings.azure_ml_ocr_key)
    async with config.build() as engine:
        markdown = await engine.extract(middle_page_image())

    assert len(markdown) > 0


async def test_llm_ocr_engine_extracts_real_page(metadata_store):
    item = KnowledgeItem(raw_content=SCANNED_PDF.read_bytes(), metadata={"content_type": "application/pdf"})
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", raw_content=b""))

    result = (await PipelineRunner([OcrAgent(engine=LLMOcrConfig(), workers=2)]).run(ctx))[0]

    assert result.status == StepStatus.SUCCESS, f"Step failed: {result.error}"
    assert item.metadata["content_type"] == "text/markdown"
    assert len(item.raw_content) > 0
