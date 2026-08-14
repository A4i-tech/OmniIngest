from pathlib import Path

import pymupdf
import pytest

from omni_ingest.agent.document import OcrAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner
from omni_ingest.port.ocr import azure_di_ocr_builder

SCANNED_PDF = Path("tests/resources/textbook-scan-middle-pages.pdf")


def middle_page_image() -> ByteContent:
    with pymupdf.open(SCANNED_PDF) as pdf:
        png = pdf[0].get_pixmap(dpi=200, alpha=False).tobytes("png")
    return ByteContent(raw_content=png, metadata={"content_type": "image/png"})


def dummy_ctx(metadata_store) -> IngestionContext:
    return IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", raw_content=b""))


@pytest.mark.parametrize("engine", ["llm", "openai"])
async def test_ocr_engine_extracts_real_page(engine, metadata_store):
    item = KnowledgeItem(raw_content=SCANNED_PDF.read_bytes(), metadata={"content_type": "application/pdf"})
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", raw_content=b""))

    result = (await PipelineRunner([OcrAgent(engine=engine, workers=2)]).run(ctx))[0]

    assert result.status == StepStatus.SUCCESS, f"Step failed: {result.error}"
    assert item.metadata["content_type"] == "text/markdown"
    assert len(item.raw_content) > 0


async def test_azure_di_ocr_engine_extracts_real_page(metadata_store):
    engine = azure_di_ocr_builder(dummy_ctx(metadata_store), src_lang=None, dst_lang=None)
    result = await engine(middle_page_image())

    assert "<table>" in result.raw_content.decode("utf-8")
