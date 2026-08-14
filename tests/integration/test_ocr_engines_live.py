from pathlib import Path

import pytest

from omni_ingest.agent.document import OcrAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner

SCANNED_PDF = Path("tests/resources/human-body-and-its-movements.pdf")


@pytest.mark.parametrize("engine", ["llm", "openai", "azure_di"])
async def test_ocr_engine_extracts_real_page(engine, metadata_store):
    item = KnowledgeItem(raw_content=SCANNED_PDF.read_bytes(), metadata={"content_type": "application/pdf"})
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", raw_content=b""))

    result = (await PipelineRunner([OcrAgent(engine=engine, workers=2)]).run(ctx))[0]

    assert result.status == StepStatus.SUCCESS, f"Step failed: {result.error}"
    assert item.metadata["content_type"] == "text/markdown"
    assert len(item.raw_content) > 0

