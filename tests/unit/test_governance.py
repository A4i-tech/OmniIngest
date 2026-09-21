import pytest

from omni_ingest.agent.governance import DataLineageAgent, IngestionLoggerAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext


@pytest.mark.asyncio
async def test_data_lineage_agent():
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[KnowledgeItem(content=ByteContent.from_bytes(b"test"))])

    result = await DataLineageAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert "run_id" in result.metadata


@pytest.mark.asyncio
async def test_ingestion_logger_agent():
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")))

    result = await IngestionLoggerAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert "timestamp" in result.metadata
