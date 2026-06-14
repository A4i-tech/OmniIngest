import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omni_ingest.agent.chunking import PageChunkingAgent
from omni_ingest.agent.cleaning import (
    MarkdownCleaningAgent,
    TranscriptionPostProcessorAgent,
)
from omni_ingest.agent.governance import (
    DataLineageAgent,
    QualityVerifierAgent,
)
from omni_ingest.agent.structure import StructureAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import GraphStoreOutput, JsonFileOutput, VectorStoreOutput
from omni_ingest.core.pipeline import IngestionContext


def create_ctx(store, content):
    if isinstance(content, str):
        content = content.encode()
    return IngestionContext(tenant_id=uuid4(), resource=ResolvedResource(uri="", content=content), store=store, domain_profile="education")


@pytest.mark.asyncio
async def test_markdown_cleaning_agent(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = MarkdownCleaningAgent()

    ctx = create_ctx(vector_store, "## Raw")
    with patch.object(agent, "_agent", return_value=PydanticAgent("test", output_type=Literal["## Clean"])):
        result = await agent.run(ctx)
        assert result.status == StepStatus.SUCCESS
        assert ctx.resource.content == b"## Clean"


@pytest.mark.asyncio
async def test_transcription_post_processor_agent(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = TranscriptionPostProcessorAgent()

    ctx = create_ctx(vector_store, "hello")
    with patch.object(agent, "_agent", return_value=PydanticAgent("test", output_type=Literal["Hello"])):
        result = await agent.run(ctx)

        assert result.status == StepStatus.SUCCESS
        assert ctx.resource.content == b"Hello"


@pytest.mark.asyncio
async def test_quality_verifier_agent_success(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = QualityVerifierAgent(min_quality_score=0.7)

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"test content")]

    from omni_ingest.agent.governance import QualityReport

    mock_output = QualityReport(overall_score=0.9, issues=[])

    with patch.object(agent, "_agent", return_value=PydanticAgent("test", output_type=QualityReport)):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = mock_output
            result = await agent.run(ctx)

            assert result.status == StepStatus.SUCCESS
            assert "quality_report" in ctx.metadata
            assert ctx.metadata["quality_report"]["overall_score"] == 0.9


@pytest.mark.asyncio
async def test_quality_verifier_agent_failure(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = QualityVerifierAgent(min_quality_score=0.7)

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"bad content")]

    from omni_ingest.agent.governance import QualityReport

    mock_output = QualityReport(overall_score=0.4, issues=["too short"])

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = mock_output
            result = await agent.run(ctx)

            assert result.status == StepStatus.FAILURE
            assert result.error is not None and "Quality check failed" in result.error
            assert result.metadata["quality_score"] == 0.4


@pytest.mark.asyncio
async def test_structure_agent_item_set(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = StructureAgent(
        metadata_key="learning_outcomes",
        prompt="Extract learning outcomes. Content: {text}",
        json_schema={
            "type": "array",
            "items": {"type": "string"},
        },
    )

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"educational content")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = {"output": ["Understand testing"]}
            result = await agent.run(ctx)

            assert result.status == StepStatus.SUCCESS
            assert ctx.items[0].metadata["learning_outcomes"] == ["Understand testing"]


@pytest.mark.asyncio
async def test_structure_agent_update(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = StructureAgent(
        method="update",
        prompt="Enrich. Content: {text}",
        json_schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary", "tags"],
        },
    )

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"content")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = {"summary": "s", "tags": ["t"]}
            result = await agent.run(ctx)

            assert result.status == StepStatus.SUCCESS
            assert ctx.items[0].metadata["summary"] == "s"
            assert ctx.items[0].metadata["tags"] == ["t"]


@pytest.mark.asyncio
async def test_data_lineage_agent(vector_store):
    agent = DataLineageAgent()
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"test")]
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "run_id" in result.metadata


@pytest.mark.asyncio
async def test_page_chunking_agent(vector_store):
    agent = PageChunkingAgent()
    with patch("omni_ingest.agent.chunking.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader.return_value.pages = [mock_page]
        ctx = create_ctx(vector_store, "")
        ctx.metadata["file_path"] = "fake.pdf"
        result = await agent.run(ctx)
        assert result.status == StepStatus.SUCCESS
        assert len(ctx.items) == 1


@pytest.mark.asyncio
async def test_json_file_output(vector_store, tmp_path):
    output = JsonFileOutput(path=tmp_path / "items.json")
    ctx = create_ctx(vector_store, "resource")
    ctx.metadata["source"] = "unit"
    ctx.items = [KnowledgeItem(content=b"test", metadata={"tag": "x"})]

    result = await output.write(ctx)
    data = json.loads(output.path.read_text())

    assert result.status == StepStatus.SUCCESS
    assert result.output_paths == {"json": str(output.path)}
    assert data["resource"]["content"] == "resource"
    assert data["metadata"] == {"source": "unit"}
    assert data["items"][0]["content"] == "test"
    assert data["items"][0]["metadata"] == {"tag": "x"}


@pytest.mark.asyncio
async def test_vector_store_output(vector_store):
    output = VectorStoreOutput(store=vector_store)
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"test", metadata={"embedding": [0.1] * 1536})]
    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS
    assert await vector_store.count() == 1


@pytest.mark.asyncio
async def test_graph_store_output(vector_store):
    mock_store = AsyncMock()
    output = GraphStoreOutput(store=mock_store)
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"test", metadata={"tags": ["tag1"]})]
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}], "edges": []}
    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS
    mock_store.upsert_nodes.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_store_output_with_edges(vector_store):
    mock_store = AsyncMock()
    output = GraphStoreOutput(store=mock_store)
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"test", metadata={"tags": ["tag1"]})]
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}], "edges": [{"source": "n1", "target": "n2"}]}

    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS
    mock_store.upsert_nodes.assert_awaited_once()
    mock_store.upsert_edges.assert_awaited_once()


@pytest.mark.asyncio
async def test_timed_chunking_agent(vector_store):
    from omni_ingest.agent.chunking import TimedChunkingAgent

    agent = TimedChunkingAgent(segment_duration="00:00:00.100000")

    ctx = create_ctx(vector_store, "test")
    ctx.metadata["word_timestamps"] = [{"Word": "test", "Offset": 0, "Duration": 2000000}]
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS


@pytest.mark.asyncio
async def test_ingestion_logger_agent(vector_store):
    from omni_ingest.agent.governance import IngestionLoggerAgent

    agent = IngestionLoggerAgent()
    ctx = create_ctx(vector_store, "")
    ctx.run_id = uuid4()
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "timestamp" in result.metadata

