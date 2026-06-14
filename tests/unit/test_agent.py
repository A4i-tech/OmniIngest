import json
from typing import Literal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omni_ingest.agent.chunking import PageChunkingAgent
from omni_ingest.agent.cleaning import (
    MarkdownCleaningAgent,
    TranscriptionPostProcessorAgent,
)
from omni_ingest.agent.governance import (
    DataLineageAgent,
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
async def test_structure_agent_item_set(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = StructureAgent(
        target=".metadata.learning_outcomes = $output",
        prompt="Extract learning outcomes.",
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
        target=".metadata += $output",
        prompt="Enrich.",
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
            mock_run.return_value.output = {"output": {"summary": "s", "tags": ["t"]}}
            result = await agent.run(ctx)

            assert result.status == StepStatus.SUCCESS
            assert ctx.items[0].metadata["summary"] == "s"
            assert ctx.items[0].metadata["tags"] == ["t"]


@pytest.mark.asyncio
async def test_structure_agent_jq_foreach_input_and_append_target(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = StructureAgent(
        foreach=".items[:1][]",
        input=".text[:4]",
        target=".metadata.keywords = ((.metadata.keywords // []) + $output)",
        prompt="Keywords.",
        json_schema={"type": "array", "items": {"type": "string"}},
    )
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"alpha", metadata={"keywords": ["old"]}), KnowledgeItem(content=b"bravo")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = {"output": ["new"]}
            result = await agent.run(ctx)

    mock_run.assert_awaited_once()
    assert mock_run.await_args.args[0] == ["Keywords.", "alph"]
    assert result.metadata["items_processed"] == 1
    assert ctx.items[0].metadata["keywords"] == ["old", "new"]
    assert ctx.items[1].metadata == {}


@pytest.mark.asyncio
async def test_structure_agent_assertions_fail_step(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = StructureAgent(
        foreach=None,
        target=".metadata.quality_report = $output",
        prompt="Quality.",
        assertions=[{"expr": "$output.overall_score >= 0.7", "message": "Quality check failed"}],
        json_schema={
            "type": "object",
            "properties": {"overall_score": {"type": "number"}, "issues": {"type": "array", "items": {"type": "string"}}},
            "required": ["overall_score", "issues"],
        },
    )
    ctx = create_ctx(vector_store, "bad content")
    ctx.items = [KnowledgeItem(content=b"bad content")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = {"output": {"overall_score": 0.4, "issues": ["too short"]}}
            result = await agent.run(ctx)

    assert result.status == StepStatus.FAILURE
    assert result.error == "Quality check failed"
    assert ctx.metadata["quality_report"]["overall_score"] == 0.4


@pytest.mark.asyncio
async def test_structure_agent_context_target(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = StructureAgent(
        foreach=None,
        input='[.items[:2][] | .text[:4]] | join("|")',
        target=".metadata.faqs = $output",
        prompt="Generate FAQs.",
        json_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "a": {"type": "string"}},
                "required": ["q", "a"],
            },
        },
    )

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"alpha"), KnowledgeItem(content=b"bravo"), KnowledgeItem(content=b"charlie")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = {"output": [{"q": "What?", "a": "This."}]}
            result = await agent.run(ctx)

    mock_run.assert_awaited_once()
    assert mock_run.await_args.args[0] == ["Generate FAQs.", "alph|brav"]
    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["faqs"] == [{"q": "What?", "a": "This."}]


@pytest.mark.asyncio
async def test_structure_agent_binary_input_uses_selected_items():
    agent = StructureAgent(
        foreach=None,
        input='[.items[] | select(.metadata.page == 2) | .id]',
        target=".metadata.metadata = $output",
        prompt="Extract metadata.",
        json_schema={"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]},
    )

    ctx = create_ctx(None, "")
    ctx.items = [
        KnowledgeItem(content=b"%PDF-1.4\npage 1", metadata={"page": 1}),
        KnowledgeItem(content=b"%PDF-1.4\npage 2", metadata={"page": 2}),
    ]
    mock_agent = AsyncMock()
    mock_agent.run.return_value.output = {"output": {"subject": "Mathematics"}}

    with patch.object(agent, "_agent", return_value=mock_agent):
        result = await agent.run(ctx)

    content = mock_agent.run.await_args.args[0][1]
    assert mock_agent.run.await_args.args[0][0] == "Extract metadata."
    assert content.data == b"%PDF-1.4\npage 2"
    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["metadata"] == {"subject": "Mathematics"}


@pytest.mark.asyncio
async def test_structure_agent_serializes_pydantic_output_for_jq_args():
    class Output(BaseModel):
        output: dict[str, str]

    agent = StructureAgent(
        foreach=None,
        target=".metadata.metadata = $output",
        prompt="Extract metadata.",
        json_schema={"type": "object", "properties": {"board": {"type": "string", "enum": ["CBSE"]}}, "required": ["board"]},
    )
    ctx = create_ctx(None, "content")
    ctx.items = [KnowledgeItem(content=b"content")]
    mock_agent = AsyncMock()
    mock_agent.run.return_value.output = Output(output={"board": "CBSE"})

    with patch.object(agent, "_agent", return_value=mock_agent):
        result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["metadata"] == {"board": "CBSE"}


@pytest.mark.asyncio
async def test_data_lineage_agent(vector_store):
    agent = DataLineageAgent()
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(content=b"test")]
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "run_id" in result.metadata


@pytest.mark.asyncio
async def test_page_chunking_agent(vector_store, resources_dir):
    with (resources_dir / "when-tomorrow-was-yesterday.pdf").open(mode="rb") as f:
        ctx = create_ctx(vector_store, f.read())
    result = await PageChunkingAgent().run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert len(ctx.items) == 12


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
