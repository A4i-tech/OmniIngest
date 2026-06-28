import asyncio
import json
from typing import Literal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import fsspec
import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai.capabilities import Hooks

from omni_ingest.agent.chunking import PageChunkingAgent
from omni_ingest.agent.enrichment import ExtractAgent, PromptAgent, TransformAgent
from omni_ingest.agent.governance import DataLineageAgent
from omni_ingest.agent.indexing import EmbeddingAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import GraphStoreOutput, JsonFileOutput, StdoutOutput, VectorStoreOutput, parse
from omni_ingest.core.pipeline import IngestionContext


def create_ctx(store, content):
    if isinstance(content, str):
        content = content.encode()
    return IngestionContext(tenant_id=uuid4(), resource=ResolvedResource(uri="", raw_content=content), store=store, domain_profile="education")


@pytest.mark.asyncio
async def test_prompt_agent(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = PromptAgent(prompt="Clean this Markdown. Output only cleaned Markdown.", input=".metadata.text", path=".metadata.cleaned")

    ctx = create_ctx(vector_store, "## Raw")
    ctx.metadata["text"] = "## Raw"
    with patch.object(agent, "_agent", return_value=PydanticAgent("test", output_type=Literal["## Clean"])):
        result = await agent.run(ctx)
        assert result.status == StepStatus.SUCCESS
        assert ctx.metadata["cleaned"] == "## Clean"


@pytest.mark.asyncio
async def test_embedding_agent_uses_jq_selection_input_and_path(vector_store):
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"keep", metadata={"kind": "paragraph"}), KnowledgeItem(raw_content=b"skip", metadata={"kind": "image"})]
    agent = EmbeddingAgent(foreach='.items[] | select(.metadata.kind == "paragraph")', input=".text", path=".metadata.vector")

    with patch("omni_ingest.agent.indexing.Embedder") as embedder:
        embedder.return_value.embed_documents = AsyncMock()
        embedder.return_value.embed_documents.return_value.embeddings = [[1.0, 2.0]]
        result = await agent.run(ctx)

    embedder.return_value.embed_documents.assert_awaited_once_with(["keep"])
    assert result.metadata == {"embedded_count": 1, "path": ".metadata.vector"}
    assert ctx.items[0].metadata["vector"] == [1.0, 2.0]
    assert "vector" not in ctx.items[1].metadata


@pytest.mark.asyncio
async def test_extract_agent_item_set(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = ExtractAgent(
        path=".metadata.learning_outcomes",
        prompt="Extract learning outcomes.",
        schema={
            "type": "array",
            "items": {"type": "string"},
        },
    )

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"educational content")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = agent._output_type(None)(output=["Understand testing"])
            result = await agent.run(ctx)

            assert result.status == StepStatus.SUCCESS
            assert ctx.items[0].metadata["learning_outcomes"] == ["Understand testing"]


@pytest.mark.asyncio
async def test_extract_agent_updates_item_metadata(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = ExtractAgent(
        path=".metadata",
        prompt="Enrich.",
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary", "tags"],
        },
    )

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"content", metadata={"learning_outcomes": ["old"]})]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = agent._output_type(None)(output={"summary": "s", "tags": ["t"]})
            result = await agent.run(ctx)

            assert result.status == StepStatus.SUCCESS
            assert ctx.items[0].metadata["learning_outcomes"] == ["old"]
            assert ctx.items[0].metadata["summary"] == "s"
            assert ctx.items[0].metadata["tags"] == ["t"]


@pytest.mark.asyncio
async def test_concurrent_extract_agents_merge_root_metadata():
    ctx = create_ctx(None, "")
    ctx.items = [KnowledgeItem(raw_content=b"content")]
    agents = [
        ExtractAgent(foreach=".", path=f".metadata.{key}", prompt=key, schema={"type": "string"})
        for key in ("first", "second")
    ]

    async def extract(agent, _ctx, item_doc, _item_by_id, _sem, _advance):
        await asyncio.sleep(0.01 if agent.path.endswith("first") else 0)
        return item_doc, agent.prompt

    with patch.object(ExtractAgent, "_extract", extract):
        await asyncio.gather(*(agent.run(ctx) for agent in agents))

    assert ctx.metadata == {"first": "first", "second": "second"}


@pytest.mark.asyncio
async def test_concurrent_extract_agents_merge_item_metadata():
    ctx = create_ctx(None, "")
    ctx.items = [KnowledgeItem(raw_content=b"content", metadata={"existing": "keep"})]
    agents = [
        ExtractAgent(path=f".metadata.generated.{key}", prompt=key, schema={"type": "string"})
        for key in ("first", "second")
    ]

    async def extract(agent, _ctx, item_doc, _item_by_id, _sem, _advance):
        await asyncio.sleep(0.01 if agent.path.endswith("first") else 0)
        return item_doc, agent.prompt

    with patch.object(ExtractAgent, "_extract", extract):
        await asyncio.gather(*(agent.run(ctx) for agent in agents))

    assert ctx.items[0].metadata == {
        "existing": "keep",
        "generated": {"first": "first", "second": "second"},
    }


@pytest.mark.asyncio
async def test_extract_agent_jq_foreach_input(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = ExtractAgent(
        foreach=".items[:1][]",
        input=".text[:4]",
        path=".metadata.keywords",
        prompt="Keywords.",
        schema={"type": "array", "items": {"type": "string"}},
    )
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"alpha"), KnowledgeItem(raw_content=b"bravo")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = agent._output_type(None)(output=["new"])
            result = await agent.run(ctx)

    mock_run.assert_awaited_once()
    assert mock_run.await_args.args[0] == ["Keywords.", "alph"]
    assert result.metadata["items_processed"] == 1
    assert ctx.items[0].metadata["keywords"] == ["new"]
    assert ctx.items[1].metadata == {}


@pytest.mark.asyncio
async def test_extract_agent_assertions_register_model_retry_validator(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = ExtractAgent(
        foreach=".",
        path=".metadata.quality_report",
        prompt="Quality.",
        assertions=[{"expr": ".overall_score >= 0.7", "feedback": "Quality is too low"}],
        schema={
            "type": "object",
            "properties": {"overall_score": {"type": "number"}, "issues": {"type": "array", "items": {"type": "string"}}},
            "required": ["overall_score", "issues"],
        },
    )
    ctx = create_ctx(vector_store, "bad content")
    ctx.items = [KnowledgeItem(raw_content=b"bad content")]
    pydantic_agent = PydanticAgent("test")
    output_type = agent._output_type(None)

    with patch.object(agent, "_agent", return_value=pydantic_agent) as mock_agent:
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = output_type(output={"overall_score": 0.4, "issues": ["too short"]})
            result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["quality_report"]["overall_score"] == 0.4
    assert mock_agent.call_args.kwargs["output_type"].outputs.model_json_schema() == output_type.model_json_schema()
    assert isinstance(mock_agent.call_args.kwargs["capabilities"][0], Hooks)


def test_extract_agent_runtime_schema():
    agent = ExtractAgent(path=".metadata.answers", prompt="Answer.", schema=".metadata.schema")
    output_type = agent._output_type({"metadata": {"schema": {
        "type": "object",
        "properties": {"a1": {"type": "string"}, "a2": {"type": "string"}},
        "required": ["a1", "a2"],
    }}})

    assert output_type(output={"a1": "a", "a2": "b"}).model_dump(mode="json")["output"] == {"a1": "a", "a2": "b"}
    with pytest.raises(ValueError):
        output_type(output={"a1": "a"})


@pytest.mark.asyncio
async def test_extract_agent_document_scope(vector_store):
    from pydantic_ai import Agent as PydanticAgent
    agent = ExtractAgent(
        foreach=".",
        input='[.items[:2][] | .text[:4]] | join("|")',
        path=".metadata.faqs",
        prompt="Generate FAQs.",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "a": {"type": "string"}},
                "required": ["q", "a"],
            },
        },
    )

    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"alpha"), KnowledgeItem(raw_content=b"bravo"), KnowledgeItem(raw_content=b"charlie")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = agent._output_type(None)(output=[{"q": "What?", "a": "This."}])
            result = await agent.run(ctx)

    mock_run.assert_awaited_once()
    assert mock_run.await_args.args[0] == ["Generate FAQs.", "alph|brav"]
    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["faqs"] == [{"q": "What?", "a": "This."}]


@pytest.mark.asyncio
async def test_extract_agent_binary_input_uses_selected_items():
    from pydantic_ai import Agent as PydanticAgent
    agent = ExtractAgent(
        foreach=".",
        input='[.items[] | select(.metadata.page == 2) | .id]',
        path=".metadata.metadata",
        prompt="Extract metadata.",
        schema={"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]},
    )

    ctx = create_ctx(None, "")
    ctx.items = [
        KnowledgeItem(raw_content=b"%PDF-1.4\npage 1", metadata={"page": 1}),
        KnowledgeItem(raw_content=b"%PDF-1.4\npage 2", metadata={"page": 2}),
    ]
    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = agent._output_type(None)(output={"subject": "Mathematics"})
            result = await agent.run(ctx)

    content = mock_run.await_args.args[0][1]
    assert mock_run.await_args.args[0][0] == "Extract metadata."
    assert content.data == b"%PDF-1.4\npage 2"
    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["metadata"] == {"subject": "Mathematics"}


@pytest.mark.asyncio
async def test_extract_agent_serializes_pydantic_output():
    from pydantic_ai import Agent as PydanticAgent

    class Output(BaseModel):
        output: dict[str, str]

    agent = ExtractAgent(
        foreach=".",
        path=".metadata.metadata",
        prompt="Extract metadata.",
        schema={"type": "object", "properties": {"board": {"type": "string", "enum": ["CBSE"]}}, "required": ["board"]},
    )
    ctx = create_ctx(None, "content")
    ctx.items = [KnowledgeItem(raw_content=b"content")]

    with patch.object(agent, "_agent", return_value=PydanticAgent("test")):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = Output(output={"board": "CBSE"})
            result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["metadata"] == {"board": "CBSE"}


@pytest.mark.asyncio
async def test_extract_agent_buffered_mode_registers_item_reader():
    from pydantic_ai import Agent as PydanticAgent
    agent = ExtractAgent(
        foreach=".",
        mode="buffered",
        input='[.items[] | .id]',
        path=".metadata.toc",
        prompt="Find TOC.",
        schema={"type": "object", "properties": {"found": {"type": "boolean"}}, "required": ["found"]},
    )
    ctx = create_ctx(None, "")
    ctx.items = [KnowledgeItem(raw_content=b"content")]
    pydantic_agent = PydanticAgent("test")

    with patch.object(agent, "_agent", return_value=pydantic_agent):
        with patch("pydantic_ai.Agent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value.output = agent._output_type(None)(output={"found": True})
            result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert mock_run.await_args.args[0] == "Find TOC."
    assert pydantic_agent._function_toolset.tools


@pytest.mark.asyncio
async def test_transform_agent_writes_jq_output_to_metadata():
    agent = TransformAgent(value=".metadata.items | map(.name)", path=".metadata.names")
    ctx = create_ctx(None, "")
    ctx.metadata["items"] = [{"name": "a"}, {"name": "b"}]

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["names"] == ["a", "b"]


@pytest.mark.asyncio
async def test_transform_agent_maps_nested_metadata():
    agent = TransformAgent(path=".metadata.records", value=r"""
        .metadata.book as $book
        | .metadata.chapters
        | map({
            id: "\($book.board)-\(.no)",
            title: .name,
            topics: (.topics | map(.numbering + ". " + .title))
            })
    """)
    ctx = create_ctx(None, "")
    ctx.metadata = {
        "book": {"board": "BSE-TG"},
        "chapters": [{"no": 1, "name": "Our Environment", "topics": [{"numbering": "1.1", "title": "Introduction"}]}],
    }

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["records"] == [{"id": "BSE-TG-1", "title": "Our Environment", "topics": ["1.1. Introduction"]}]


def test_transform_agent_rejects_invalid_jq():
    with pytest.raises(ValidationError):
        TransformAgent(value=".metadata |", path=".metadata.records")


@pytest.mark.asyncio
async def test_data_lineage_agent(vector_store):
    agent = DataLineageAgent()
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"test")]
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "run_id" in result.metadata


@pytest.mark.asyncio
async def test_page_chunking_agent(vector_store, resources_dir):
    with (resources_dir / "when-tomorrow-was-yesterday.pdf").open(mode="rb") as f:
        ctx = create_ctx(vector_store, f.read())
    ctx.items = [KnowledgeItem(raw_content=ctx.resource.raw_content)]
    result = await PageChunkingAgent().run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert len(ctx.items) == 12


@pytest.mark.asyncio
async def test_json_file_output(vector_store, tmp_path):
    path = tmp_path / "items.json"
    output = JsonFileOutput(path=str(path))
    ctx = create_ctx(vector_store, "resource")
    ctx.metadata["source"] = "unit"
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"tag": "x"}), KnowledgeItem(raw_content=b"\xff", content_encoding="binary")]

    result = await output.write(ctx)
    data = json.loads(path.read_text())

    assert result.status == StepStatus.SUCCESS
    assert result.output_paths == {"json": str(path)}
    assert data["resource"]["content"] == "resource"
    assert data["metadata"] == {"source": "unit"}
    assert data["items"][0]["content"] == "test"
    assert data["items"][0]["metadata"] == {"tag": "x"}
    assert data["items"][1]["content"] == "/w=="
    assert data["items"][1]["content_encoding"] == "base64"


@pytest.mark.asyncio
async def test_parse_output_and_stdout(vector_store, tmp_path, capsys):
    assert isinstance(parse("-"), StdoutOutput)
    assert isinstance(parse(str(tmp_path / "items.json")), JsonFileOutput)
    assert isinstance(parse("vector:default"), VectorStoreOutput)
    assert isinstance(parse(str(tmp_path / "items.yml"), "json"), JsonFileOutput)
    output = parse("-")
    ctx = create_ctx(vector_store, "resource")

    result = await output.write(ctx)
    data = json.loads(capsys.readouterr().out)

    assert result.status == StepStatus.SUCCESS
    assert data["resource"]["content"] == "resource"


@pytest.mark.asyncio
async def test_json_file_output_supports_fsspec_urls(vector_store):
    output = parse("memory://omni-ingest/items.txt", "json")
    ctx = create_ctx(vector_store, "resource")

    await output.write(ctx)

    with fsspec.open("memory://omni-ingest/items.txt", "rt") as f:
        assert json.load(f)["resource"]["content"] == "resource"


@pytest.mark.asyncio
async def test_vector_store_output(vector_store):
    output = VectorStoreOutput(store=vector_store)
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"embedding": [0.1] * 1536})]
    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS
    assert await vector_store.count() == 1


@pytest.mark.asyncio
async def test_graph_store_output(vector_store):
    mock_store = AsyncMock()
    output = GraphStoreOutput(store=mock_store)
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"tags": ["tag1"]})]
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}], "edges": []}
    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS
    mock_store.upsert_nodes.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_store_output_with_edges(vector_store):
    mock_store = AsyncMock()
    output = GraphStoreOutput(store=mock_store)
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"tags": ["tag1"]})]
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
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"word_timestamps": [{"Word": "test", "Offset": 0, "Duration": 2000000}]})]
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
