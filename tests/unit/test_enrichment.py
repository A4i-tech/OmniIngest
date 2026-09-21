import asyncio
import json
from uuid import uuid4

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, BinaryContent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from omni_ingest.agent.enrichment import ExtractAgent, PromptAgent, TransformAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner
from omni_ingest.port.metadata_store import NullMetadataStore


def create_ctx(content: str = "") -> IngestionContext:
    return IngestionContext(
        tenant_id=uuid4(),
        resource=ResolvedResource(uri="", content=ByteContent.from_bytes(content.encode())),
        store=NullMetadataStore(),
    )


def use_test_model(agent: ExtractAgent, output, monkeypatch) -> TestModel:
    model = TestModel(
        custom_output_text=json.dumps({"output": output}),
        profile={"supports_json_schema_output": True},
    )
    monkeypatch.setattr(agent, "_agent", lambda _ctx, **kwargs: Agent(model, **kwargs))
    return model


@pytest.mark.asyncio
async def test_prompt_agent():
    agent = PromptAgent(model="test", prompt="Clean this Markdown.", input=".metadata.text", path=".metadata.cleaned")
    ctx = create_ctx("## Raw")
    ctx.metadata["text"] = "## Raw"

    results = await PipelineRunner([agent]).run(ctx)

    assert results[0].status == StepStatus.SUCCESS
    assert ctx.metadata["cleaned"] == "success (no tool calls)"


def test_extract_agent_runtime_schema():
    agent = ExtractAgent(path=".metadata.answers", prompt="Answer.", schema=".metadata.schema")
    output_type = agent._output_type(
        {
            "metadata": {
                "schema": {
                    "type": "object",
                    "properties": {"a1": {"type": "string"}, "a2": {"type": "string"}},
                    "required": ["a1", "a2"],
                }
            }
        }
    )

    assert output_type(output={"a1": "a", "a2": "b"}).model_dump(mode="json")["output"] == {"a1": "a", "a2": "b"}
    with pytest.raises(ValueError):
        output_type(output={"a1": "a"})


@pytest.mark.asyncio
async def test_extract_agent_merges_item_metadata(monkeypatch):
    agent = ExtractAgent(
        path=".metadata",
        prompt="Enrich.",
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary", "tags"],
        },
    )
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"content"), metadata={"existing": "keep"})]
    use_test_model(agent, {"summary": "s", "tags": ["t"]}, monkeypatch)

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.items[0].metadata == {"existing": "keep", "summary": "s", "tags": ["t"]}


@pytest.mark.asyncio
async def test_extract_agent_jq_foreach_and_input(monkeypatch):
    agent = ExtractAgent(
        foreach=".items[:1][]",
        input=".text[:4]",
        path=".metadata.keywords",
        prompt="Keywords.",
        schema={"type": "array", "items": {"type": "string"}},
    )
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"alpha")), KnowledgeItem(content=ByteContent.from_bytes(b"bravo"))]
    use_test_model(agent, ["new"], monkeypatch)

    with capture_run_messages() as messages:
        result = await agent.run(ctx)

    assert messages[0].parts[0].content == ["Keywords.", "alph"]
    assert result.metadata["items_processed"] == 1
    assert ctx.items[0].metadata["keywords"] == ["new"]
    assert ctx.items[1].metadata == {}


@pytest.mark.asyncio
async def test_extract_agent_document_scope(monkeypatch):
    agent = ExtractAgent(
        foreach=".",
        input='[.items[:2][] | .text[:4]] | join("|")',
        path=".metadata.faqs",
        prompt="Generate FAQs.",
        schema={"type": "array", "items": {"type": "string"}},
    )
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(value)) for value in (b"alpha", b"bravo", b"charlie")]
    use_test_model(agent, ["answer"], monkeypatch)

    with capture_run_messages() as messages:
        await agent.run(ctx)

    assert messages[0].parts[0].content == ["Generate FAQs.", "alph|brav"]
    assert ctx.metadata["faqs"] == ["answer"]


@pytest.mark.asyncio
async def test_extract_agent_binary_input_uses_selected_item(monkeypatch):
    agent = ExtractAgent(
        foreach=".",
        input="[.items[] | select(.metadata.page == 2) | .id]",
        path=".metadata.document",
        prompt="Extract metadata.",
        schema={"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]},
    )
    ctx = create_ctx()
    ctx.items = [
        KnowledgeItem(content=ByteContent.from_bytes(b"\xffpage 1", content_type="application/pdf"), metadata={"page": 1}),
        KnowledgeItem(content=ByteContent.from_bytes(b"\xffpage 2", content_type="application/pdf"), metadata={"page": 2}),
    ]
    use_test_model(agent, {"subject": "Mathematics"}, monkeypatch)

    with capture_run_messages() as messages:
        await agent.run(ctx)

    content = messages[0].parts[0].content
    assert content[0] == "Extract metadata."
    assert isinstance(content[1], BinaryContent)
    assert content[1].data == b"\xffpage 2"
    assert ctx.metadata["document"] == {"subject": "Mathematics"}


@pytest.mark.asyncio
async def test_extract_agent_buffered_mode_registers_item_reader(monkeypatch):
    agent = ExtractAgent(
        foreach=".",
        mode="buffered",
        input="[.items[] | .id]",
        path=".metadata.toc",
        prompt="Find TOC.",
        schema={"type": "object", "properties": {"found": {"type": "boolean"}}, "required": ["found"]},
    )
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"content"))]
    model = TestModel(
        custom_output_text=json.dumps({"output": {"found": True}}),
        profile={"supports_json_schema_output": True},
    )
    pydantic_agents = []

    def create_agent(_ctx, **kwargs):
        pydantic_agents.append(Agent(model, **kwargs))
        return pydantic_agents[-1]

    monkeypatch.setattr(agent, "_agent", create_agent)

    await agent.run(ctx)

    assert "see_next" in pydantic_agents[0]._function_toolset.tools
    assert ctx.metadata["toc"] == {"found": True}


@pytest.mark.asyncio
async def test_concurrent_extract_agents_merge_metadata(monkeypatch):
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"content"), metadata={"existing": "keep"})]
    agents = [ExtractAgent(path=f".metadata.generated.{key}", prompt=key, schema={"type": "string"}) for key in ("first", "second")]

    async def extract(agent, _ctx, item_doc, _item_by_id, _sem, _advance):
        await asyncio.sleep(0.01 if agent.path.endswith("first") else 0)
        return item_doc, agent.prompt

    monkeypatch.setattr(ExtractAgent, "_extract", extract)
    await asyncio.gather(*(agent.run(ctx) for agent in agents))

    assert ctx.items[0].metadata == {"existing": "keep", "generated": {"first": "first", "second": "second"}}


@pytest.mark.asyncio
async def test_concurrent_extract_agents_merge_root_metadata(monkeypatch):
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"content"))]
    agents = [ExtractAgent(foreach=".", path=f".metadata.{key}", prompt=key, schema={"type": "string"}) for key in ("first", "second")]

    async def extract(agent, _ctx, item_doc, _item_by_id, _sem, _advance):
        await asyncio.sleep(0.01 if agent.path.endswith("first") else 0)
        return item_doc, agent.prompt

    monkeypatch.setattr(ExtractAgent, "_extract", extract)
    await asyncio.gather(*(agent.run(ctx) for agent in agents))

    assert ctx.metadata == {"first": "first", "second": "second"}


@pytest.mark.asyncio
async def test_extract_agent_assertion_retries_with_feedback(monkeypatch):
    agent = ExtractAgent(
        foreach=".",
        path=".metadata.quality",
        prompt="Quality.",
        assertions=[{"expr": ".score >= 0.7", "feedback": "Quality is too low"}],
        schema={"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]},
    )
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"content"))]
    use_test_model(agent, {"score": 0.4}, monkeypatch)

    with capture_run_messages() as messages, pytest.raises(UnexpectedModelBehavior):
        await agent.run(ctx)

    assert messages[2].parts[0].content == "Quality is too low"


@pytest.mark.asyncio
async def test_transform_agent_writes_jq_output_to_metadata():
    agent = TransformAgent(value=".metadata.names = (.metadata.items | map(.name))")
    ctx = create_ctx()
    ctx.metadata["items"] = [{"name": "a"}, {"name": "b"}]

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["names"] == ["a", "b"]


@pytest.mark.asyncio
async def test_transform_agent_emits_items():
    agent = TransformAgent(value=".items |= map(. + {text: (.text | ascii_upcase)})", update=["items"])
    ctx = create_ctx()
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"hello", content_type="text/plain"))]

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert len(ctx.items) == 1
    assert await ctx.items[0].content.decode(ctx) == "HELLO"


@pytest.mark.asyncio
async def test_transform_agent_maps_nested_metadata():
    agent = TransformAgent(
        value=r"""
        .metadata.records = (
            .metadata.book as $book
            | .metadata.chapters
            | map({
                id: "\($book.board)-\(.no)",
                title: .name,
                topics: (.topics | map(.numbering + ". " + .title))
                })
        )
    """,
    )
    ctx = create_ctx()
    ctx.metadata = {
        "book": {"board": "BSE-TG"},
        "chapters": [{"no": 1, "name": "Our Environment", "topics": [{"numbering": "1.1", "title": "Introduction"}]}],
    }

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata["records"] == [{"id": "BSE-TG-1", "title": "Our Environment", "topics": ["1.1. Introduction"]}]


def test_transform_agent_rejects_invalid_jq():
    with pytest.raises(ValidationError):
        TransformAgent(value=".metadata |")
