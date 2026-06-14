from unittest.mock import AsyncMock

import pytest

from omni_ingest.agent.structure import StructureAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import GraphStoreOutput
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner


@pytest.mark.asyncio
async def test_domain_enrichment_flow(metadata_store):
    agent = StructureAgent(
        method="update",
        prompt="Analyze text and provide JSON summary/tags. Content: {text}",
        json_schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary", "tags"],
        },
    )
    item = KnowledgeItem(content=b"Sample text about machine learning and neural networks.")
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", content=b""))

    res = await PipelineRunner([agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS, f"Step failed: {res[0].error}"
    assert "summary" in item.metadata
    assert "tags" in item.metadata and len(item.metadata["tags"]) > 0


@pytest.mark.asyncio
async def test_learning_outcome_extraction(metadata_store):
    ed_agent = StructureAgent(
        metadata_key="learning_outcomes",
        prompt="Extract a list of learning outcomes from the following educational content. Content: {text}",
        json_schema={
            "type": "array",
            "items": {"type": "string"},
        },
    )
    item = KnowledgeItem(content=b"The student will learn about Newton's laws of motion.")
    ctx = IngestionContext(items=[item], store=metadata_store, domain_profile="education", resource=ResolvedResource(uri="", content=b""))

    res = await PipelineRunner([ed_agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS, f"Step failed: {res[0].error}"
    assert "learning_outcomes" in item.metadata


@pytest.mark.asyncio
async def test_structure_agent_does_not_branch_on_domain(metadata_store):
    ed_agent = StructureAgent(
        metadata_key="learning_outcomes",
        prompt="Extract a list of learning outcomes from the following educational content. Content: {text}",
        json_schema={
            "type": "array",
            "items": {"type": "string"},
        },
    )
    item = KnowledgeItem(content=b"Some content.")
    ctx = IngestionContext(items=[item], store=metadata_store, domain_profile="generic", resource=ResolvedResource(uri="", content=b""))

    res = await PipelineRunner([ed_agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS
    assert "learning_outcomes" in item.metadata


@pytest.mark.asyncio
async def test_graph_store_output(metadata_store):
    output = GraphStoreOutput(store=AsyncMock())
    item = KnowledgeItem(content=b"Test content")
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", content=b""))
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": []}

    res = await output.write(ctx)
    assert res.status == StepStatus.SUCCESS
    assert res.metadata["graph_nodes"] == 2
