from unittest.mock import AsyncMock

import pytest

from omni_ingest.agent.enrichment import DomainEnrichmentAgent, LearningOutcomeExtractionAgent
from omni_ingest.agent.indexing import GraphIndexingAgent
from omni_ingest.core.model import IngestionContext, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import PipelineRunner


@pytest.mark.asyncio
async def test_domain_enrichment_flow(metadata_store):
    agent = DomainEnrichmentAgent()
    item = KnowledgeItem(content="Sample text about machine learning and neural networks.")
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", content=""))

    res = await PipelineRunner([agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS, f"Step failed: {res[0].error}"
    assert "summary" in item.metadata
    assert "tags" in item.metadata and len(item.metadata["tags"]) > 0


@pytest.mark.asyncio
async def test_learning_outcome_extraction(metadata_store):
    ed_agent = LearningOutcomeExtractionAgent()
    item = KnowledgeItem(content="The student will learn about Newton's laws of motion.")
    ctx = IngestionContext(items=[item], store=metadata_store, domain_profile="education", resource=ResolvedResource(uri="", content=""))

    res = await PipelineRunner([ed_agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS, f"Step failed: {res[0].error}"
    assert "learning_outcomes" in item.metadata


@pytest.mark.asyncio
async def test_learning_outcome_skips_non_education(metadata_store):
    ed_agent = LearningOutcomeExtractionAgent()
    item = KnowledgeItem(content="Some content.")
    ctx = IngestionContext(items=[item], store=metadata_store, domain_profile="generic", resource=ResolvedResource(uri="", content=""))

    res = await PipelineRunner([ed_agent]).run(ctx)
    assert res[0].status == StepStatus.SKIPPED


@pytest.mark.asyncio
async def test_graph_indexing_agent(metadata_store):
    graph_agent = GraphIndexingAgent(graph_store=AsyncMock())
    item = KnowledgeItem(content="Test content")
    ctx = IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", content=""))
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": []}

    res = await PipelineRunner([graph_agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS
    assert res[0].metadata["graph_nodes"] == 2
    assert item.structured_views["knowledge_graph"]["nodes"][0]["id"] == "n1"
