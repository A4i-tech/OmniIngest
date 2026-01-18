from unittest.mock import AsyncMock

import pytest

from omni_ingest.agent.enrichment import DomainEnrichmentAgent, LearningOutcomeExtractionAgent
from omni_ingest.agent.indexing import GraphIndexingAgent
from omni_ingest.core.config import settings
from omni_ingest.core.model import IngestionContext, KnowledgeItem, StepStatus


@pytest.fixture
def skip_if_no_openai():
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")


@pytest.mark.asyncio
async def test_domain_enrichment_flow(skip_if_no_openai):
    agent = DomainEnrichmentAgent()
    item = KnowledgeItem(content="Sample text about machine learning and neural networks.")
    ctx = IngestionContext(items=[item])

    res = await agent.run(ctx)
    assert res.status == StepStatus.SUCCESS
    assert "summary" in item.metadata
    assert "tags" in item.metadata and len(item.metadata["tags"]) > 0


@pytest.mark.asyncio
async def test_learning_outcome_extraction(skip_if_no_openai):
    ed_agent = LearningOutcomeExtractionAgent()
    item = KnowledgeItem(content="The student will learn about Newton's laws of motion.")
    ctx = IngestionContext(items=[item], domain_profile="education")

    res = await ed_agent.run(ctx)
    assert res.status == StepStatus.SUCCESS
    assert "learning_outcomes" in item.metadata


@pytest.mark.asyncio
async def test_learning_outcome_skips_non_education():
    ed_agent = LearningOutcomeExtractionAgent()
    item = KnowledgeItem(content="Some content.")
    ctx = IngestionContext(items=[item], domain_profile="generic")

    res = await ed_agent.run(ctx)
    assert res.status == StepStatus.SKIPPED


@pytest.mark.asyncio
async def test_graph_indexing_agent():
    graph_agent = GraphIndexingAgent(graph_store=AsyncMock())
    item = KnowledgeItem(content="Test content")
    ctx = IngestionContext(items=[item])
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": []}

    res = await graph_agent.run(ctx)
    assert res.status == StepStatus.SUCCESS
    assert res.metadata["graph_nodes"] == 2
    assert item.structured_views["knowledge_graph"]["nodes"][0]["id"] == "n1"
