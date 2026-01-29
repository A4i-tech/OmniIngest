from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omni_ingest.agent.chunking import PageChunkingAgent
from omni_ingest.agent.cleaning import (
    MarkdownCleaningAgent,
    TranscriptionPostProcessorAgent,
)
from omni_ingest.agent.enrichment import (
    FAQGenerationAgent,
    LearningOutcomeExtractionAgent,
)
from omni_ingest.agent.governance import (
    DataLineageAgent,
    QualityVerifierAgent,
)
from omni_ingest.agent.indexing import GraphIndexingAgent, VectorIndexingAgent
from omni_ingest.core.model import IngestionContext, KnowledgeItem, StepStatus


@pytest.fixture
def ctx(store):
    return IngestionContext(tenant_id=uuid4(), store=store, domain_profile="education")


@pytest.mark.asyncio
async def test_markdown_cleaning_agent(ctx):
    agent = MarkdownCleaningAgent()
    ctx.metadata["text"] = "## Raw"

    with patch("omni_ingest.agent.cleaning.Agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value.output = "## Clean"
        result = await agent.run(ctx)

        assert result.status == StepStatus.SUCCESS
        assert ctx.metadata["text"] == "## Clean"


@pytest.mark.asyncio
async def test_transcription_post_processor_agent(ctx):
    agent = TranscriptionPostProcessorAgent()
    ctx.metadata["text"] = "hello"

    with patch("omni_ingest.agent.cleaning.Agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value.output = "Hello"
        result = await agent.run(ctx)

        assert result.status == StepStatus.SUCCESS
        assert ctx.metadata["text"] == "Hello"


@pytest.mark.asyncio
async def test_quality_verifier_agent_success(ctx):
    agent = QualityVerifierAgent(min_quality_score=0.7)
    ctx.items = [KnowledgeItem(content="test content", metadata={})]

    from omni_ingest.agent.governance import QualityReport

    mock_output = QualityReport(overall_score=0.9, issues=[])

    with patch("omni_ingest.agent.governance.Agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value.output = mock_output
        result = await agent.run(ctx)

        assert result.status == StepStatus.SUCCESS
        assert "quality_report" in ctx.metadata
        assert ctx.metadata["quality_report"]["overall_score"] == 0.9


@pytest.mark.asyncio
async def test_quality_verifier_agent_failure(ctx):
    agent = QualityVerifierAgent(min_quality_score=0.7)
    ctx.items = [KnowledgeItem(content="bad content", metadata={})]

    from omni_ingest.agent.governance import QualityReport

    mock_output = QualityReport(overall_score=0.4, issues=["too short"])

    with patch("omni_ingest.agent.governance.Agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value.output = mock_output
        result = await agent.run(ctx)

        assert result.status == StepStatus.FAILURE
        assert result.error is not None and "Quality check failed" in result.error
        assert result.metadata["quality_score"] == 0.4


@pytest.mark.asyncio
async def test_faq_generation_agent(ctx):
    agent = FAQGenerationAgent()
    ctx.items = [KnowledgeItem(content="context for faq", metadata={})]

    from omni_ingest.agent.enrichment import FAQItem, FAQList

    mock_output = FAQList(faqs=[FAQItem(q="What is this?", a="A test.")])

    with patch("omni_ingest.agent.enrichment.Agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value.output = mock_output
        result = await agent.run(ctx)

        assert result.status == StepStatus.SUCCESS
        assert "faqs" in ctx.metadata
        assert len(ctx.metadata["faqs"]) == 1
        assert ctx.metadata["faqs"][0]["q"] == "What is this?"


@pytest.mark.asyncio
async def test_learning_outcome_extraction_agent(ctx):
    agent = LearningOutcomeExtractionAgent()
    ctx.domain_profile = "education"
    ctx.items = [KnowledgeItem(content="educational content", metadata={})]

    from omni_ingest.agent.enrichment import LearningOutcomes

    mock_output = LearningOutcomes(learning_outcomes=["Understand testing"])

    with patch("omni_ingest.agent.enrichment.Agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value.output = mock_output
        result = await agent.run(ctx)

        assert result.status == StepStatus.SUCCESS
        assert "learning_outcomes" in ctx.items[0].metadata
        assert ctx.items[0].metadata["learning_outcomes"] == ["Understand testing"]


@pytest.mark.asyncio
async def test_data_lineage_agent(ctx):
    agent = DataLineageAgent()
    ctx.items = [KnowledgeItem(content="test", metadata={})]
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "run_id" in ctx.items[0].lineage


@pytest.mark.asyncio
async def test_page_chunking_agent(ctx):
    agent = PageChunkingAgent()
    with patch("omni_ingest.agent.chunking.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader.return_value.pages = [mock_page]
        ctx.metadata["file_path"] = "fake.pdf"
        result = await agent.run(ctx)
        assert result.status == StepStatus.SUCCESS
        assert len(ctx.items) == 1


@pytest.mark.asyncio
async def test_vector_indexing_agent(ctx, store):
    agent = VectorIndexingAgent(store=store)
    ctx.items = [KnowledgeItem(content="test", metadata={"embedding": [0.1] * 1536})]
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_graph_indexing_agent(ctx):
    agent = GraphIndexingAgent(graph_store=AsyncMock())
    ctx.items = [KnowledgeItem(content="test", metadata={"tags": ["tag1"]})]
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}], "edges": []}
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "knowledge_graph" in ctx.items[0].structured_views


@pytest.mark.asyncio
async def test_graph_indexing_agent_with_store(ctx):
    mock_store = AsyncMock()
    agent = GraphIndexingAgent(graph_store=mock_store)
    ctx.items = [KnowledgeItem(content="test", metadata={"tags": ["tag1"]})]
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}], "edges": [{"source": "n1", "target": "n2"}]}

    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "knowledge_graph" in ctx.items[0].structured_views
    mock_store.upsert_nodes.assert_awaited_once()
    mock_store.upsert_edges.assert_awaited_once()


@pytest.mark.asyncio
async def test_audio_parser_agent_failure(ctx):
    from omni_ingest.agent.parsing import AudioParserAgent

    agent = AudioParserAgent()
    ctx.metadata["audio_path"] = "test.wav"

    with patch("omni_ingest.agent.parsing.settings") as mock_settings:
        mock_settings.azure_speech_key = None
        result = await agent.run(ctx)
        assert result.status == StepStatus.FAILURE
        assert result.error is not None and "credentials missing" in result.error


@pytest.mark.asyncio
async def test_timed_chunking_agent(ctx):
    from omni_ingest.agent.chunking import TimedChunkingAgent

    agent = TimedChunkingAgent(segment_duration_ms=100)
    ctx.metadata["text"] = "test"
    ctx.metadata["word_timestamps"] = [{"Word": "test", "Offset": 0, "Duration": 2000000}]
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS


@pytest.mark.asyncio
async def test_ingestion_logger_agent(ctx, store):
    from omni_ingest.agent.governance import IngestionLoggerAgent

    agent = IngestionLoggerAgent()
    ctx.store = store
    ctx.run_id = uuid4()
    result = await agent.run(ctx)
    assert result.status == StepStatus.SUCCESS
    assert "timestamp" in result.metadata
