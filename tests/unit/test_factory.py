import tempfile

import pytest
import yaml

from omni_ingest.agent.chunking import SentenceChunkingAgent
from omni_ingest.agent.indexing import VectorIndexingAgent
from omni_ingest.core.model import IngestionContext, StepStatus
from omni_ingest.core.pipeline import _resolve_step_kwargs, create_pipeline_from_config


@pytest.mark.parametrize(
    "cls, config, shared, expected_keys_values",
    [
        (SentenceChunkingAgent, {"chunk_size": 256, "overlap": 30}, {}, {"chunk_size": 256, "overlap": 30}),
        (VectorIndexingAgent, {}, {"store": "MOCK_STORE"}, {"store": "MOCK_STORE"}),
        (SentenceChunkingAgent, {"chunk_size": 100}, {"chunk_size": 999}, {"chunk_size": 100}),
    ],
)
def test_resolve_step_kwargs(cls, config, shared, expected_keys_values):
    kwargs = _resolve_step_kwargs(cls, config, shared)
    for k, v in expected_keys_values.items():
        assert kwargs[k] == v


@pytest.mark.asyncio
async def test_create_pipeline_with_config():
    config = {"steps": [{"agent": "chunking", "config": {"chunk_size": 128, "overlap": 25}}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(f.name)

        assert len(pipeline.steps) == 1
        chunking_step = pipeline.steps[0]
        assert isinstance(chunking_step, SentenceChunkingAgent)
        assert chunking_step.chunk_size == 128
        assert chunking_step.overlap == 25


@pytest.mark.asyncio
async def test_pipeline_execution_with_configured_chunking():
    config = {"steps": [{"agent": "chunking", "config": {"chunk_size": 50, "overlap": 10}}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(f.name)
        ctx = IngestionContext(metadata={"text": "First sentence. Second sentence. Third sentence."})

        results = await pipeline.run(ctx)

        assert len(results) == 1
        assert results[0].status == StepStatus.SUCCESS
        assert results[0].metadata["chunk_size"] == 50


@pytest.mark.asyncio
async def test_pipeline_with_default_config():
    config = {"steps": [{"agent": "chunking"}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(f.name)
        chunking_step = pipeline.steps[0]

        assert chunking_step.chunk_size == 512  # type: ignore
        assert chunking_step.overlap == 50  # type: ignore


@pytest.mark.asyncio
async def test_knowledge_base_adapter_search_text():
    from unittest.mock import AsyncMock, MagicMock, patch

    from omni_ingest.core.pipeline import KnowledgeBaseAdapter

    mock_store = AsyncMock()
    mock_vector = AsyncMock()
    adapter = KnowledgeBaseAdapter(metadata_store=mock_store, vector_store=mock_vector)

    with patch("omni_ingest.core.pipeline._get_openai_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 1536)]

        await adapter.query_text("tenant-1", "query")

        mock_client.embeddings.create.assert_awaited_once()
        mock_vector.search.assert_awaited_once()
