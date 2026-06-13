import tempfile
from pathlib import Path
from uuid import UUID

import pytest
import yaml

import omni_ingest.core.pipeline as pipeline_module
from omni_ingest.agent.chunking import SentenceChunkingAgent
from omni_ingest.agent.indexing import VectorIndexingAgent
from omni_ingest.core.config import InMemoryVectorStoreConfig, LanceDBVectorStoreConfig
from omni_ingest.core.model import IngestionContext, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import _resolve_step_kwargs, create_pipeline_from_config
from omni_ingest.port.vector_store import InMemoryVectorStore, LanceDBVectorStore


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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(Path(f.name))

        assert len(pipeline.steps) == 1
        chunking_step = pipeline.steps[0]
        assert isinstance(chunking_step, SentenceChunkingAgent)
        assert chunking_step.chunk_size == 128
        assert chunking_step.overlap == 25


@pytest.mark.asyncio
async def test_pipeline_execution_with_configured_chunking(metadata_store):
    config = {"steps": [{"agent": "chunking", "config": {"chunk_size": 50, "overlap": 10}}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(Path(f.name))
        ctx = IngestionContext(resource=ResolvedResource(uri="", content=b"First sentence. Second sentence. Third sentence."), store=metadata_store)

        results = await pipeline.run(ctx)

        assert len(results) == 1
        assert results[0].status == StepStatus.SUCCESS
        assert results[0].metadata["chunk_size"] == 50


@pytest.mark.asyncio
async def test_pipeline_with_default_config():
    config = {"steps": [{"agent": "chunking"}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(Path(f.name))
        chunking_step = pipeline.steps[0]

        assert chunking_step.chunk_size == 512  # type: ignore
        assert chunking_step.overlap == 50  # type: ignore


def test_create_vector_store_uses_env_backed_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pipeline_module.settings,
        "vector_store",
        LanceDBVectorStoreConfig(table_name="env_vectors", path=tmp_path / "lancedb", vector_size=256),
    )

    store = pipeline_module.settings.vector_store.build()

    assert isinstance(store, LanceDBVectorStore)
    assert store.table_name == "env_vectors"
    assert store.vector_size == 256


@pytest.mark.asyncio
async def test_knowledge_base_adapter_search_text():
    from unittest.mock import AsyncMock, patch

    from omni_ingest.core.pipeline import KnowledgeBaseAdapter

    mock_store = AsyncMock()
    mock_vector = AsyncMock()
    adapter = KnowledgeBaseAdapter(metadata_store=mock_store, vector_store=mock_vector)

    with patch.object(adapter.embedder, "embed_query", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value.embeddings = [[0.1] * 1536]

        await adapter.query_text(UUID("461724ce-ce7f-4a11-8db0-a03c2d1093d0"), "query")

        mock_embed.assert_awaited_once_with("query")
        mock_vector.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_store_selection_ignores_yaml_storage_config(monkeypatch):
    monkeypatch.setattr(pipeline_module.settings, "vector_store", InMemoryVectorStoreConfig(vector_size=1536))

    config = {
        "store": {"type": "qdrant", "collection": "yaml_should_be_ignored"},
        "steps": [{"agent": "indexing"}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
        yaml.dump(config, tmp)
        tmp.flush()
        runner = create_pipeline_from_config(Path(tmp.name))

    assert isinstance(runner.vector_store, InMemoryVectorStore)
    assert isinstance(runner.steps[0], VectorIndexingAgent)
    assert runner.steps[0].store is runner.vector_store


@pytest.mark.asyncio
async def test_pipeline_with_undefined_toolset(metadata_store):
    config = {"steps": [{"agent": "enrichment", "config": {"toolsets": ["converter"]}}]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
        yaml.dump(config, tmp)
        runner = create_pipeline_from_config(Path(tmp.name))

    result = await runner.run(IngestionContext(items=[KnowledgeItem(content=b"I am clueless about the current state of LLMs.")], store=metadata_store, resource=ResolvedResource(uri="", content=b"")))
    assert all(r.status == StepStatus.FAILURE for r in result)
    assert all(r.error == "Toolsets not configured in context: ['converter']" for r in result)
