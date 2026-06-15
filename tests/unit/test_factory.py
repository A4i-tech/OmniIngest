import tempfile
from pathlib import Path
from uuid import UUID

import pytest
import yaml

import omni_ingest.core.pipeline as pipeline_module
from omni_ingest.agent.chunking import SentenceChunkingAgent
from omni_ingest.agent.enrichment import ExtractAgent
from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.config import LanceDBVectorStoreConfig
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, _resolve_step_kwargs, create_pipeline_from_config
from omni_ingest.port.vector_store import LanceDBVectorStore


@pytest.mark.parametrize(
    "cls, config, shared, expected_keys_values",
    [
        (SentenceChunkingAgent, {"chunk_size": 256, "overlap": 30}, {}, {"chunk_size": 256, "overlap": 30}),
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


def test_pipeline_inlines_local_yaml_component(tmp_path):
    (tmp_path / "component.yaml").write_text(yaml.dump({"steps": [{"agent": "text"}, {"agent": "chunking", "config": {"chunk_size": 64}}]}))
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "component"}]}))

    runner = create_pipeline_from_config(root)

    assert [type(step) for step in runner.steps] == [TextAgent, SentenceChunkingAgent]
    assert runner.steps[1].chunk_size == 64  # type: ignore


def test_pipeline_inlines_profile_yaml_before_factory(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "chunking.yaml").write_text(yaml.dump({"steps": [{"agent": "text"}]}))
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "chunking"}]}))
    monkeypatch.chdir(tmp_path)

    runner = create_pipeline_from_config(root)

    assert len(runner.steps) == 1
    assert isinstance(runner.steps[0], TextAgent)


def test_pipeline_detects_component_cycles(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.dump({"steps": [{"agent": "b"}]}))
    (tmp_path / "b.yaml").write_text(yaml.dump({"steps": [{"agent": "a"}]}))

    with pytest.raises(ValueError, match="Pipeline component cycle detected"):
        create_pipeline_from_config(tmp_path / "a.yaml")


def test_pipeline_component_parameters_apply_defaults_and_overrides(tmp_path):
    (tmp_path / "component.yaml").write_text("""
parameters:
  properties:
    prompt:
      type: string
      default: "Default prompt."
steps:
  - agent: extract
    config:
      into: .metadata
      prompt: "${.config.prompt}"
      schema:
        type: object
        properties:
          summary:
            type: string
        required: [summary]
""")
    default_root = tmp_path / "default.yaml"
    default_root.write_text(yaml.dump({"steps": [{"agent": "component"}]}))
    override_root = tmp_path / "override.yaml"
    override_root.write_text(yaml.dump({"steps": [{"agent": "component", "config": {"prompt": "Custom prompt."}}]}))

    default_runner = create_pipeline_from_config(default_root)
    override_runner = create_pipeline_from_config(override_root)

    assert isinstance(default_runner.steps[0], ExtractAgent)
    assert default_runner.steps[0].prompt == "Default prompt."  # type: ignore[attr-defined]
    assert override_runner.steps[0].prompt == "Custom prompt."  # type: ignore[attr-defined]


def test_pipeline_component_parameters_reject_undeclared_keys(tmp_path):
    (tmp_path / "component.yaml").write_text("""
parameters:
  properties:
    prompt:
      type: string
      default: "Default prompt."
steps:
  - agent: text
""")
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "component", "config": {"unknown": "value"}}]}))

    with pytest.raises(ValueError, match="Invalid parameters"):
        create_pipeline_from_config(root)


def test_pipeline_parameters_support_jq_and_preserve_full_value_type(tmp_path):
    (tmp_path / "component.yaml").write_text("""
parameters:
  properties:
    chunk_size:
      type: integer
      default: 64
steps:
  - agent: chunking
    config:
      chunk_size: "${.config.chunk_size * 2}"
""")
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "component", "config": {"chunk_size": 128}}]}))

    runner = create_pipeline_from_config(root)

    assert isinstance(runner.steps[0], SentenceChunkingAgent)
    assert runner.steps[0].chunk_size == 256  # type: ignore[attr-defined]


def test_enrichment_profile_exposes_prompt_parameter(tmp_path):
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "enrichment", "config": {"prompt": "Custom enrichment."}}]}))

    runner = create_pipeline_from_config(root)

    assert isinstance(runner.steps[0], ExtractAgent)
    assert runner.steps[0].prompt == "Custom enrichment."  # type: ignore[attr-defined]


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
async def test_pipeline_with_undefined_toolset(metadata_store):
    config = {
        "steps": [
            {
                "agent": "extract",
                "config": {
                    "toolsets": ["converter"],
                    "into": ".metadata",
                    "prompt": "Analyze text.",
                    "schema": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                    },
                },
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
        yaml.dump(config, tmp)
        runner = create_pipeline_from_config(Path(tmp.name))

    result = await runner.run(IngestionContext(items=[KnowledgeItem(content=b"I am clueless about the current state of LLMs.")], store=metadata_store, resource=ResolvedResource(uri="", content=b"")))
    assert all(r.status == StepStatus.FAILURE for r in result)
    assert all(r.error == "Toolsets not configured in context: ['converter']" for r in result)
