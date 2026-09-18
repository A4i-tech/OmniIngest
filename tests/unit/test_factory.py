import re
import tempfile
from pathlib import Path
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

import omni_ingest.core.pipeline as pipeline_module
from omni_ingest.agent.enrichment import ExtractAgent, TransformAgent
from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepResult, StepStatus, ValidationCheck
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner, _resolve_step_kwargs, create_pipeline_from_config
from omni_ingest.port.metadata_store import NullMetadataStore


@pytest.mark.parametrize(
    "cls, config, shared, expected_keys_values",
    [
        (TransformAgent, {"value": ".metadata.x = 1", "update": ["metadata"]}, {}, {"update": ["metadata"]}),
        (TransformAgent, {"value": ".metadata.y = 2"}, {"value": ".metadata.z = 3"}, {"value": ".metadata.y = 2"}),
    ],
)
def test_resolve_step_kwargs(cls, config, shared, expected_keys_values):
    kwargs = _resolve_step_kwargs(cls, config, shared)
    for k, v in expected_keys_values.items():
        assert kwargs[k] == v


@pytest.mark.asyncio
async def test_create_pipeline_with_config():
    config = {"steps": [{"agent": "sentence_chunking", "config": {"chunk_size": 128}}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(Path(f.name))

        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], TransformAgent)
        assert "128" in pipeline.steps[0].value


def test_pipeline_groups_explicit_stages_and_generates_sequential_defaults(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "text", "stage": "prepare"}, {"agent": "lineage", "stage": "prepare"}, {"agent": "text"}]}))

    runner = create_pipeline_from_config(path)

    assert list(runner.stages)[:1] == ["prepare"]
    assert re.fullmatch(r"stage-2-[0-9a-f]{8}", list(runner.stages)[1])


def test_pipeline_generates_unique_stage_for_each_unstaged_step(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "text"}, {"agent": "lineage"}]}))

    runner = create_pipeline_from_config(path)

    assert all(re.fullmatch(rf"stage-{i}-[0-9a-f]{{8}}", stage) for i, stage in enumerate(runner.stages, 1))


def test_pipeline_rejects_stage_on_component(tmp_path):
    (tmp_path / "component.yaml").write_text(yaml.dump({"steps": [{"agent": "text"}]}))
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "component", "stage": "prepare"}]}))

    with pytest.raises(ValueError, match="Stage cannot be assigned to pipeline component component"):
        create_pipeline_from_config(root)


def test_pipeline_rejects_validation_on_component(tmp_path):
    (tmp_path / "component.yaml").write_text(yaml.dump({"steps": [{"agent": "text"}]}))
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "component", "validation": [{"mode": "plain", "expr": "true", "message": "bad"}]}]}))

    with pytest.raises(ValueError, match="Validation cannot be assigned to pipeline component component"):
        create_pipeline_from_config(root)


def test_pipeline_rejects_non_contiguous_stage(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "text", "stage": "prepare"}, {"agent": "lineage", "stage": "process"}, {"agent": "text", "stage": "prepare"}]}))

    with pytest.raises(ValueError, match="Stage prepare must be contiguous"):
        create_pipeline_from_config(path)


@pytest.mark.asyncio
async def test_pipeline_execution_with_configured_chunking(metadata_store):
    config = {"steps": [{"agent": "sentence_chunking", "config": {"chunk_size": 50}}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(Path(f.name))
        ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b"First sentence. Second sentence. Third sentence."), store=metadata_store)

        results = await pipeline.run(ctx)

        assert len(results) == 1
        assert results[0].status == StepStatus.SUCCESS
        assert results[0].metadata["update"] == ["items"]


@pytest.mark.asyncio
async def test_pipeline_before_validation_prevents_step_execution(metadata_store, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "transform", "config": {"value": "."}, "validation": [{"mode": "before", "expr": "false", "message": "blocked"}]}]}))
    pipeline = create_pipeline_from_config(path)

    results = await pipeline.run(IngestionContext(resource=ResolvedResource(uri="", raw_content=b"First sentence."), store=metadata_store))

    assert results[0].status == StepStatus.FAILURE
    assert results[0].error == "blocked"


@pytest.mark.asyncio
async def test_pipeline_condition_skips_step(metadata_store, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "transform", "config": {"value": "."}, "validation": [{"mode": "condition", "expr": "false", "message": "not requested"}]}]}))

    results = await create_pipeline_from_config(path).run(IngestionContext(resource=ResolvedResource(uri="", raw_content=b"First sentence."), store=metadata_store))

    assert results[0].status == StepStatus.SKIPPED
    assert results[0].error == "not requested"


@pytest.mark.asyncio
async def test_pipeline_after_validation_fails_with_interpolated_message(metadata_store, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "transform", "config": {"value": "."}, "validation": [{"mode": "after", "expr": "false", "message": "bad update ${.result.metadata.update}"}]}]}))
    pipeline = create_pipeline_from_config(path)

    results = await pipeline.run(IngestionContext(resource=ResolvedResource(uri="", raw_content=b"First sentence."), store=metadata_store))

    assert results[0].status == StepStatus.FAILURE
    assert results[0].error == "bad update ['metadata']"


def test_pipeline_validation_does_not_serialize_item_content():
    runner = PipelineRunner([])
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""))
    result = StepResult(status=StepStatus.SUCCESS, items=[KnowledgeItem(raw_content=b"%PDF-\xe2", metadata={"page": 1})])

    check = ValidationCheck(mode="after", expr='.result.items[0].metadata.page == 1 and (.result.items[0] | has("raw_content") | not)', message="bad")
    assert runner._validate(ctx, [check], {"after"}, result) == (None, False)


@pytest.mark.asyncio
async def test_pipeline_after_validation_skips_skipped_step(metadata_store, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "timed_chunking", "validation": [{"mode": "after", "expr": "false", "message": "should not run"}]}]}))
    pipeline = create_pipeline_from_config(path)

    results = await pipeline.run(IngestionContext(resource=ResolvedResource(uri="", raw_content=b""), store=metadata_store))

    assert results[0].status == StepStatus.SKIPPED


@pytest.mark.asyncio
async def test_pipeline_plain_validation_terminates_validation_chain(metadata_store, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "transform", "config": {"value": "."}, "validation": [{"mode": "plain", "expr": ".metadata.ok == true", "message": "missing ok"}, {"mode": "after", "expr": "false", "message": "should not run"}]}]}))
    pipeline = create_pipeline_from_config(path)
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b"First sentence."), store=metadata_store)
    ctx.metadata["ok"] = True

    results = await pipeline.run(ctx)

    assert results[0].status == StepStatus.SUCCESS


def test_pipeline_validation_requires_mode(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"steps": [{"agent": "transform", "config": {"value": "."}, "validation": [{"expr": "true", "message": "bad"}]}]}))

    with pytest.raises(ValidationError):
        create_pipeline_from_config(path)


@pytest.mark.asyncio
async def test_pipeline_with_default_config():
    config = {"steps": [{"agent": "sentence_chunking"}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(config, f)
        f.flush()

        pipeline = create_pipeline_from_config(Path(f.name))
        step = pipeline.steps[0]

        assert isinstance(step, TransformAgent)
        assert "1024" in step.value


def test_pipeline_inlines_local_yaml_component(tmp_path):
    (tmp_path / "component.yaml").write_text(yaml.dump({"steps": [{"agent": "text", "stage": "prepare"}, {"agent": "text", "stage": "prepare"}]}))
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "component"}]}))

    runner = create_pipeline_from_config(root)

    assert [type(step) for step in runner.steps] == [TextAgent, TextAgent]
    assert list(runner.stages) == ["prepare"]


def test_pipeline_inlines_profile_yaml_before_factory(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "chunking.yaml").write_text(yaml.dump({"steps": [{"agent": "text"}]}))
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "chunking"}]}))
    monkeypatch.chdir(tmp_path)

    runner = create_pipeline_from_config(root)

    assert isinstance(runner.steps[0], TextAgent)


def test_pipeline_resolves_configured_profiles_dir(tmp_path, monkeypatch):
    profiles = tmp_path / "components"
    profiles.mkdir()
    (profiles / "shared.yaml").write_text(yaml.dump({"steps": [{"agent": "text"}]}))
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "shared"}]}))
    monkeypatch.setattr(pipeline_module.settings, "profiles_dir", profiles)

    assert isinstance(create_pipeline_from_config(root).steps[0], TextAgent)


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
      path: ".metadata"
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
  - agent: sentence_chunking
    config:
      chunk_size: "${.config.chunk_size * 2}"
""")
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "component", "config": {"chunk_size": 128}}]}))

    runner = create_pipeline_from_config(root)

    assert isinstance(runner.steps[0], TransformAgent)
    assert "256" in runner.steps[0].value


def test_enrichment_profile_exposes_prompt_parameter(tmp_path):
    root = tmp_path / "root.yaml"
    root.write_text(yaml.dump({"steps": [{"agent": "enrichment", "config": {"prompt": "Custom enrichment."}}]}))

    runner = create_pipeline_from_config(root)

    assert isinstance(runner.steps[0], ExtractAgent)
    assert runner.steps[0].prompt == "Custom enrichment."  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_knowledge_base_adapter_search_text(vector_store):
    from omni_ingest.core.pipeline import KnowledgeBaseAdapter

    model = TestEmbeddingModel(dimensions=1536)
    adapter = KnowledgeBaseAdapter(
        metadata_store=NullMetadataStore(),
        vector_store=vector_store,
        embedder=Embedder(model),
    )

    result = await adapter.query_text(UUID("461724ce-ce7f-4a11-8db0-a03c2d1093d0"), "query")

    assert result == []
    assert model.last_settings == {}


@pytest.mark.asyncio
async def test_pipeline_with_undefined_toolset(metadata_store):
    config = {
        "steps": [
            {
                "agent": "extract",
                "config": {
                    "toolsets": ["converter"],
                    "path": ".metadata",
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

    result = await runner.run(IngestionContext(items=[KnowledgeItem(raw_content=b"I am clueless about the current state of LLMs.")], store=metadata_store, resource=ResolvedResource(uri="", raw_content=b"")))
    assert all(r.status == StepStatus.FAILURE for r in result)
    assert all(r.error == "Toolsets not configured in context: ['converter']" for r in result)
