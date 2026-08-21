from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import ResolvedResource, StepStatus
from omni_ingest.core.output import VectorStoreOutput
from omni_ingest.core.pipeline import IngestionContext, KnowledgeBaseAdapter, PipelineRunner, create_pipeline_from_config


@pytest.fixture
def profiles_dir():
    return Path(__file__).parent / "profiles"


@pytest.mark.parametrize("p", [
    "document_v1",
    "textbook_v1",
    "audio_v1",
    "health_v1",
    "education_v2",
    "audio_v2",
    "governance_v1",
    "precision_v1",
    "noisy_v1",
    "custom_prompt_v1",
])
def test_profiles_instantiation(p, profiles_dir):
    path = profiles_dir / f"{p}.yaml"
    runner = create_pipeline_from_config(path)
    assert runner is not None
    assert len(runner.steps) > 0


@pytest.mark.asyncio
async def test_education_v2_pipeline(metadata_store, tenant, profiles_dir):
    path = profiles_dir / "education_v2.yaml"
    runner = create_pipeline_from_config(path)
    ctx = IngestionContext(
        tenant_id=tenant.id,
        store=metadata_store,
        domain_profile="education",
        resource=ResolvedResource(uri="", raw_content=b"Students will learn the advanced concepts of Physics."),
    )

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"

    assert len(ctx.items) > 0
    assert "learning_outcomes" in ctx.items[0].metadata
    assert "summary" in ctx.items[0].metadata


@pytest.mark.asyncio
async def test_audio_v2_pipeline(metadata_store, tenant, profiles_dir, resources_dir):
    path = profiles_dir / "audio_v2.yaml"
    runner = create_pipeline_from_config(path)

    ctx = IngestionContext(tenant_id=tenant.id, store=metadata_store, resource=resources_dir / "sample.mp3")
    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"
    assert ctx.metadata["transcript"].strip()
    assert ctx.items
    texts = [await item.decode(ctx) for item in ctx.items]
    assert all(text.strip() for text in texts)


@pytest.mark.asyncio
async def test_governance_v1_pipeline(metadata_store, tenant, profiles_dir):
    path = profiles_dir / "governance_v1.yaml"
    runner = create_pipeline_from_config(path)
    ctx = IngestionContext(tenant_id=tenant.id, store=metadata_store, resource=ResolvedResource(uri="", raw_content=b"Audit this file for compliance."))

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"

    assert any(r.metadata and "timestamp" in r.metadata for r in results)


@pytest.mark.asyncio
async def test_health_v1_pipeline(metadata_store, tenant, profiles_dir):
    path = profiles_dir / "health_v1.yaml"
    runner = create_pipeline_from_config(path)
    ctx = IngestionContext(tenant_id=tenant.id, store=metadata_store, resource=ResolvedResource(uri="", raw_content=b"The patient was diagnosed with type 2 diabetes."))

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"
    assert ctx.items[0].metadata["summary"]
    assert ctx.items[0].metadata["tags"]


@pytest.mark.asyncio
async def test_semantic_ingestion_and_search(tmp_path, metadata_store, vector_store, resources_dir):
    config = {
        "steps": [
            {"agent": "chunking", "config": {"chunk_size": 200, "overlap": 20}},
            {"agent": "embedding"},
        ],
    }
    config_path = tmp_path / "pipeline.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f)

    pipeline = create_pipeline_from_config(config_path)
    output = VectorStoreOutput(store=vector_store)

    tid = uuid4()

    blueberries_path = resources_dir / "blueberries-are-now-green.txt"
    res1 = await pipeline.run(IngestionContext(tenant_id=tid, resource=blueberries_path, store=metadata_store), output=output)
    assert res1[-1].status == StepStatus.SUCCESS

    patients_path = resources_dir / "all-patients-are-stable-except-when-they-arent.md"
    res2 = await pipeline.run(IngestionContext(tenant_id=tid, resource=patients_path, store=metadata_store), output=output)
    assert res2[-1].status == StepStatus.SUCCESS

    adapter = KnowledgeBaseAdapter(metadata_store=metadata_store, vector_store=vector_store)

    query_ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""))
    results = await adapter.query_text(tenant_id=tid, query="what color are blueberries", k=3)
    assert len(results) > 0
    assert any(["blueberries" in (await r.decode(query_ctx)).lower() for r in results])
    assert not any(["jaipur" in (await r.decode(query_ctx)).lower() for r in results])

    results_j = await adapter.query_text(tenant_id=tid, query="Jaipur Protocol", k=3)
    assert len(results_j) > 0
    assert any(["jaipur" in (await r.decode(query_ctx)).lower() for r in results_j])
    assert not any(["blueberries" in (await r.decode(query_ctx)).lower() for r in results_j])


@pytest.mark.asyncio
async def test_image_to_text_pipeline(metadata_store, resources_dir):
    image_path = resources_dir / "supermarket-shelf.webp"
    ctx = IngestionContext(resource=ResolvedResource(uri=str(image_path), raw_content=image_path.read_bytes()), store=metadata_store)

    result = (await PipelineRunner([TextAgent()]).run(ctx))[0]
    assert result.status == StepStatus.SUCCESS, f"Pipeline failed: {result.error}"

    text = (await ctx.items[0].decode(ctx)).lower()
    assert ctx.items[0].metadata["selected_parser"] == "image"
    assert any(word in text for word in ["shelf", "store", "supermarket", "product", "package"])
