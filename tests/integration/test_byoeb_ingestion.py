from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omni_ingest.core.model import IngestionContext, StepStatus
from omni_ingest.core.pipeline import KnowledgeBaseAdapter, create_pipeline_from_config, lookup_step_id
from omni_ingest.parser import read_resources

RESOURCES_DIR = Path(__file__).parent.parent / "resources"


@pytest.mark.asyncio
async def test_byoeb_ingestion_and_search(tmp_path, metadata_store):
    config = {
        "store": {"type": "in_memory"},
        "steps": [
            {"agent": "chunking", "config": {"chunk_size": 200, "overlap": 20}},
            {"agent": "embedding"},
            {"agent": "indexing", "config": {"vector_store": "in_memory"}},
        ],
    }
    config_path = tmp_path / "pipeline.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f)

    pipeline = create_pipeline_from_config(config_path)

    tid = uuid4()

    blueberries_path = RESOURCES_DIR / "blueberries-are-now-green.txt"

    ctxs = [IngestionContext(tenant_id=tid, resource=resource, store=metadata_store) async for resource in read_resources(str(blueberries_path))]
    assert len(ctxs) == 1
    ctx1 = ctxs[0]
    res1 = await pipeline.run(ctx1)
    assert res1[-1].status == StepStatus.SUCCESS

    indexing_step = next(s for s in pipeline.steps if lookup_step_id(s) == "indexing")
    vector_store = indexing_step.store  # type: ignore

    patients_path = RESOURCES_DIR / "all-patients-are-stable-except-when-they-arent.md"

    ctxs = [IngestionContext(tenant_id=tid, resource=resource, store=metadata_store) async for resource in read_resources(str(patients_path))]
    assert len(ctxs) == 1
    ctx2 = ctxs[0]
    res2 = await pipeline.run(ctx2)
    assert res2[-1].status == StepStatus.SUCCESS

    class DummyMetadataStore:
        async def list_knowledge_items(self, *args, **kwargs):
            return []

        async def get_knowledge_item(self, *args, **kwargs):
            return None

        async def create_knowledge_item(self, *args, **kwargs):
            return uuid4()

        async def create_chunk(self, *args, **kwargs):
            return "chunk_id"

        async def create_pipeline_run(self, *args, **kwargs):
            return uuid4()

        async def update_pipeline_run(self, *args, **kwargs):
            pass

        async def create_step_run(self, *args, **kwargs):
            return uuid4()

        async def update_step_run(self, *args, **kwargs):
            pass

    adapter = KnowledgeBaseAdapter(metadata_store=DummyMetadataStore(), vector_store=vector_store)  # type: ignore

    results = await adapter.query_text(tenant_id=tid, query="what color are blueberries", k=3)
    assert len(results) > 0
    assert any("blueberries" in r.content.lower() for r in results)
    assert not any("jaipur" in r.content.lower() for r in results)

    results_j = await adapter.query_text(tenant_id=tid, query="Jaipur Protocol", k=3)
    assert len(results_j) > 0
    assert any("jaipur" in r.content.lower() for r in results_j)
    assert not any("blueberries" in r.content.lower() for r in results_j)
