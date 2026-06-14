from uuid import uuid4

import pytest
import yaml

from omni_ingest.core.model import StepStatus
from omni_ingest.core.output import VectorStoreOutput
from omni_ingest.core.pipeline import IngestionContext, KnowledgeBaseAdapter, create_pipeline_from_config


@pytest.mark.asyncio
async def test_byoeb_ingestion_and_search(tmp_path, metadata_store, vector_store, resources_dir):
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
    assert any("blueberries" in r.decode().lower() for r in results)
    assert not any("jaipur" in r.decode().lower() for r in results)

    results_j = await adapter.query_text(tenant_id=tid, query="Jaipur Protocol", k=3)
    assert len(results_j) > 0
    assert any("jaipur" in r.decode().lower() for r in results_j)
    assert not any("blueberries" in r.decode().lower() for r in results_j)
