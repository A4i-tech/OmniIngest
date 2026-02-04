import tempfile

import pytest
import yaml

from omni_ingest.agent.chunking import SentenceChunkingAgent
from omni_ingest.agent.indexing import EmbeddingAgent, VectorIndexingAgent
from omni_ingest.core.model import IngestionContext, Step, StepResult, StepStatus
from omni_ingest.core.pipeline import PipelineRunner, create_pipeline_from_config
from omni_ingest.port.vector_store import InMemoryVectorStore


class EchoStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        return StepResult(status=StepStatus.SUCCESS, metadata={"echo": "echo"})


class FailingStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        return StepResult(status=StepStatus.FAILURE, error="Boom")


@pytest.mark.asyncio
async def test_pipeline_execution_basic(metadata_store):
    runner = PipelineRunner(steps=[EchoStep()])
    results = await runner.run(IngestionContext(store=metadata_store))
    assert results[0].status == StepStatus.SUCCESS and results[0].metadata["echo"] == "echo"


@pytest.mark.asyncio
async def test_pipeline_failure_stops_execution(metadata_store):
    runner = PipelineRunner(steps=[FailingStep(), EchoStep()])
    results = await runner.run(IngestionContext(store=metadata_store))
    assert len(results) == 1 and results[0].status == StepStatus.FAILURE


@pytest.mark.asyncio
async def test_embedding_agent_standalone(metadata_store):
    from omni_ingest.core.model import KnowledgeItem

    item = KnowledgeItem(content="hello world", metadata={})
    res = await EmbeddingAgent().run(IngestionContext(items=[item], store=metadata_store))
    assert res.status == StepStatus.SUCCESS and len(res.items) == 1


@pytest.mark.asyncio
async def test_full_ingestion_flow(metadata_store):
    store = InMemoryVectorStore()
    steps = [SentenceChunkingAgent(), EmbeddingAgent(), VectorIndexingAgent(store=store)]

    results = await PipelineRunner(steps=steps).run(IngestionContext(metadata={"text": "Hello world. Test."}, store=metadata_store))
    assert all(r.status == StepStatus.SUCCESS for r in results) and await store.count() >= 1


@pytest.mark.asyncio
async def test_yaml_pipeline_factory(metadata_store):
    config = {"steps": [{"agent": "chunking"}, {"agent": "embedding"}, {"agent": "indexing"}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
        yaml.dump(config, tmp)
        tmp.flush()
        results = await create_pipeline_from_config(tmp.name).run(IngestionContext(metadata={"text": "YAML test."}, store=metadata_store))
        assert all(r.status == StepStatus.SUCCESS for r in results) and len(results) == 3
