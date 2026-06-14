import tempfile
from pathlib import Path

import pytest
import yaml

from omni_ingest.agent.chunking import SentenceChunkingAgent
from omni_ingest.agent.indexing import EmbeddingAgent
from omni_ingest.core.model import ResolvedResource, Step, StepResult, StepStatus
from omni_ingest.core.output import VectorStoreOutput
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner, create_pipeline_from_config


class EchoStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        return StepResult(status=StepStatus.SUCCESS, metadata={"echo": "echo"})


class FailingStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        return StepResult(status=StepStatus.FAILURE, error="Boom")


@pytest.mark.asyncio
async def test_pipeline_execution_basic(metadata_store):
    runner = PipelineRunner(steps=[EchoStep()])
    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=b"")))
    assert results[0].status == StepStatus.SUCCESS and results[0].metadata["echo"] == "echo"


@pytest.mark.asyncio
async def test_pipeline_failure_stops_execution(metadata_store):
    runner = PipelineRunner(steps=[FailingStep(), EchoStep()])
    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=b"")))
    assert len(results) == 1 and results[0].status == StepStatus.FAILURE


@pytest.mark.asyncio
async def test_embedding_agent_standalone(metadata_store):
    from omni_ingest.core.model import KnowledgeItem

    item = KnowledgeItem(content=b"hello world")
    res = await EmbeddingAgent().run(IngestionContext(items=[item], store=metadata_store, resource=ResolvedResource(uri="", content=b"")))
    assert res.status == StepStatus.SUCCESS and len(res.items) == 1


@pytest.mark.asyncio
async def test_full_ingestion_flow(metadata_store, vector_store):
    steps = [SentenceChunkingAgent(), EmbeddingAgent()]

    runner = PipelineRunner(steps=steps)
    results = await runner.run(IngestionContext(resource=ResolvedResource(uri="", content=b"Hello world. Test."), store=metadata_store), output=VectorStoreOutput(store=vector_store))
    assert all(r.status == StepStatus.SUCCESS for r in results) and await vector_store.count() >= 1
    assert runner.output_result is not None and runner.output_result.status == StepStatus.SUCCESS


@pytest.mark.asyncio
async def test_yaml_pipeline_factory(metadata_store):
    config = {"steps": [{"agent": "chunking"}, {"agent": "embedding"}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
        yaml.dump(config, tmp)
        tmp.flush()
        results = await create_pipeline_from_config(Path(tmp.name)).run(IngestionContext(resource=ResolvedResource(uri="", content=b"YAML test."), store=metadata_store))
        assert all(r.status == StepStatus.SUCCESS for r in results) and len(results) == 2


@pytest.mark.asyncio
async def test_flatten_reads_multiple_files_in_directory(metadata_store):
    with tempfile.TemporaryDirectory() as tmp_dir:
        files = ["a.txt", "b.txt", "c.txt"]
        for name in files:
            Path(tmp_dir, name).write_text("noop")

        ctx = IngestionContext(store=metadata_store, resource=tmp_dir + "/*.txt")
        runner = PipelineRunner(steps=[EchoStep()])
        all_results = [r async for r in runner.run_flattened(ctx)]

        assert len(all_results) == len(files)
        assert all(s.status == StepStatus.SUCCESS for _, steps in all_results for s in steps)


@pytest.mark.asyncio
async def test_runner_fails_on_directories(metadata_store):
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = IngestionContext(store=metadata_store, resource=tmp_dir)
        runner = PipelineRunner(steps=[])
        with pytest.RaisesGroup(IsADirectoryError):
            [r async for r in runner.run_flattened(ctx)]
