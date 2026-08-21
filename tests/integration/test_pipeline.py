import tempfile
from pathlib import Path

import pytest

from omni_ingest.core.model import ResolvedResource, Step, StepResult, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner


class EchoStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        return StepResult(status=StepStatus.SUCCESS, metadata={"echo": "echo"})


class FailingStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        return StepResult(status=StepStatus.FAILURE, error="Boom")


class SaveStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        ctx.metadata["saved"] = ctx.metadata.get("saved", 0) + 1
        return StepResult(status=StepStatus.SUCCESS)


class MutatingFailingStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        ctx.metadata["failed_stage_mutation"] = True
        return StepResult(status=StepStatus.FAILURE, error="Boom")


class FixedStep(Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        ctx.metadata["fixed"] = True
        return StepResult(status=StepStatus.SUCCESS)


@pytest.mark.asyncio
async def test_pipeline_execution_basic(metadata_store):
    runner = PipelineRunner(steps=[EchoStep()])
    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", raw_content=b"")))
    assert results[0].status == StepStatus.SUCCESS and results[0].metadata["echo"] == "echo"


@pytest.mark.asyncio
async def test_pipeline_failure_stops_execution(metadata_store):
    runner = PipelineRunner(steps=[FailingStep(), EchoStep()])
    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", raw_content=b"")))
    assert len(results) == 1 and results[0].status == StepStatus.FAILURE


@pytest.mark.asyncio
async def test_pipeline_resume_restores_last_successful_stage(metadata_store):
    ctx = IngestionContext(store=metadata_store, resource=ResolvedResource(uri="file://input.pdf", raw_content=b"%PDF"))
    results = await PipelineRunner([SaveStep(), MutatingFailingStep()]).run(ctx)
    assert results[-1].status == StepStatus.FAILURE
    checkpoint = await metadata_store.get_pipeline_checkpoint(ctx.run_id)
    assert checkpoint is not None
    assert checkpoint.stage == 1
    assert checkpoint.metadata == {"saved": 1}

    restored, resumed_results = await PipelineRunner([SaveStep(), FixedStep()]).resume(ctx.run_id, metadata_store)
    assert all(result.status == StepStatus.SUCCESS for result in await resumed_results)
    assert restored.metadata == {"saved": 1, "fixed": True}
    assert await metadata_store.get_pipeline_checkpoint(ctx.run_id) is None


@pytest.mark.asyncio
async def test_successful_pipeline_deletes_checkpoint(metadata_store):
    ctx = IngestionContext(store=metadata_store, resource=ResolvedResource(uri="file://input.pdf", raw_content=b"%PDF"))
    await PipelineRunner([SaveStep()]).run(ctx)
    assert await metadata_store.get_pipeline_checkpoint(ctx.run_id) is None


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
