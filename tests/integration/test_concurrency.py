import asyncio

import pytest

from omni_ingest.agent.chunking import SentenceChunkingAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, Step, StepResult, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner


class RecordingStep(Step):
    def __init__(self, name, calls, active=None, fail=False, delay=0):
        self.name = name
        self.calls = calls
        self.active = active
        self.fail = fail
        self.delay = delay

    async def run(self, ctx):
        self.calls.append(f"{self.name}:start")
        if self.active is not None:
            self.active["count"] += 1
            self.active["max"] = max(self.active["max"], self.active["count"])
        await asyncio.sleep(self.delay)
        if self.active is not None:
            self.active["count"] -= 1
        self.calls.append(f"{self.name}:end")
        return StepResult(status=StepStatus.FAILURE if self.fail else StepStatus.SUCCESS, error="Boom" if self.fail else None, metadata={"name": self.name})


@pytest.mark.asyncio
async def test_parallel_pipeline_execution(metadata_store, tenant):
    num_runs = 5

    async def run_pipeline(idx):
        ctx = IngestionContext(
            tenant_id=tenant.id,
            items=[KnowledgeItem(content=f"Parallel content {idx}".encode(), source_uri=f"file://{idx}.txt")],
            store=metadata_store,
            resource=ResolvedResource(uri="", content=f"Parallel content {idx}".encode()),
        )
        runner = PipelineRunner(steps=[SentenceChunkingAgent(chunk_size=10, overlap=0)])
        results = await runner.run(ctx)
        return results

    tasks = [run_pipeline(i) for i in range(num_runs)]
    all_results = await asyncio.gather(*tasks)

    assert len(all_results) == num_runs
    for results in all_results:
        assert results[0].status == StepStatus.SUCCESS
        assert len(results[0].items) > 0
    async with metadata_store.session_factory() as s:
        from sqlalchemy import func, select

        from omni_ingest.port.metadata_store import PipelineStepRun

        res = await s.execute(select(func.count(PipelineStepRun.id)))
        count = res.scalar()
        assert count == num_runs


@pytest.mark.asyncio
async def test_steps_in_stage_run_concurrently_and_stages_wait(metadata_store):
    calls: list[str] = []
    active = {"count": 0, "max": 0}
    runner = PipelineRunner(
        steps={"s1": [RecordingStep("a", calls, active, delay=0.02), RecordingStep("b", calls, active, delay=0.01)], "s2": [RecordingStep("c", calls, active)]},
    )

    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=b"")))

    assert active["max"] == 2
    assert calls.index("c:start") > calls.index("a:end")
    assert calls.index("c:start") > calls.index("b:end")
    assert [result.metadata["name"] for result in results] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_unstaged_steps_remain_sequential(metadata_store):
    calls: list[str] = []
    active = {"count": 0, "max": 0}
    runner = PipelineRunner(steps=[RecordingStep("a", calls, active, delay=0.01), RecordingStep("b", calls, active, delay=0.01)])

    await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=b"")))

    assert active["max"] == 1
    assert calls == ["a:start", "a:end", "b:start", "b:end"]


@pytest.mark.asyncio
async def test_failing_stage_finishes_and_stops_later_stages(metadata_store):
    calls: list[str] = []
    runner = PipelineRunner(
        steps={"s1": [RecordingStep("slow", calls, delay=0.02), RecordingStep("fail", calls, fail=True)], "s2": [RecordingStep("later", calls)]},
    )

    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=b"")))

    assert [result.metadata["name"] for result in results] == ["slow", "fail"]
    assert "slow:end" in calls
    assert "later:start" not in calls
