import asyncio

import pytest

from omni_ingest.agent.enrichment import TransformAgent
from omni_ingest.core import event
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, Step, StepResult, StepStatus
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
            items=[KnowledgeItem(content=ByteContent.from_bytes(f"Parallel content {idx}".encode()), source_uri=f"file://{idx}.txt")],
            store=metadata_store,
            resource=ResolvedResource(uri="", content=ByteContent.from_bytes(f"Parallel content {idx}".encode())),
        )
        runner = PipelineRunner(steps=[TransformAgent(value=".items |= map(. + {text: (.text | ascii_upcase)})", update=["items"])])
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

        from omni_ingest.core.model import PipelineStepRunRecord

        res = await s.execute(select(func.count(PipelineStepRunRecord.id)))
        count = res.scalar()
        assert count == num_runs


@pytest.mark.asyncio
async def test_steps_in_stage_run_concurrently_and_stages_wait(metadata_store):
    calls: list[str] = []
    active = {"count": 0, "max": 0}
    runner = PipelineRunner(
        steps={"s1": [RecordingStep("a", calls, active, delay=0.02), RecordingStep("b", calls, active, delay=0.01)], "s2": [RecordingStep("c", calls, active)]},
    )

    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b""))))

    assert active["max"] == 2
    assert calls.index("c:start") > calls.index("a:end")
    assert calls.index("c:start") > calls.index("b:end")
    assert [result.metadata["name"] for result in results] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_unstaged_steps_remain_sequential(metadata_store):
    calls: list[str] = []
    active = {"count": 0, "max": 0}
    runner = PipelineRunner(steps=[RecordingStep("a", calls, active, delay=0.01), RecordingStep("b", calls, active, delay=0.01)])

    await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b""))))

    assert active["max"] == 1
    assert calls == ["a:start", "a:end", "b:start", "b:end"]


@pytest.mark.asyncio
async def test_failing_stage_finishes_and_stops_later_stages(metadata_store):
    calls: list[str] = []
    runner = PipelineRunner(
        steps={"s1": [RecordingStep("slow", calls, delay=0.02), RecordingStep("fail", calls, fail=True)], "s2": [RecordingStep("later", calls)]},
    )

    results = await runner.run(IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b""))))

    assert [result.metadata["name"] for result in results] == ["slow", "fail"]
    assert "slow:end" in calls
    assert "later:start" not in calls


class ProgressingStep(Step):
    def __init__(self, ticks, delay, gate=None):
        self.ticks = ticks
        self.delay = delay
        self.gate = gate

    async def run(self, ctx):
        for i in range(1, self.ticks + 1):
            if self.gate and i == self.ticks:
                await self.gate.wait()
            ctx.progress(i, self.ticks)
            await asyncio.sleep(self.delay)
        return StepResult(status=StepStatus.SUCCESS)


@pytest.mark.asyncio
async def test_finished_run_shutdown_keeps_other_run_events(metadata_store):
    def make(ticks, delay, gate=None):
        ctx = IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")))
        return ctx, PipelineRunner([ProgressingStep(ticks, delay, gate)])

    async def listen(run_id, seen):
        async for e in event.subscribe(run_id=run_id, populate_with_buffered=False):
            seen.append(e)

    async def run_and_close(ctx, runner):
        await runner.run(ctx)
        event.shutdown(ctx.run_id)

    gate = asyncio.Event()
    (ctx_a, runner_a), (ctx_b, runner_b) = make(2, 0), make(8, 0, gate)
    seen_a, seen_b = [], []
    listeners = [asyncio.create_task(listen(ctx_a.run_id, seen_a)), asyncio.create_task(listen(ctx_b.run_id, seen_b))]
    await asyncio.sleep(0)
    task_a, task_b = asyncio.create_task(run_and_close(ctx_a, runner_a)), asyncio.create_task(run_and_close(ctx_b, runner_b))

    await task_a
    assert not task_b.done()
    gate.set()
    await asyncio.gather(task_b, *listeners)

    assert isinstance(seen_a[-1], event.PipelineEndEvent)
    assert isinstance(seen_b[-1], event.PipelineEndEvent)
    assert sum(isinstance(e, event.PipelineStepProgressEvent) for e in seen_b) == 8
    assert {e.run_id for e in seen_a} == {ctx_a.run_id}
    assert {e.run_id for e in seen_b} == {ctx_b.run_id}


@pytest.mark.asyncio
async def test_run_flattened_caps_in_flight_resources(metadata_store, tmp_path):
    for i in range(6):
        (tmp_path / f"{i}.txt").write_text("x")
    active = {"count": 0, "max": 0}
    runner = PipelineRunner(steps=[RecordingStep("s", [], active=active, delay=0.05)])
    ctx = IngestionContext(store=metadata_store, resource=str(tmp_path / "*.txt"))

    results = [r async for r in runner.run_flattened(ctx, max_concurrent=2)]

    assert len(results) == 6
    assert active["max"] == 2
