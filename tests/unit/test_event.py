import asyncio
import json
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from omni_ingest.core import event
from omni_ingest.core.config import settings
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, Step, StepResult, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner


class ProgressStep(Step):
    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        ctx.progress(1, 2, "half")
        return StepResult(status=StepStatus.SUCCESS)


@pytest.fixture(autouse=True)
def reset_event_state():
    event.shutdown()
    yield
    event.shutdown()


@dataclass(kw_only=True)
class RunEvent(event.Event, event._PipelineMixin):
    pass


RUN = uuid4()


def make_event(i: int, run_id: UUID = RUN) -> event.Event:
    return RunEvent(identifier=f"e{i}", metadata={"i": i}, run_id=run_id, tenant_id=run_id)


async def wait_for_subscribers(n: int):
    for _ in range(100):
        if len(event._subscribers) == n:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {n} subscribers, have {len(event._subscribers)}")


@pytest.mark.asyncio
async def test_concurrent_publish_subscribe():
    subscribers = [asyncio.Queue(maxsize=512) for _ in range(3)]
    for q in subscribers:
        event._subscribers.add((None, q))

    async def publisher(offset: int):
        for i in range(25):
            event.publish(make_event(offset + i))
            await asyncio.sleep(0)

    await asyncio.gather(*(publisher(n * 25) for n in range(4)))

    expected = 100
    for q in subscribers:
        assert q.qsize() == expected
        ids = [q.get_nowait().identifier for _ in range(expected)]
        assert len(ids) == expected
        assert len(set(ids)) == expected


@pytest.mark.asyncio
async def test_subscribe_cleanup_on_exception():
    stream = event.subscribe()

    async def consumer():
        async for _ in stream:
            raise RuntimeError("boom")

    task = asyncio.create_task(consumer())
    await wait_for_subscribers(1)
    event.publish(event.Event(identifier="x"))
    with pytest.raises(RuntimeError):
        await task
    await stream.aclose()
    assert not event._subscribers


@pytest.mark.asyncio
async def test_publish_drops_oldest_when_subscriber_queue_full():
    q: asyncio.Queue[event.Event | None] = asyncio.Queue(maxsize=32)
    event._subscribers.add((None, q))

    for i in range(40):
        event.publish(make_event(i))

    assert q.qsize() == 32
    ids = [q.get_nowait().identifier for _ in range(32)]
    assert ids[0] == "e8"
    assert ids[-1] == "e39"


@pytest.mark.asyncio
async def test_new_subscriber_receives_buffered_events(monkeypatch):
    monkeypatch.setattr(settings, "event_queue_size", 32)
    for i in range(50):
        event.publish(make_event(i))

    stream = event.subscribe()
    replayed = [await anext(stream) for _ in range(32)]
    await stream.aclose()

    assert [e.identifier for e in replayed] == [f"e{i}" for i in range(18, 50)]


@pytest.mark.asyncio
async def test_buffer_overflow_keeps_last_events(monkeypatch):
    monkeypatch.setattr(settings, "event_queue_size", 32)
    for i in range(50):
        event.publish(make_event(i))

    buffer = event._buffers[RUN]
    assert len(buffer) == 32
    assert buffer[0].identifier == "e18"
    assert buffer[-1].identifier == "e49"


@pytest.mark.asyncio
async def test_subscribe_stops_on_none_sentinel():
    stream = event.subscribe()
    pull = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    _, q = next(iter(event._subscribers))
    q.put_nowait(None)

    with pytest.raises(StopAsyncIteration):
        await pull

    await stream.aclose()
    assert not event._subscribers


@pytest.mark.asyncio
async def test_publish_while_iterating_subscribers():
    streams = [event.subscribe() for _ in range(3)]
    pulls = [asyncio.create_task(anext(s)) for s in streams]
    await asyncio.sleep(0)

    event.publish(make_event(1))

    values = await asyncio.gather(*pulls)
    assert [v.identifier for v in values] == ["e1", "e1", "e1"]
    for s in streams:
        await s.aclose()


@pytest.mark.asyncio
async def test_shutdown_with_active_subscribers():
    streams = [event.subscribe() for _ in range(3)]
    pulls = [asyncio.create_task(anext(s)) for s in streams]
    await wait_for_subscribers(3)

    event.shutdown()

    results = await asyncio.gather(*pulls, return_exceptions=True)
    assert all(isinstance(r, StopAsyncIteration) for r in results)


@pytest.mark.asyncio
async def test_subscribe_after_shutdown():
    event.publish(make_event(1))
    event.shutdown()

    stream = event.subscribe()
    pull = asyncio.create_task(anext(stream))
    await wait_for_subscribers(1)

    event.publish(make_event(2))
    e = await pull
    assert e.identifier == "e2"
    await stream.aclose()


@pytest.mark.asyncio
async def test_subscribe_with_async_for():
    for i in range(5):
        event.publish(make_event(i))

    received = []
    async def consume():
        async for e in event.subscribe():
            received.append(e.identifier)
            if len(received) == 7:  # 5 buffered + 2 new events
                break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    event.publish(make_event(5))
    event.publish(make_event(6))

    await task

    await asyncio.sleep(0)
    assert len(event._subscribers) == 0
    assert received == [f"e{i}" for i in range(7)]


@pytest.mark.asyncio
async def test_pipeline_step_progress_event(metadata_store):
    stream = event.subscribe(populate_with_buffered=False)
    runner = PipelineRunner([ProgressStep()])
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), store=metadata_store)
    task = asyncio.create_task(runner.run(ctx))

    async for e in stream:
        if isinstance(e, event.PipelineStepProgressEvent):
            progress = e
            break

    await task
    await stream.aclose()
    assert progress.completed == 1
    assert progress.total == 2
    assert progress.message == "half"


@pytest.mark.asyncio
async def test_replay_is_per_run():
    other = uuid4()
    event.publish(make_event(1))
    event.publish(make_event(2, other))
    event.publish(event.Event(identifier="no_run"))

    stream = event.subscribe(run_id=other)
    assert (await anext(stream)).identifier == "e2"
    await stream.aclose()

    stream = event.subscribe()
    assert [(await anext(stream)).identifier for _ in range(2)] == ["e1", "e2"]
    await stream.aclose()
    assert set(event._buffers) == {RUN, other}


@pytest.mark.asyncio
async def test_run_filter_and_run_shutdown_keep_other_subscribers():
    other = uuid4()
    a, b, everyone = event.subscribe(run_id=RUN, populate_with_buffered=False), event.subscribe(run_id=other, populate_with_buffered=False), event.subscribe(populate_with_buffered=False)
    pulls = [asyncio.create_task(anext(s)) for s in (a, b, everyone)]
    await wait_for_subscribers(3)

    event.publish(make_event(1, other))
    event.shutdown(other)
    await asyncio.sleep(0)

    assert (await pulls[1]).identifier == "e1"
    assert (await pulls[2]).identifier == "e1"
    assert not pulls[0].done()
    assert len(event._subscribers) == 2
    assert other not in event._buffers
    with pytest.raises(StopAsyncIteration):
        await anext(b)

    event.shutdown()
    with pytest.raises(StopAsyncIteration):
        await pulls[0]
    assert not event._subscribers


def test_to_dict_is_json_safe_for_every_event():
    run_id, step = uuid4(), ProgressStep()
    result = StepResult(status=StepStatus.SUCCESS, items=[KnowledgeItem(content=ByteContent.from_bytes(b"\xff\x00raw"), source_uri="a.txt")], metadata={"k": uuid4()})
    base = {"run_id": run_id, "tenant_id": run_id}
    events = [
        event.Event(identifier="custom", metadata={"id": run_id}),
        event.PipelineBeginEvent(completed_steps=[step], **base),
        event.PipelineEndEvent(status="failed", error="boom", **base),
        event.PipelineStepBeginEvent(step_name="progress", step=step, **base),
        event.PipelineStepProgressEvent(step_name="progress", step=step, completed=1, total=2, **base),
        event.PipelineStepEndEvent(step_name="progress", step=step, step_result=result, **base),
    ]
    for e in events:
        data = e.to_dict()
        assert data["identifier"] == e.identifier
        assert "step" not in data
        assert "completed_steps" not in data
        assert "raw" not in json.dumps(data)
    assert "items" not in events[-1].to_dict()["step_result"]
    assert events[-1].to_dict()["run_id"] == str(run_id)


@pytest.mark.asyncio
async def test_shutdown_keeps_queued_events_for_full_subscriber():
    q = asyncio.Queue(maxsize=1)
    first = make_event(0)
    q.put_nowait(first)
    event._subscribers.add((RUN, q))

    event.shutdown(RUN)

    assert not event._subscribers
    assert q.get_nowait() is first
    assert await asyncio.wait_for(q.get(), 1) is None


def test_buffers_are_capped_by_run_count(monkeypatch):
    monkeypatch.setattr(event, "_MAX_BUFFERED_RUNS", 2)
    runs = [uuid4() for _ in range(3)]
    for run_id in runs:
        event.publish(make_event(0, run_id))
    event.publish(make_event(1, runs[1]))
    event.publish(make_event(0, uuid4()))

    assert len(event._buffers) == 2
    assert runs[0] not in event._buffers
    assert runs[1] not in event._buffers or len(event._buffers[runs[1]]) == 2
