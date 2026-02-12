import asyncio
import pytest

from omni_ingest.core import event


@pytest.fixture(autouse=True)
def reset_event_state():
    event.shutdown()
    yield
    event.shutdown()


def make_event(i: int) -> event.Event:
    return event.Event(identifier=f"e{i}", metadata={"i": i})


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
        event._subscribers.add(q)

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
    event._subscribers.add(q)

    for i in range(40):
        event.publish(make_event(i))

    assert q.qsize() == 32
    ids = [q.get_nowait().identifier for _ in range(32)]
    assert ids[0] == "e8"
    assert ids[-1] == "e39"


@pytest.mark.asyncio
async def test_new_subscriber_receives_buffered_events():
    for i in range(50):
        event.publish(make_event(i))

    stream = event.subscribe()
    replayed = [await anext(stream) for _ in range(32)]
    await stream.aclose()

    assert [e.identifier for e in replayed] == [f"e{i}" for i in range(18, 50)]


@pytest.mark.asyncio
async def test_buffer_overflow_keeps_last_32_events():
    for i in range(50):
        event.publish(make_event(i))

    assert len(event._buffer) == 32
    assert event._buffer[0].identifier == "e18"
    assert event._buffer[-1].identifier == "e49"


@pytest.mark.asyncio
async def test_subscribe_stops_on_none_sentinel():
    stream = event.subscribe()
    pull = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    q = next(iter(event._subscribers))
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
