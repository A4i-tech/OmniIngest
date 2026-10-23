# Working with event stream

Events are how OmniIngest reports pipeline activity while work is running. CLI progress bars use them. Tests use them. Any app embedding OmniIngest can use them to show live status without poking inside agent code.

Event system is deliberately small. Agents do not know about queues or UI. They call `ctx.progress(...)` when they have useful progress. Pipeline runner publishes lifecycle events as it starts and finishes pipeline stages.

!!! warning
    Events are live status hints. They are not checkpoints or audit records. Checkpoints live in metadata store. Queue overflow can drop old events. Event objects may include Python step instances. Use `to_dict()` to serialize them.

## What gets published

A normal run emits this shape:

```text
pipeline_begin
pipeline_step_begin
pipeline_step_progress
pipeline_step_end
pipeline_step_begin
pipeline_step_end
pipeline_end
```

Not every step emits progress. Simple steps may only emit begin and end. Longer agents such as page chunking, rasterising, image extraction, stitching, and extraction call `ctx.progress(...)`.

Every event has:

- `identifier`: stable event name.
- `timestamp`: UTC creation time.
- `metadata`: extra event metadata.

Pipeline events also carry:

- `run_id`: pipeline run id.
- `tenant_id`: tenant id for run.

Step events also carry:

- `step_name`: registry name such as `sentence_chunking` or `extract`.
- `step`: actual step instance.

!!! note
    This is a Python event stream. OmniIngest does not ship an HTTP or WebSocket server. The host application serves events to its clients. Call `Event.to_dict()` to get JSON-safe data.

## Serializing events

Call `to_dict()` on any event to get a dictionary that `json.dumps` accepts.

```python
payload = json.dumps(e.to_dict())
```

The dictionary has all event fields, including `identifier`. These rules apply:

- UUID values become strings.
- `timestamp` becomes an ISO 8601 string.
- `step_result` becomes a dictionary without `items`, so no binary content is included.
- `step` and `completed_steps` are omitted. Use `step_name` to identify the step.

## Event types

### `PipelineBeginEvent`

Published when a run starts.

```python
event.PipelineBeginEvent(
    run_id=ctx.run_id,
    tenant_id=ctx.tenant_id,
    completed_steps=[]
)
```

`completed_steps` is usually empty. During resume it contains steps already covered by checkpoint. CLI uses this to mark completed tasks green before resumed work starts.

### `PipelineStepBeginEvent`

Published just before a step runs.

```python
event.PipelineStepBeginEvent(
    run_id=ctx.run_id,
    tenant_id=ctx.tenant_id,
    step_name="extract",
    step=extract_step
)
```

Use this to start timers, mark a UI row as running, or attach step-local logs.

### `PipelineStepProgressEvent`

Published when a step calls `ctx.progress(completed, total, message)`.

```python
event.PipelineStepProgressEvent(
    run_id=ctx.run_id,
    tenant_id=ctx.tenant_id,
    step_name="page_chunking",
    step=page_chunking_step,
    completed=12,
    total=40,
    message="page 12/40"
)
```

`completed` and `total` are floats so callers can report units that are not whole items. In practice most agents use counts.

Progress is attached to current step through a context variable set by pipeline runner. If `ctx.progress(...)` is called outside a running step, nothing is published. That keeps helper code safe when reused outside pipelines.

### `PipelineStepEndEvent`

Published after a step returns or fails.

```python
event.PipelineStepEndEvent(
    run_id=ctx.run_id,
    tenant_id=ctx.tenant_id,
    step_name="sentence_chunking",
    step=chunking_step,
    step_result=result
)
```

`step_result` contains `status`, `metadata`, `items`, `output_paths`, and `error`.

If a step raises, runner catches exception and turns it into `StepResult(status=FAILURE, error=str(exc))`. Event still gets published.

### `PipelineEndEvent`

Published once run is finished.

```python
event.PipelineEndEvent(
    run_id=ctx.run_id,
    tenant_id=ctx.tenant_id,
    status="succeeded",
    error=None
)
```

Status is `succeeded` or `failed`. On failure, `error` is copied from first failed step result.

## Subscribing

Use `event.subscribe()` as an async iterator. By default it yields events of all runs. Pass `run_id` to receive only the events of one run.

```python
import asyncio

from omni_ingest.core import event


async def watch():
    async for e in event.subscribe():
        print(e.identifier, e.timestamp)


asyncio.run(watch())
```

Most apps should run subscriber as a background task before starting pipeline.

```python
import asyncio
from pathlib import Path

from omni_ingest.core import event
from omni_ingest.core.model import ByteContent, ResolvedResource
from omni_ingest.core.pipeline import IngestionContext, create_pipeline_from_config


async def watch(run_id):
    async for e in event.subscribe(run_id=run_id, populate_with_buffered=False):
        if isinstance(e, event.PipelineStepProgressEvent):
            print(f"{e.step_name}: {e.completed}/{e.total} {e.message or ''}")
        elif isinstance(e, event.PipelineEndEvent):
            break


async def main():
    pipeline = create_pipeline_from_config(Path("school_notes.yaml"))
    ctx = IngestionContext(resource=ResolvedResource(uri="notes.txt", content=ByteContent.from_bytes(b"First sentence. Second sentence.")))
    listener = asyncio.create_task(watch(ctx.run_id))
    try:
        await pipeline.run(ctx)
    finally:
        event.shutdown(ctx.run_id)
        await listener


asyncio.run(main())
```

!!! tip
    Use `run_id` and `populate_with_buffered=False` for one live job. Subscriber receives only future events of that run. CLI uses this because events from another run would make progress bars look odd.

## Buffered replay

Event module keeps the latest events of each run in memory. The `event_queue_size` setting controls how many events it keeps per run. The default is 256. Events without a `run_id` are not buffered. New subscribers receive buffered events by default.

```python
async for e in event.subscribe():
    ...
```

A subscriber without `run_id` receives the buffered events of all runs. A subscriber with `run_id` receives only the buffered events of that run.

This is handy for late subscribers in tests or lightweight dashboards. It is not durable storage. Restart process and buffer is gone.

Set replay off when listener has one live job to follow:

```python
async for e in event.subscribe(run_id=ctx.run_id, populate_with_buffered=False):
    ...
```

!!! warning
    Each subscriber has a queue of `event_queue_size` events. If queue is full, oldest queued event is dropped before new event is added. This favours live progress over perfect history. Slow dashboards should treat event stream as status hints, not an audit log.

## Settings

`event_queue_size` sets the replay buffer size of each run and the queue size of each subscriber. Set it with the `EVENT_QUEUE_SIZE` environment variable or in `.env`. OmniIngest reads settings once per process.

## Publishing custom events

You can publish custom events with base `Event` if you need app-specific signals.

```python
from omni_ingest.core import event

event.publish(event.Event(
    identifier="dashboard_note",
    metadata={"message": "manual review requested"}
))
```

!!! tip
    Use custom events sparingly. For step progress prefer `ctx.progress(...)` because runner fills in run id, tenant id, step name, and step instance.

## Progress from an agent

Agent code should call `ctx.progress(...)`.

```python
from pydantic import BaseModel

from omni_ingest.core.model import Step, StepResult, StepStatus
from omni_ingest.core.pipeline import IngestionContext


class ImportRowsAgent(BaseModel, Step):
    async def run(self, ctx: IngestionContext) -> StepResult:
        rows = ctx.metadata.get("rows", [])
        ctx.progress(0, len(rows), "importing rows")
        for i, row in enumerate(rows, 1):
            # Do real row work here.
            ctx.progress(i, len(rows), f"row {i}/{len(rows)}")
        return StepResult(status=StepStatus.SUCCESS)
```

Keep progress messages short. CLI renders them inline beside step label.

## Shutdown

Call `event.shutdown(run_id)` when a run is finished.

```python
event.shutdown(ctx.run_id)
```

This sends a sentinel to the subscribers of that run and removes them. It also deletes the replay buffer of that run. Subscribers of other runs keep receiving events. Subscribers without `run_id` also keep receiving events.

Call `event.shutdown()` without argument to end all subscribers and clear all buffers. Tests use it before and after each event test so runs do not leak events into each other.

!!! warning
    Event system keeps the buffers of the 128 most recent runs. Older buffers are dropped, so a caller that never calls `event.shutdown(run_id)` loses old progress history but does not leak memory. Call it after each run to free a buffer early. `omni_ingest.run` does this for you.

## Common patterns

Print failed step:

```python
async for e in event.subscribe(run_id=run_id, populate_with_buffered=False):
    if isinstance(e, event.PipelineStepEndEvent) and e.step_result.status.value == "failure":
        print(f"{e.step_name} failed: {e.step_result.error}")
```

Show one-line progress:

```python
async for e in event.subscribe(run_id=run_id, populate_with_buffered=False):
    if isinstance(e, event.PipelineStepProgressEvent):
        pct = 0 if not e.total else round(e.completed / e.total * 100)
        print(f"{e.step_name}: {pct}% {e.message or ''}")
```

Stop when pipeline ends:

```python
async for e in event.subscribe(run_id=run_id, populate_with_buffered=False):
    if isinstance(e, event.PipelineEndEvent):
        print(e.status)
        break
```

These examples mirror CLI strategy. Subscribe to the run first, start run, consume events until pipeline ends, then call `event.shutdown(run_id)` and wait for the listener.
