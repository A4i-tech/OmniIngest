from __future__ import annotations

import asyncio
import collections
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic_core import to_jsonable_python

from .config import settings
from .model import Step, StepResult

_OMIT = {"step", "completed_steps", "step_result"}


@dataclass(kw_only=True)
class Event:
    """Base class for all events."""

    identifier: str
    """Unique identifier for the event"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Event creation timestamp"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional event metadata"""

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-safe fields. Step instances are omitted, `step_name` identifies the step."""
        return cast("dict[str, Any]", to_jsonable_python({f.name: getattr(self, f.name) for f in fields(self) if f.name not in _OMIT}))


@dataclass(kw_only=True)
class _PipelineMixin:
    run_id: uuid.UUID
    """ID of the pipeline run"""

    tenant_id: uuid.UUID
    """ID of the tenant owning the pipeline"""


@dataclass(kw_only=True)
class _StepMixin:
    step_name: str
    """Name of the pipeline step"""

    step: Step
    """Reference to the step instance"""


@dataclass(kw_only=True)
class PipelineBeginEvent(Event, _PipelineMixin):
    """Event emitted at the start of a pipeline."""

    identifier: str = field(default="pipeline_begin", init=False)
    completed_steps: list[Step] = field(default_factory=list)


@dataclass(kw_only=True)
class PipelineEndEvent(Event, _PipelineMixin):
    """Event emitted at the end of a pipeline."""

    identifier: str = field(default="pipeline_end", init=False)

    status: Literal["succeeded", "failed"]
    """Outcome of the pipeline"""

    error: str | None
    """Human-readable error message if the pipeline failed"""


@dataclass(kw_only=True)
class PipelineStepBeginEvent(Event, _PipelineMixin, _StepMixin):
    """Event emitted at the start of a pipeline step."""

    identifier: str = field(default="pipeline_step_begin", init=False)


@dataclass(kw_only=True)
class PipelineStepProgressEvent(Event, _PipelineMixin, _StepMixin):
    identifier: str = field(default="pipeline_step_progress", init=False)
    completed: float
    total: float
    message: str | None = None


@dataclass(kw_only=True)
class PipelineStepEndEvent(Event, _PipelineMixin, _StepMixin):
    """Event emitted at the end of a pipeline step."""

    identifier: str = field(default="pipeline_step_end", init=False)

    step_result: StepResult
    """Result of the step execution"""

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {"step_result": self.step_result.model_dump(mode="json", exclude={"items"})}


_MAX_BUFFERED_RUNS = 128
_buffers: dict[uuid.UUID, collections.deque[Event]] = {}
_closers: set[asyncio.Task] = set()
_subscribers: set[tuple[uuid.UUID | None, asyncio.Queue[Event | None]]] = set()


def _put(q: asyncio.Queue[Event | None], item: Event | None):
    if q.full():
        q.get_nowait()
    q.put_nowait(item)


def publish(event: Event):
    """Publishes an event to all matching subscribers and buffers it under its run."""
    run_id = getattr(event, "run_id", None)
    if run_id is not None:
        buf = _buffers.pop(run_id, None) or collections.deque(maxlen=settings.event_queue_size)
        _buffers[run_id] = buf
        buf.append(event)
        while len(_buffers) > _MAX_BUFFERED_RUNS:
            del _buffers[next(iter(_buffers))]
    for sub_run_id, q in _subscribers:
        if sub_run_id is not None and sub_run_id != run_id:
            continue
        _put(q, event)


async def subscribe(run_id: uuid.UUID | None = None, populate_with_buffered: bool = True) -> AsyncIterator[Event]:
    """
    Subscribes to the event stream, yielding events as they are published.

    Args:
        run_id: Only yield events of this pipeline run. None yields events of all runs.
        populate_with_buffered: Whether to initialize this subscriber with buffered events.

    Example:
        ```python
        from omni_ingest.core.event import subscribe

        async for event in subscribe():
            print(event)
        ```
    """
    q: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=settings.event_queue_size)
    if populate_with_buffered:
        buffered = [e for b in (_buffers.values() if run_id is None else [_buffers.get(run_id, ())]) for e in b]
        for buffered_event in sorted(buffered, key=lambda e: e.timestamp)[-q.maxsize :]:
            q.put_nowait(buffered_event)

    sub = (run_id, q)
    _subscribers.add(sub)
    try:
        while True:
            e = await q.get()
            if not e:
                break
            yield e
    finally:
        _subscribers.discard(sub)


def shutdown(run_id: uuid.UUID | None = None):
    """Shuts down the event system, notifying subscribers. With a run ID, only that run's subscribers end and its buffer clears."""
    for sub in [s for s in _subscribers if run_id is None or s[0] == run_id]:
        _subscribers.discard(sub)
        try:
            sub[1].put_nowait(None)
        except asyncio.QueueFull:
            closer = asyncio.get_running_loop().create_task(sub[1].put(None))
            _closers.add(closer)
            closer.add_done_callback(_closers.discard)
    if run_id is None:
        _buffers.clear()
    else:
        _buffers.pop(run_id, None)
