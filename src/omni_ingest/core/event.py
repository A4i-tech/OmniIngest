import asyncio
import collections
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Base class for all events."""

    identifier: str = Field(..., description="Unique identifier for the event")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Event creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


class _PipelineMixin:
    run_id: uuid.UUID = Field(..., description="ID of the pipeline run")
    tenant_id: uuid.UUID = Field(..., description="ID of the tenant owning the pipeline")


class _StepMixin:
    step_name: str = Field(..., description="Name of the pipeline step")
    step: Any = Field(..., description="Reference to the step instance")


class PipelineBeginEvent(Event, _PipelineMixin):
    """Event emitted at the start of a pipeline."""
    identifier: str = "pipeline_begin"


class PipelineEndEvent(Event, _PipelineMixin):
    """Event emitted at the end of a pipeline."""
    identifier: str = "pipeline_end"
    status: Literal["succeeded", "failed"] = Field(..., description="Outcome of the pipeline")
    error: str | None = Field(..., description="Human-readable error message if the pipeline failed")


class PipelineStepBeginEvent(Event, _PipelineMixin, _StepMixin):
    """Event emitted at the start of a pipeline step."""
    identifier: str = "pipeline_step_begin"


class PipelineStepEndEvent(Event, _PipelineMixin, _StepMixin):
    """Event emitted at the end of a pipeline step."""
    identifier: str = "pipeline_step_end"
    step_result: Any = Field(..., description="Result of the step execution")


_buffer = collections.deque[Event](maxlen=32)
_subscribers: set[asyncio.Queue[Event | None]] = set()


def publish(event: Event):
    """Publishes an event to all subscribers."""
    _buffer.append(event)
    for q in _subscribers:
        if q.full():
            q.get_nowait()
        q.put_nowait(event)


async def subscribe(populate_with_buffered: bool = True) -> AsyncIterator[Event]:
    """
    Subscribes to the event stream, yielding events as they are published.

    Args:
        populate_with_buffered: Whether to initialize this subscriber with buffered events.

    Example:
        ```python
        from omni_ingest.core.event import subscribe

        async for event in subscribe():
            print(event)
        ```
    """
    q: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=32)
    if populate_with_buffered:
        for event in _buffer:
            q.put_nowait(event)

    _subscribers.add(q)
    try:
        while True:
            e = await q.get()
            if not e:
                break
            yield e
    finally:
        _subscribers.discard(q)


def shutdown():
    """Shuts down the event system, notifying all subscribers."""
    for q in _subscribers:
        q.put_nowait(None)
    _subscribers.clear()
    _buffer.clear()
