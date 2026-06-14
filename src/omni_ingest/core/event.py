from __future__ import annotations

import asyncio
import collections
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from .model import Step, StepResult


@dataclass(kw_only=True)
class Event:
    """Base class for all events."""

    identifier: str
    """Unique identifier for the event"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Event creation timestamp"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional event metadata"""


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
class PipelineStepEndEvent(Event, _PipelineMixin, _StepMixin):
    """Event emitted at the end of a pipeline step."""
    identifier: str = field(default="pipeline_step_end", init=False)

    step_result: StepResult
    """Result of the step execution"""


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
