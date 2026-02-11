import asyncio
import collections
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Base class for all events."""

    identifier: str = Field(..., description="Unique identifier for the event")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Event creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


class PipelineStepEvent(Event):
    """Base class for events related to pipeline steps."""

    run_id: uuid.UUID = Field(..., description="ID of the pipeline run")
    step_name: str = Field(..., description="Name of the pipeline step")
    step: Any = Field(..., description="Reference to the step instance")


class PipelineStepEndEvent(PipelineStepEvent):
    """Event emitted at the end of a pipeline step."""

    identifier: str = "pipeline_step_end"
    step_result: Any = Field(..., description="Result of the step execution")


buffer = collections.deque[Event](maxlen=32)
subscribers: set[asyncio.Queue[Event]] = set()


def publish(event: Event):
    """Publishes an event to all subscribers."""
    buffer.append(event)

    dead = []
    for q in subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)

    for q in dead:
        subscribers.discard(q)
