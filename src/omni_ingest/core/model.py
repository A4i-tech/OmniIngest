"""Pydantic data models for knowledge items, contexts, and results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, TypeVar, final, overload
from uuid import UUID, uuid4

import jq
import magic
from fastmcp.utilities.mcp_config import MCPServerTypes
from pydantic import AfterValidator, AnyUrl, BaseModel, ConfigDict, Field, StrictBytes
from pydantic_ai import Agent, AgentCapability
from pydantic_ai.models import KnownModelName
from pydantic_ai.output import OutputSpec
from sqlalchemy import JSON, Column, LargeBinary
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

from .config import settings

if TYPE_CHECKING:
    from .pipeline import IngestionContext


DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


def _validate_jq_expression(value: str) -> str:
    jq.compile(value, args={"output": None, "item": None, "ids": None})
    return value


JqExpression: TypeAlias = Annotated[str, AfterValidator(_validate_jq_expression)]


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ByteContent(BaseModel):
    raw_content: StrictBytes = Field(default=b"")
    content_uri: str | None = Field(default=None)
    content_encoding: str = Field(default="utf-8")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_bytes(content: StrictBytes, content_type: str | None = None, content_encoding: str = "utf-8") -> ByteContent:
        return ByteContent(raw_content=content, content_encoding=content_encoding, metadata={"content_type": content_type})

    @staticmethod
    def from_uri(uri: str, content_type: str | None = None, content_encoding: str = "utf-8") -> ByteContent:
        return ByteContent(content_uri=uri, content_encoding=content_encoding, metadata={"content_type": content_type})

    async def content(self, ctx: IngestionContext[ResolvedResource]) -> bytes:
        if self.content_uri is None:
            return self.raw_content
        return await ctx.content_resolver.resolve(self.content_uri, ctx)

    async def content_type(self, ctx: IngestionContext[ResolvedResource]) -> str:
        if isinstance(content_type := self.metadata.get("content_type"), str):
            return content_type
        return magic.from_buffer(await self.content(ctx), mime=True)

    async def decode(self, ctx: IngestionContext[ResolvedResource]) -> str:
        content_type = await self.content_type(ctx)
        if not content_type.startswith("text/"):
            raise ValueError(f"Expected text content, got {content_type}")
        return (await self.content(ctx)).decode(self.content_encoding)


class ResolvedResource(ByteContent):
    uri: str = Field(...)

Resource: TypeAlias = ResolvedResource | AnyUrl | Path


class KnowledgeItem(ByteContent):
    """
    Represents a discrete unit of information within the framework.

    A KnowledgeItem serves as a container for processed data, such as
    text segments, audio transcripts, or structured metadata. It is the primary
    entity processed and transformed throughout the ingestion pipeline.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    raw_content: StrictBytes = Field(default=b"", alias="content")
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the knowledge item")
    tenant_id: UUID | None = Field(default=None, description="Identifier for the associated tenant or organization")
    source_uri: str | None = Field(default=None, description="Origin URI of the raw data")
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata_", description="Arbitrary key-value pairs for additional contextual information")


class Tenant(SQLModel, table=True):
    __tablename__ = "tenants"  # pyright: ignore[reportAssignmentType]
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    slug: str
    display_name: str
    created_at: datetime = SQLField(default_factory=datetime.now)


class Pipeline(SQLModel, table=True):
    __tablename__ = "pipelines"  # pyright: ignore[reportAssignmentType]
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    name: str
    domain_profile: str
    created_at: datetime = SQLField(default_factory=datetime.now)


class PipelineVersion(SQLModel, table=True):
    __tablename__ = "pipeline_versions"  # pyright: ignore[reportAssignmentType]
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    pipeline_id: UUID = SQLField(foreign_key="pipelines.id")
    version: str
    yaml_profile_name: str
    is_default: bool = False
    created_at: datetime = SQLField(default_factory=datetime.now)


class PipelineRunRecord(SQLModel, table=True):
    __tablename__ = "pipeline_runs"  # pyright: ignore[reportAssignmentType]
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = SQLField(foreign_key="tenants.id")
    pipeline_version_id: UUID = SQLField(foreign_key="pipeline_versions.id")
    source_uri: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class PipelineCheckpointRecord(SQLModel, table=True):
    __tablename__ = "pipeline_checkpoints"  # pyright: ignore[reportAssignmentType]
    pipeline_run_id: UUID = SQLField(foreign_key="pipeline_runs.id", primary_key=True)
    checkpoint: bytes = SQLField(sa_column=Column(LargeBinary, nullable=False))


class PipelineStepRunRecord(SQLModel, table=True):
    __tablename__ = "pipeline_step_runs"  # pyright: ignore[reportAssignmentType]
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    pipeline_run_id: UUID = SQLField(foreign_key="pipeline_runs.id")
    step_name: str
    status: str
    output_paths: dict[str, Any] | None = SQLField(default=None, sa_column=Column(JSON))
    metadata_: dict[str, Any] | None = SQLField(default=None, sa_column=Column("metadata", JSON))
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class KnowledgeItemRecord(SQLModel, table=True):
    __tablename__ = "knowledge_items"  # pyright: ignore[reportAssignmentType]
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = SQLField(foreign_key="tenants.id")
    domain_profile: str
    modality: str
    source_uri: str
    content: bytes = SQLField(default=b"", sa_column=Column(LargeBinary, nullable=False))
    embeddings: list[dict[str, Any]] | None = SQLField(default=None, sa_column=Column(JSON))
    pipeline_version_id: UUID | None = SQLField(default=None, foreign_key="pipeline_versions.id")
    title: str | None = None
    description: str | None = None
    structured_views: dict[str, Any] | None = SQLField(default=None, sa_column=Column(JSON))
    lineage: dict[str, Any] | None = SQLField(default=None, sa_column=Column(JSON))
    metadata_: dict[str, Any] = SQLField(default_factory=dict, sa_column=Column("metadata", JSON))
    created_at: datetime = SQLField(default_factory=datetime.now)
    updated_at: datetime = SQLField(default_factory=datetime.now)


class PipelineCheckpoint(BaseModel):
    """State restored when a pipeline run resumes."""

    stage: int = Field(description="Number of completed stages")
    resource: ResolvedResource = Field(description="Resolved input resource")
    tenant_id: UUID = Field(description="Tenant that owns the pipeline run")
    domain_profile: str = Field(description="Domain profile used by the pipeline")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Shared pipeline metadata")
    items: list[KnowledgeItem] = Field(default_factory=list, description="Knowledge items produced by completed stages")


class StepResult(BaseModel):
    """Result of an individual agentic step."""

    status: StepStatus = Field(..., description="The completion status of the step")
    output_paths: dict[str, str] = Field(default_factory=dict, description="Paths to any artifacts generated by the step")
    error: str | None = Field(default=None, description="Error message if the step failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Step-specific execution metadata")
    items: list[KnowledgeItem] = Field(default_factory=list, description="Knowledge items produced or modified by the step")


class Step(ABC):
    @abstractmethod
    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        """Executes the specific logic for this ingestion step."""
        ...


class AgentMixin:
    model: KnownModelName | str | None = Field(default=None, description="Model to use for this step")
    toolsets: list[str] = Field(default_factory=list, description="Toolsets to enable for the agent")
    retries: int | None = Field(default=None, description="Per-category retry budgets for tools and output validation")

    @overload
    def _agent(self, ctx: IngestionContext[Any], deps_type: type[object] = object, output_type: OutputSpec[str] = str, capabilities: Sequence[AgentCapability[object]] | None = None) -> Agent[object, str]: ...

    @overload
    def _agent(self, ctx: IngestionContext[Any], deps_type: type[DepsT], output_type: OutputSpec[OutputT], capabilities: Sequence[AgentCapability[DepsT]] | None = None) -> Agent[DepsT, OutputT]: ...

    @final
    def _agent(self, ctx: IngestionContext[Any], deps_type: type[Any] = object, output_type: OutputSpec[Any] = str, capabilities: Sequence[AgentCapability[Any]] | None = None) -> Agent[Any, Any]:
        from ..core.pipeline import PIPELINE_RUNNERS
        tool_vendors = PIPELINE_RUNNERS[ctx.run_id].toolsets
        missing = [ts for ts in self.toolsets if ts not in tool_vendors]
        if missing:
            raise ValueError(f"Toolsets not configured in context: {missing}")
        model = self.model or settings.default_chat_completion_model
        toolsets = [tool_vendors[k] for k in self.toolsets]
        return Agent(model=model, toolsets=toolsets, retries=self.retries, deps_type=deps_type, output_type=output_type, capabilities=capabilities)


ValidationMode: TypeAlias = Literal["condition", "before", "plain", "after"]


class ValidationCheck(BaseModel):
    """
    A jq validation attached to a pipeline step.

    `condition` checks skip the step when false. `before` checks run before the
    step. `after` checks run after a successful step and can inspect `.result`.
    `plain` checks run before the step, but when they pass, they stop the remaining
    validation chain for that step.
    """

    mode: ValidationMode = Field(..., description="When the validation runs relative to the step")
    expr: str = Field(..., description="jq expression that must evaluate truthy")
    message: str = Field(..., description="Error message when the check fails")


class StepConfig(BaseModel):
    agent: str = Field(..., description="The registry name of the agent to execute")
    description: str | None = Field(default=None, description="Human-readable step label")
    stage: str | None = Field(default=None, description="Execution stage shared by concurrently running steps")
    config: dict[str, Any] = Field(default_factory=dict, description="Step-specific configuration parameters")
    validation: list[ValidationCheck] = Field(default_factory=list, description="Validation checks for this configured step")


class PipelineConfig(BaseModel):
    """Configuration for a complete ingestion pipeline."""

    pipeline_id: str | None = Field(default=None, description="Identifier for the pipeline template")
    domain_profile: str = Field(default="default", description="Domain profile name")
    input_modality: str | None = Field(default=None, description="Expected input modality", examples=["text", "document", "audio"])
    description: str | None = Field(default=None, description="Purpose of this pipeline")
    parameters: dict[str, Any] = Field(default_factory=dict, description="JSON schema for configurable pipeline parameters")
    steps: list[StepConfig] = Field(..., description="Agentic steps partitioned into ordered execution stages")
    tool_vendors: dict[str, MCPServerTypes] = Field(default_factory=dict, description="Configured tool vendors available for agents")
