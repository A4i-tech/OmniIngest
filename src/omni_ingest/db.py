"""SQLAlchemy database schema and model definitions for persistent metadata storage."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models in the ingestion framework."""
    pass


class Tenant(Base):
    """Represents a customer or organizational unit that owns data."""

    __tablename__ = "tenants"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class Pipeline(Base):
    __tablename__ = "pipelines"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    domain_profile: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class PipelineVersion(Base):
    __tablename__ = "pipeline_versions"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pipeline_id: Mapped[UUID] = mapped_column(ForeignKey("pipelines.id"))
    version: Mapped[str] = mapped_column(String, nullable=False)
    yaml_profile_name: Mapped[str] = mapped_column(String, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    pipeline_version_id: Mapped[UUID] = mapped_column(ForeignKey("pipeline_versions.id"))
    source_uri: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)


class PipelineStepRun(Base):
    __tablename__ = "pipeline_step_runs"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pipeline_run_id: Mapped[UUID] = mapped_column(ForeignKey("pipeline_runs.id"))
    step_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    output_paths: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class KnowledgeItemRecord(Base):
    __tablename__ = "knowledge_items"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    domain_profile: Mapped[str] = mapped_column(String, nullable=False)
    modality: Mapped[str] = mapped_column(String, nullable=False)
    source_uri: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=True)
    embeddings: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    pipeline_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("pipeline_versions.id"), nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    structured_views: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    lineage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    knowledge_item_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_items.id"))
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=True)
    vector_store_kind: Mapped[str] = mapped_column(String, nullable=False)
    vector_store_space: Mapped[str] = mapped_column(String, nullable=False)
    vector_store_ref_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    __table_args__ = (Index("ix_chunks_knowledge_item", "knowledge_item_id"), Index("ix_chunks_tenant", "tenant_id"))
