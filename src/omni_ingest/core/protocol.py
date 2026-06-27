"""Protocol definitions (Abstract Base Classes) for framework extensibility."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import PositiveInt

from ..db import PipelineVersion, Tenant
from .model import KnowledgeItem, PipelineCheckpoint, ResolvedResource

if TYPE_CHECKING:
    from .pipeline import IngestionContext

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@runtime_checkable
class ContentResolver(Protocol):
    """Resolve virtual content URIs into bytes during ingestion.

    `ByteContent.content(ctx)` delegates to `ctx.content_resolver` whenever an
    item has `content_uri` instead of inline `raw_content`. Implementations can
    use full ingestion context to resolve routes against source resources,
    object stores, caches, or other pipeline-scoped state.
    """

    async def resolve(self, uri: str, ctx: IngestionContext[ResolvedResource]) -> bytes:
        """Return resolved bytes or raise `ValueError` when URI is unsupported."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """
    Interface for vector-based search and retrieval systems.

    The VectorStore protocol specifies methods for storing and querying
    mathematical vector representations of data, enabling semantic
    search capabilities based on contextual meaning.
    """

    async def upsert(self, items: list[KnowledgeItem], ctx: Any) -> None:
        """Persists or updates knowledge items in the store."""
        ...

    async def search(self, query: list[float], k: PositiveInt = 5, tenant_id: UUID | None = None) -> list[KnowledgeItem]:
        """Searches for items similar to the given query vector."""
        ...

    async def count(self) -> int:
        """Returns the total number of items in the store."""
        ...

    async def __aenter__(self) -> VectorStore:
        return self

    async def __aexit__(self, exc_type, exc, tb):
        ...

    async def close(self) -> None:
        """Closes the connection to the store."""
        ...

    @property
    def kind(self) -> str:
        """Stable identifier for the backing vector store implementation."""
        ...

    @property
    def space(self) -> str:
        """Logical namespace used by the vector store, such as a table or collection."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    async def upsert_nodes(self, nodes: list[dict]) -> None: ...
    async def upsert_edges(self, edges: list[dict]) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class MetadataStore(Protocol):
    """
    Interface for metadata management and data lineage tracking.

    The MetadataStore protocol defines the operations required for
    managing relational data, including multi-tenancy records, pipeline
    execution logs, and data provenance information.
    """

    async def get_tenant(self, slug: str) -> Tenant | None:
        """Retrieves a tenant by slug."""
        ...

    async def create_tenant(self, slug: str, display_name: str) -> Tenant:
        """Creates a new tenant."""
        ...

    async def create_pipeline_run(self, tenant_id: UUID, pipeline_version_id: UUID, source_uri: str, id: UUID | None = None) -> UUID:
        """Records the start of a pipeline run."""
        ...

    async def update_pipeline_run(self, run_id: UUID, status: str, completed_at: datetime | None = None, error: str | None = None) -> None:
        """Updates the status of a pipeline run."""
        ...

    async def get_pipeline_checkpoint(self, id: UUID) -> PipelineCheckpoint | None:
        """Returns the latest checkpoint for a pipeline run."""
        ...

    async def set_pipeline_checkpoint(self, id: UUID, checkpoint: PipelineCheckpoint | None) -> None:
        """Replaces the checkpoint, or deletes it when checkpoint is None."""
        ...

    async def create_step_run(self, run_id: UUID, step_name: str) -> UUID:
        """Records the start of a pipeline step."""
        ...

    async def update_step_run(self, step_id: UUID, status: str, metadata: JsonObject | None = None, error: str | None = None) -> None:
        """Updates the status and metadata of a step run."""
        ...

    async def create_knowledge_item(
        self,
        tenant_id: UUID,
        domain_profile: str,
        modality: str,
        source_uri: str,
        content: bytes | None = None,
        pipeline_version_id: UUID | None = None,
        title: str | None = None,
        description: str | None = None,
        structured_views: JsonObject | None = None,
        lineage: JsonObject | None = None,
        metadata: JsonObject | None = None,
    ) -> UUID:
        """Persists a new knowledge item."""
        ...

    async def get_knowledge_item(self, item_id: UUID) -> KnowledgeItem | None:
        """Retrieves a knowledge item by ID."""
        ...

    async def list_knowledge_items(self, tenant_id: UUID, domain_profile: str | None = None, **filters: object) -> list[KnowledgeItem]:
        """Lists knowledge items for a tenant."""
        ...

    async def get_pipeline_version(self, pipeline_name: str, version: str) -> PipelineVersion | None:
        """Retrieves a specific pipeline version."""
        ...

    async def get_default_pipeline_version(self, pipeline_name: str) -> PipelineVersion | None:
        """Retrieves the default version for a pipeline."""
        ...

    async def upsert_pipeline(self, name: str, domain_profile: str) -> UUID:
        """Creates or updates a pipeline definition."""
        ...

    async def upsert_pipeline_version(self, pipeline_id: UUID, version: str, yaml_profile_name: str, is_default: bool = False) -> UUID:
        """Creates or updates a pipeline version."""
        ...
