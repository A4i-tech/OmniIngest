"""Protocol definitions (Abstract Base Classes) for framework extensibility."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .model import KnowledgeItem


@runtime_checkable
class ObjectStore(Protocol):
    """
    Interface for binary data storage systems.

    The ObjectStore protocol defines methods for the persistence and
    retrieval of raw binary data, such as documents, images, or audio files,
    across various storage backends.
    """

    async def put(self, data: bytes, key: str, metadata: dict | None = None) -> str: ...
    async def get(self, key: str) -> bytes: ...
    async def exists(self, key: str) -> bool: ...
    async def delete(self, key: str) -> None: ...
    async def close(self) -> None: ...


class VectorStore(Protocol):
    """
    Interface for vector-based search and retrieval systems.

    The VectorStore protocol specifies methods for storing and querying
    mathematical vector representations of data, enabling semantic
    search capabilities based on contextual meaning.
    """

    async def upsert(self, items: list[KnowledgeItem]) -> None:
        """Persists or updates knowledge items in the store."""
        ...

    async def search(self, query: list[float], k: int = 5, tenant_id: str | None = None) -> list[KnowledgeItem]:
        """Searches for items similar to the given query vector."""
        ...

    async def count(self) -> int:
        """Returns the total number of items in the store."""
        ...

    async def close(self) -> None:
        """Closes the connection to the store."""
        ...


class GraphStore(Protocol):
    async def upsert_nodes(self, nodes: list[dict]) -> None: ...
    async def upsert_edges(self, edges: list[dict]) -> None: ...
    async def close(self) -> None: ...


class SourceResource(BaseModel):
    id: str
    uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceConnector(Protocol):
    async def list_resources(self, filters: dict[str, Any] | None = None) -> list[SourceResource]:
        """Retrieves a list of available resources from the source."""
        ...

    async def fetch(self, resource_id: str) -> bytes:
        """Retrieves the raw content of a specific resource."""
        ...


@runtime_checkable
class MetadataStore(Protocol):
    """
    Interface for metadata management and data lineage tracking.

    The MetadataStore protocol defines the operations required for
    managing relational data, including multi-tenancy records, pipeline
    execution logs, and data provenance information.
    """

    async def get_tenant(self, slug: str) -> Any:
        """Retrieves a tenant by slug."""
        ...

    async def create_tenant(self, slug: str, display_name: str) -> Any:
        """Creates a new tenant."""
        ...

    async def create_pipeline_run(self, tenant_id: Any, pipeline_version_id: Any, source_uri: str) -> Any:
        """Records the start of a pipeline run."""
        ...

    async def update_pipeline_run(self, run_id: Any, status: str, completed_at: Any = None, error: str | None = None) -> None:
        """Updates the status of a pipeline run."""
        ...

    async def create_step_run(self, run_id: Any, step_name: str) -> Any:
        """Records the start of a pipeline step."""
        ...

    async def update_step_run(self, step_id: Any, status: str, metadata: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Updates the status and metadata of a step run."""
        ...

    async def create_knowledge_item(
        self,
        tenant_id: Any,
        domain_profile: str,
        modality: str,
        source_uri: str,
        content: str | None = None,
        pipeline_version_id: Any | None = None,
        title: str | None = None,
        description: str | None = None,
        structured_views: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Persists a new knowledge item."""
        ...

    async def get_knowledge_item(self, item_id: Any) -> KnowledgeItem | None:
        """Retrieves a knowledge item by ID."""
        ...

    async def create_chunk(
        self,
        chunk_id: str,
        knowledge_item_id: Any,
        tenant_id: Any,
        text: str,
        vector_store_kind: str,
        vector_store_space: str,
        vector_store_ref_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Records a data chunk and its vector store reference."""
        ...

    async def get_chunks_by_knowledge_item(self, knowledge_item_id: Any) -> list[Any]:
        """Retrieves all chunks for a specific knowledge item."""
        ...

    async def get_chunk(self, chunk_id: str) -> Any:
        """Retrieves a specific chunk."""
        ...

    async def list_knowledge_items(self, tenant_id: Any, domain_profile: str | None = None, **filters) -> list[KnowledgeItem]:
        """Lists knowledge items for a tenant."""
        ...

    async def list_chunks(self, tenant_id: Any, **filters) -> list[Any]:
        """Lists chunks for a tenant."""
        ...

    async def get_pipeline_version(self, pipeline_name: str, version: str) -> Any:
        """Retrieves a specific pipeline version."""
        ...

    async def get_default_pipeline_version(self, pipeline_name: str) -> Any:
        """Retrieves the default version for a pipeline."""
        ...

    async def upsert_pipeline(self, name: str, domain_profile: str) -> Any:
        """Creates or updates a pipeline definition."""
        ...

    async def upsert_pipeline_version(self, pipeline_id: Any, version: str, yaml_profile_name: str, is_default: bool = False) -> Any:
        """Creates or updates a pipeline version."""
        ...
