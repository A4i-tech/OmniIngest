"""Orchestration logic for defining and executing ingestion pipelines."""

import inspect
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import yaml
from pydantic_ai.embeddings import Embedder

from ..port.vector_store import InMemoryVectorStore
from .model import IngestionContext, KnowledgeItem, PipelineConfig, Step, StepResult, StepStatus
from .protocol import VectorStore


class PipelineRunner:
    """
    Orchestrates the execution of data processing steps within an ingestion pipeline.

    The PipelineRunner is responsible for initializing the IngestionContext,
    executing a sequence of configured steps (agents), and managing the overall
    lifecycle of the ingestion process, including error handling and status reporting.
    """

    def __init__(self, steps: list[Step], pipeline_version_id: UUID | None = None, vector_store: VectorStore | None = None, graph_store: Any | None = None):
        self.steps = steps
        self.pipeline_version_id = pipeline_version_id
        self.vector_store = vector_store
        self.graph_store = graph_store

    async def run(self, ctx: IngestionContext) -> list[StepResult]:
        """
        Executes the ingestion pipeline.

        Runs each step sequentially, updating the context and reporting progress
        to the metadata store if available.

        Args:
            ctx: The shared ingestion context containing items and state.

        Returns:
            A list of StepResult objects, one for each executed step.
        """
        run_id = None
        if ctx.store and self.pipeline_version_id:
            source_uri = ctx.metadata.get("source_uri") or ctx.metadata.get("file_path") or "unknown"
            run_id = await ctx.store.create_pipeline_run(ctx.tenant_id, self.pipeline_version_id, source_uri)
            ctx.run_id = run_id

        results = []
        final_status = "succeeded"
        error_msg = None

        for step in self.steps:
            sid = await ctx.store.create_step_run(ctx.run_id, lookup_step_id(step) or "unknown") if ctx.store else None
            try:
                res = await step.run(ctx)
                results.append(res)
                if sid and ctx.store:
                    await ctx.store.update_step_run(sid, status=res.status.value, metadata=res.metadata, error=res.error)
                if res.status == StepStatus.FAILURE:
                    final_status = "failed"
                    error_msg = res.error
                    break
            except Exception as e:
                err = str(e)
                results.append(StepResult(status=StepStatus.FAILURE, error=err))
                if sid and ctx.store:
                    await ctx.store.update_step_run(sid, status="failure", error=err)
                final_status = "failed"
                error_msg = err
                break

        if run_id and ctx.store:
            await ctx.store.update_pipeline_run(run_id, status=final_status, completed_at=datetime.now(UTC), error=error_msg)

        return results


_STEP_FACTORY: dict[str, type[Step]] = {}


def step(id: str):
    """
    Decorator to register a Step class in the global registry.

    Args:
        id: The name used to identify this step in configurations.
    """

    def decorator(cls: type):
        if not issubclass(cls, Step):
            raise TypeError(f"Class {cls.__name__} must inherit from Step")
        if cls in _STEP_FACTORY.values():
            raise ValueError(f"Step {id} already registered")
        _STEP_FACTORY[id] = cls
        return cls

    return decorator


def get_step_class(id: str) -> type[Step]:
    if id not in _STEP_FACTORY:
        raise ValueError(f"Step {id} not found")
    return _STEP_FACTORY[id]


def lookup_step_id(step: Step) -> str | None:
    for k, v in _STEP_FACTORY.items():
        if v == step.__class__:
            return k
    return None


def create_pipeline_from_config(config_path: str) -> PipelineRunner:
    """
    Factory function to create a PipelineRunner from a YAML configuration file.

    Args:
        config_path: Path to the .yaml configuration file.
    """
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    config = PipelineConfig.model_validate(raw_config)
    store = _create_vector_store(config.store)
    graph_store = _create_graph_store(config.graph_store)

    steps = []
    for s in config.steps:
        cls = get_step_class(s.agent)
        kwargs = _resolve_step_kwargs(cls, s.config, {"store": store, "graph_store": graph_store})
        steps.append(cls(**kwargs))

    return PipelineRunner(steps=steps, vector_store=store, graph_store=graph_store)


def _create_vector_store(c: dict[str, Any]):
    t = c.get("type", "in_memory")
    if t == "lancedb":
        from ..port.vector_store import LanceDBVectorStore

        return LanceDBVectorStore(table_name=c.get("table", "default_vectors"), db_path=c.get("path"), vector_size=c.get("vector_size", 1536))
    if t == "qdrant":
        from ..port.vector_store import QdrantVectorStore

        return QdrantVectorStore(
            col=c.get("collection", "default"),
            url=c.get("url"),
            key=c.get("api_key"),
            size=c.get("vector_size", 1536),
        )
    return InMemoryVectorStore()


def _create_graph_store(c: dict[str, Any]):
    t = c.get("type", "in_memory")
    if t == "neo4j":
        from ..port.graph_store import Neo4jGraphStore

        return Neo4jGraphStore(uri=str(c.get("uri")), user=str(c.get("user")), password=str(c.get("password")))
    from ..port.graph_store import InMemoryGraphStore

    return InMemoryGraphStore()


def _resolve_step_kwargs(cls: type, conf: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    from pydantic import BaseModel

    if issubclass(cls, BaseModel):
        fields = cls.model_fields
        return {p: conf[p] if p in conf else resources[p] for p in fields if p in conf or p in resources}

    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}
    return {p: conf[p] if p in conf else resources[p] for p in sig.parameters if p != "self" and (p in conf or p in resources)}


class KnowledgeBaseAdapter:
    """
    Provides a high-level interface for querying and retrieving ingested knowledge.

    The adapter abstracts the underlying storage mechanisms, such as vector
    databases and metadata stores, allowing for unified search and retrieval
    operations across different data domain profiles.
    """

    from .protocol import MetadataStore

    def __init__(self, metadata_store: "MetadataStore", vector_store: VectorStore, embedder: Embedder | None = None):
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.embedder = embedder or Embedder("openai:text-embedding-3-small")

    async def list_items(self, tenant_id: str | UUID, domain_profile: str | None = None, **filters) -> list[KnowledgeItem]:
        tid = UUID(str(tenant_id))
        return await self.metadata_store.list_knowledge_items(tenant_id=tid, domain_profile=domain_profile, **filters)

    async def query(self, tenant_id: str, query: list[float], k: int = 5, domain_profile: str | None = None) -> list[KnowledgeItem]:
        return await self.vector_store.search(query, k=k, tenant_id=tenant_id)

    async def get(self, item_id: str) -> KnowledgeItem | None:
        return await self.metadata_store.get_knowledge_item(item_id)

    async def query_text(
        self,
        tenant_id: str,
        query: str,
        k: int = 5,
        domain_profile: str | None = None,
    ) -> list[KnowledgeItem]:
        try:
            res = await self.embedder.embed_query(query)
            emb = list(res.embeddings[0])
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e
        return await self.query(tenant_id, emb, k=k, domain_profile=domain_profile)
