"""Orchestration logic for defining and executing ingestion pipelines."""

import inspect
import traceback
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from fastmcp.mcp_config import MCPConfig
from pydantic import PositiveInt, validate_call
from pydantic_ai import AbstractToolset
from pydantic_ai.embeddings import Embedder
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from .. import agent
from ..core import event
from ..core.config import settings
from ..port.vector_store import InMemoryVectorStore
from .model import IngestionContext, KnowledgeItem, PipelineConfig, Step, StepResult, StepStatus
from .protocol import VectorStore

PIPELINE_RUNNERS: dict[UUID, "PipelineRunner"] = {}

type StepFactory = dict[str, type[Step]]
_STEP_FACTORY: StepFactory = {
    "chunking": agent.chunking.SentenceChunkingAgent,
    "page_chunking": agent.chunking.PageChunkingAgent,
    "timed_chunking": agent.chunking.TimedChunkingAgent,

    "markdown_cleaning": agent.cleaning.MarkdownCleaningAgent,
    "transcription_post_processor": agent.cleaning.TranscriptionPostProcessorAgent,

    "enrichment": agent.enrichment.DomainEnrichmentAgent,
    "learning_outcome": agent.enrichment.LearningOutcomeExtractionAgent,
    "kg_builder": agent.enrichment.KnowledgeGraphBuilderAgent,
    "faq_generation": agent.enrichment.FAQGenerationAgent,

    "ingestion_logger": agent.governance.IngestionLoggerAgent,
    "lineage": agent.governance.DataLineageAgent,
    "quality": agent.governance.QualityVerifierAgent,

    "embedding": agent.indexing.EmbeddingAgent,
    "indexing": agent.indexing.VectorIndexingAgent,
    "keyword_indexing": agent.indexing.KeywordIndexingAgent,
    "graph_indexing": agent.indexing.GraphIndexingAgent,

    "text_parser": agent.parsing.TextParserAgent,
    "document_parser": agent.parsing.DocumentParserAgent,
    "audio_parser": agent.parsing.AudioParserAgent,
}

class PipelineRunner:
    """
    Orchestrates the execution of data processing steps within an ingestion pipeline.

    The PipelineRunner is responsible for initializing the IngestionContext,
    executing a sequence of configured steps (agents), and managing the overall
    lifecycle of the ingestion process, including error handling and status reporting.
    """

    def __init__(self, steps: list[Step], pipeline_version_id: UUID | None = None, vector_store: VectorStore | None = None, graph_store: Any | None = None, toolsets: Mapping[str, AbstractToolset] = {}):
        self.steps = steps
        self.pipeline_version_id = pipeline_version_id or UUID(int=0)
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.toolsets = toolsets

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

        if not ctx.store:
            raise ValueError("Metadata store required in context")

        source_uri = ctx.metadata.get("source_uri") or ctx.metadata.get("file_path") or "unknown"
        ctx.run_id = await ctx.store.create_pipeline_run(ctx.tenant_id, self.pipeline_version_id, source_uri, ctx.run_id)
        PIPELINE_RUNNERS[ctx.run_id] = self
        try:
            result = []
            async for e in self._run(ctx):
                event.publish(e)
                if isinstance(e, event.PipelineStepEndEvent):
                    result.append(e.step_result)
            return result
        finally:
            del PIPELINE_RUNNERS[ctx.run_id]

    async def _run(self, ctx: IngestionContext) -> AsyncIterator[event.Event]:
        final_status = "succeeded"
        error_msg = None

        yield event.Event(identifier="pipeline_begin", metadata={"run_id": str(ctx.run_id), "tenant_id": str(ctx.tenant_id)})
        for step in self.steps:
            step_name = lookup_step_id(step) or "unknown"
            sid = await ctx.store.create_step_run(ctx.run_id, step_name)
            yield event.PipelineStepEvent(identifier="pipeline_step_begin", run_id=ctx.run_id, step_name=step_name, step=step)
            try:
                res = await step.run(ctx)
            except Exception as e:
                res = StepResult(status=StepStatus.FAILURE, error=str(e))
                traceback.print_exception(e)

            yield event.PipelineStepEndEvent(run_id=ctx.run_id, step_name=step_name, step=step, step_result=res)
            await ctx.store.update_step_run(sid, status=res.status.value, metadata=res.metadata, error=res.error)
            if res.status == StepStatus.FAILURE:
                final_status = "failed"
                error_msg = res.error
                break

        await ctx.store.update_pipeline_run(ctx.run_id, status=final_status, completed_at=datetime.now(UTC), error=error_msg)
        yield event.Event(identifier="pipeline_end", metadata={"run_id": str(ctx.run_id), "tenant_id": str(ctx.tenant_id), "status": final_status, "error": error_msg})


@validate_call
def register_step(id: str, cls: type[Step], override: bool = False):
    """
    Register a Step class in the global factory.

    Args:
        id: Name used to identify this step in configurations.
        cls: Step class the identifier must model.
        override: Whether to override if step with the ID already exists.
    """
    if not override and cls in _STEP_FACTORY.values():
        raise ValueError(f"Step {id} already registered")
    _STEP_FACTORY[id] = cls


def get_step_class(id: str, factory: StepFactory | None = None) -> type[Step]:
    factory = factory or _STEP_FACTORY
    if id not in factory:
        raise ValueError(f"Step {id} not found")
    return factory[id]


def lookup_step_id(step: Step, factory: StepFactory | None = None) -> str | None:
    factory = factory or _STEP_FACTORY
    for k, v in factory.items():
        if v == step.__class__:
            return k
    return None


def create_pipeline_from_config(config_path: Path, factory: StepFactory | None = None) -> PipelineRunner:
    """
    Factory function to create a PipelineRunner from a YAML configuration file.

    Args:
        config_path: Path to the .yaml configuration file.
        factory: Factory to use to resolve and load steps from.
    """
    with config_path.open() as f:
        raw_config = yaml.safe_load(f)

    config = PipelineConfig.model_validate(raw_config)
    store = _create_vector_store(config.store)
    graph_store = _create_graph_store(config.graph_store)

    steps = []
    for s in config.steps:
        cls = get_step_class(s.agent, factory)
        kwargs = _resolve_step_kwargs(cls, s.config, {"store": store, "graph_store": graph_store})
        steps.append(cls(**kwargs))

    toolsets = {k: FastMCPToolset(MCPConfig(mcpServers={k: v})) for k, v in config.tool_vendors.items()}
    return PipelineRunner(steps=steps, vector_store=store, graph_store=graph_store, toolsets=toolsets)


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
        self.embedder = embedder or Embedder(settings.default_embedding_model)

    async def list_items(self, tenant_id: UUID, domain_profile: str | None = None, **filters) -> list[KnowledgeItem]:
        return await self.metadata_store.list_knowledge_items(tenant_id=tenant_id, domain_profile=domain_profile, **filters)

    async def query(self, tenant_id: UUID, query: list[float], k: PositiveInt = 5, domain_profile: str | None = None) -> list[KnowledgeItem]:
        return await self.vector_store.search(query, k=k, tenant_id=tenant_id)

    async def get(self, item_id: UUID) -> KnowledgeItem | None:
        return await self.metadata_store.get_knowledge_item(item_id)

    async def query_text(
        self,
        tenant_id: UUID,
        query: str,
        k: PositiveInt = 5,
        domain_profile: str | None = None,
    ) -> list[KnowledgeItem]:
        try:
            res = await self.embedder.embed_query(query)
            emb = list(res.embeddings[0])
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e
        return await self.query(tenant_id, emb, k=k, domain_profile=domain_profile)
