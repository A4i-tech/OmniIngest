"""Orchestration logic for defining and executing ingestion pipelines."""

import asyncio
import inspect
import logging
import traceback
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from fastmcp import Client
from fastmcp.mcp_config import MCPConfig
from pydantic import PositiveInt, validate_call
from pydantic_ai import AbstractToolset
from pydantic_ai.embeddings import Embedder
from pydantic_ai.mcp import MCPToolset

from .. import agent
from ..core import event
from ..core.config import settings
from ..parser import read_resources
from .model import IngestionContext, KnowledgeItem, PipelineConfig, ResolvedResource, Step, StepConfig, StepResult, StepStatus
from .output import Output, OutputResult
from .protocol import VectorStore

PIPELINE_RUNNERS: dict[UUID, "PipelineRunner"] = {}

type StepFactory = dict[str, type[Step]]
_STEP_FACTORY: StepFactory = {
    "text": agent.modality.TextAgent,

    "chunking": agent.chunking.SentenceChunkingAgent,
    "page_chunking": agent.chunking.PageChunkingAgent,
    "timed_chunking": agent.chunking.TimedChunkingAgent,

    "markdown_cleaning": agent.cleaning.MarkdownCleaningAgent,
    "transcription_post_processor": agent.cleaning.TranscriptionPostProcessorAgent,

    "structure": agent.structure.StructureAgent,
    "kg_builder": agent.enrichment.KnowledgeGraphBuilderAgent,
    "faq_generation": agent.enrichment.FAQGenerationAgent,

    "ingestion_logger": agent.governance.IngestionLoggerAgent,
    "lineage": agent.governance.DataLineageAgent,
    "quality": agent.governance.QualityVerifierAgent,

    "embedding": agent.indexing.EmbeddingAgent,

    "pipeline": agent.misc.PipelineAgent
}

class PipelineRunner:
    """
    Orchestrates the execution of data processing steps within an ingestion pipeline.

    The PipelineRunner is responsible for initializing the IngestionContext,
    executing a sequence of configured steps (agents), and managing the overall
    lifecycle of the ingestion process, including error handling and status reporting.
    """

    def __init__(self, steps: list[Step], pipeline_version_id: UUID | None = None, toolsets: Mapping[str, AbstractToolset] = {}, logger: logging.Logger | None = None):
        self.steps = steps
        self.pipeline_version_id = pipeline_version_id or UUID(int=0)
        self.toolsets = toolsets
        self.output_result: OutputResult | None = None

        if logger is None:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
        self.logger = logger

    async def run(self, ctx: IngestionContext[Any], output: Output | None = None) -> list[StepResult]:
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

        ctx = await self._resolve_resource_single(ctx)
        ctx.run_id = await ctx.store.create_pipeline_run(ctx.tenant_id, self.pipeline_version_id, ctx.resource.uri, ctx.run_id)
        PIPELINE_RUNNERS[ctx.run_id] = self
        try:
            result = []
            async for e in self._run(ctx):
                event.publish(e)
                if isinstance(e, event.PipelineStepEndEvent):
                    result.append(e.step_result)
            if output is not None and all(r.status != StepStatus.FAILURE for r in result):
                self.output_result = await output.write(ctx)
            return result
        finally:
            del PIPELINE_RUNNERS[ctx.run_id]

    async def flatten(self, ctx: IngestionContext[Any]) -> AsyncIterator[IngestionContext[ResolvedResource]]:
        if isinstance(ctx.resource, ResolvedResource):
            yield ctx
            return

        async for resource in read_resources(str(ctx.resource)):
            yield ctx.model_copy(update={"run_id": None, "resource": resource})

    async def run_flattened(self, ctx: IngestionContext[Any]) -> AsyncIterator[tuple[IngestionContext[ResolvedResource], list[StepResult]]]:
        queue = asyncio.Queue[tuple[IngestionContext[ResolvedResource], list[StepResult]] | BaseException | None]()
        async def runner(child):
            try:
                await queue.put((child, await self.run(child)))
            except BaseException as e:
                await queue.put(e)
                raise

        async def producer():
            try:
                async with asyncio.TaskGroup() as tg:
                    async for child in self.flatten(ctx):
                        tg.create_task(runner(child))
            finally:
                await queue.put(None)

        task = asyncio.get_event_loop().create_task(producer())
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            await task

    async def _run(self, ctx: IngestionContext[ResolvedResource]) -> AsyncIterator[event.Event]:
        final_status = "succeeded"
        error_msg = None

        yield event.PipelineBeginEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id)
        for step in self.steps:
            step_name = lookup_step_id(step) or "unknown"
            sid = await ctx.store.create_step_run(ctx.run_id, step_name)
            yield event.PipelineStepBeginEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id, step_name=step_name, step=step)
            try:
                res = await step.run(ctx)
            except Exception as e:
                res = StepResult(status=StepStatus.FAILURE, error=str(e))
                traceback.print_exception(e)

            yield event.PipelineStepEndEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id, step_name=step_name, step=step, step_result=res)
            await ctx.store.update_step_run(sid, status=res.status.value, metadata=res.metadata, error=res.error)
            if res.status == StepStatus.FAILURE:
                final_status = "failed"
                error_msg = res.error
                break

        await ctx.store.update_pipeline_run(ctx.run_id, status=final_status, completed_at=datetime.now(UTC), error=error_msg)
        yield event.PipelineEndEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id, status=final_status, error=error_msg)

    async def _resolve_resource_single(self, ctx: IngestionContext[Any], **kwargs) -> IngestionContext[ResolvedResource]:
        if isinstance(ctx.resource, ResolvedResource):
            return ctx

        resolved = None
        skipped = []
        async for res in read_resources(str(ctx.resource), parse_limit=1, **kwargs):
            if resolved is None:
                resolved = res
            else:
                skipped.append(res.uri)

        if resolved is None:
            raise RuntimeError("Could not resolve resource: " + str(ctx.resource))

        if len(skipped) > 0:
            self.logger.warning("Resource %s was resolved to %s, but also returned %d other resource(s) which were ignored. "
                "Consider using %s() instead, or %s() followed by %s() for a more granularity over execution.",
                str(ctx.resource), resolved.uri, len(skipped), self.run_flattened.__qualname__, self.flatten.__qualname__, self.run.__qualname__
            )

        ctx.resource = resolved
        return ctx


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


def create_pipeline_from_config(config_path: Path, factory: StepFactory | None = None, config_args: dict[str, str] | None = None) -> PipelineRunner:
    """
    Factory function to create a PipelineRunner from a YAML configuration file.

    Args:
        config_path: Path to the .yaml configuration file.
        factory: Factory to use to resolve and load steps from.
    """
    config = _load_pipeline_config(config_path)

    steps = []
    for s in _flatten_steps(config.steps, config_path.parent, factory):
        cls = get_step_class(s.agent, factory)
        #resolved_config = json.loads(json.dumps(s.config).format_map(config_args or {}))
        kwargs = _resolve_step_kwargs(cls, s.config, {})
        steps.append(cls(**kwargs))
    toolsets = {k: MCPToolset(Client(MCPConfig(mcpServers={k: v}).to_dict())) for k, v in config.tool_vendors.items()}
    return PipelineRunner(steps=steps, toolsets=toolsets)


def _load_pipeline_config(path: Path) -> PipelineConfig:
    with path.open() as f:
        return PipelineConfig.model_validate(yaml.safe_load(f))


def _flatten_steps(steps: list[StepConfig], base_dir: Path, factory: StepFactory | None = None, stack: tuple[Path, ...] = ()) -> list[StepConfig]:
    flat = []
    for step in steps:
        if path := _resolve_step_pipeline_path(step.agent, base_dir, factory):
            path = path.resolve()
            if path in stack:
                cycle = " -> ".join(str(p) for p in (*stack, path))
                raise ValueError(f"Pipeline component cycle detected: {cycle}")
            flat.extend(_flatten_steps(_load_pipeline_config(path).steps, path.parent, factory, (*stack, path)))
        else:
            flat.append(step)
    return flat


def _resolve_step_pipeline_path(agent_id: str, base_dir: Path, factory: StepFactory | None = None) -> Path | None:
    path = Path(agent_id)
    if not path.suffix:
        path = path.with_suffix(".yaml")

    local = path if path.is_absolute() else base_dir / path
    if local.is_file():
        return local

    profile = Path("profiles") / path
    if profile.is_file():
        return profile

    get_step_class(agent_id, factory)
    return None


def _resolve_step_kwargs(cls: type, conf: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    from pydantic import BaseModel

    if issubclass(cls, BaseModel):
        fields = cls.model_fields
        kwargs = {p: resources[p] for p in fields if p in resources}
        kwargs.update(conf)
        return kwargs

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
