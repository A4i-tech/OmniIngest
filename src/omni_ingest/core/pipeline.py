"""Orchestration logic for defining and executing ingestion pipelines."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import traceback
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID, uuid4

import jq
import yaml
from fastmcp import Client
from fastmcp.mcp_config import MCPConfig
from jsonschema import ValidationError, validate
from pydantic import PositiveInt, validate_call
from pydantic_ai import AbstractToolset
from pydantic_ai.embeddings import Embedder
from pydantic_ai.mcp import MCPToolset

from .. import agent
from ..core import event
from ..core.config import settings
from ..parser import read_resources
from ..port.content import default_content_resolver
from .model import KnowledgeItem, PipelineCheckpoint, PipelineConfig, ResolvedResource, Resource, Step, StepConfig, StepResult, StepStatus, ValidationCheck, ValidationMode
from .output import Output, OutputResult
from .protocol import ContentResolver, MetadataStore, VectorStore

PIPELINE_RUNNERS: dict[UUID, PipelineRunner] = {}

ResourceT = TypeVar("ResourceT", bound=Resource)
_INTERPOLATION_RE = re.compile(r"\$\{([^{}]+)\}")
_CURRENT_STEP: ContextVar[tuple[str, Step] | None] = ContextVar("_CURRENT_STEP", default=None)


@cache
def default_metadata_store() -> MetadataStore:
    from ..port.metadata_store import NullMetadataStore, SQLAlchemyMetadataStore
    if settings.metadata_store == "null":
        return NullMetadataStore()
    return SQLAlchemyMetadataStore()


@dataclass(kw_only=True)
class IngestionContext(Generic[ResourceT]):  # noqa: UP046
    """
    Maintains state and shared metadata across a single ingestion pipeline execution.

    The IngestionContext object facilitates communication between sequential
    processing steps by carrying operational state, configuration parameters,
    and the current set of KnowledgeItems.
    """

    resource: ResourceT

    store: MetadataStore = field(default_factory=default_metadata_store)
    """Reference to a storage engine or backend"""

    content_resolver: ContentResolver = field(default_factory=default_content_resolver)
    """Reference to a storage engine or backend"""

    run_id: UUID = field(default_factory=uuid4)
    """Unique identifier for the current pipeline run"""

    tenant_id: UUID = field(default_factory=uuid4)
    """Identifier for the active tenant"""

    domain_profile: str = "default"
    """The configuration profile being executed"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Shared state or context across steps"""

    items: list[KnowledgeItem] = field(default_factory=list)
    """The working set of knowledge items"""

    def progress(self, completed: float, total: float, message: str | None = None) -> None:
        if current := _CURRENT_STEP.get():
            step_name, step = current
            event.publish(event.PipelineStepProgressEvent(run_id=self.run_id, tenant_id=self.tenant_id, step_name=step_name, step=step, completed=completed, total=total, message=message))


type StepFactory = dict[str, type[Step]]
_STEP_FACTORY: StepFactory = {
    "text": agent.modality.TextAgent,

    "chunking": agent.chunking.SentenceChunkingAgent,
    "image_extraction": agent.document.ImageExtractionAgent,
    "rasterize": agent.document.RasterizeAgent,
    "page_chunking": agent.chunking.PageChunkingAgent,
    "page_stitching": agent.chunking.PageStitchingAgent,
    "timed_chunking": agent.chunking.TimedChunkingAgent,

    "prompt": agent.enrichment.PromptAgent,

    "extract": agent.enrichment.ExtractAgent,
    "transform": agent.enrichment.TransformAgent,
    "translate": agent.enrichment.TranslateAgent,

    "ingestion_logger": agent.governance.IngestionLoggerAgent,
    "lineage": agent.governance.DataLineageAgent,

    "embedding": agent.indexing.EmbeddingAgent,
}

class PipelineRunner:
    """
    Orchestrates the execution of data processing steps within an ingestion pipeline.

    The PipelineRunner is responsible for initializing the IngestionContext,
    executing ordered stages of configured steps (agents), and managing the overall
    lifecycle of the ingestion process, including error handling and status reporting.
    Steps in a stage run concurrently against the same ingestion context.
    """

    def __init__(self, steps: list[Step] | Mapping[str, list[Step]], pipeline_version_id: UUID | None = None, toolsets: Mapping[str, AbstractToolset] = {}, logger: logging.Logger | None = None, step_descriptions: Mapping[int, str] | None = None, step_validations: Mapping[int, list[ValidationCheck]] | None = None):
        self.stages = dict(steps) if isinstance(steps, Mapping) else {f"stage-{i}-{uuid4().hex[:8]}": [step] for i, step in enumerate(steps, 1)}
        self.steps = [step for stage in self.stages.values() for step in stage]
        self.pipeline_version_id = pipeline_version_id or UUID(int=0)
        self.toolsets = toolsets
        self.output_result: OutputResult | None = None
        self.step_descriptions = dict(step_descriptions or {})
        self.step_validations = dict(step_validations or {})

        if logger is None:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
        self.logger = logger

    async def run(self, ctx: IngestionContext[Any], output: Output | None = None) -> list[StepResult]:
        """
        Executes the ingestion pipeline.

        Runs stages sequentially and their steps concurrently, updating the shared
        context and reporting progress to the metadata store.

        Args:
            ctx: The shared ingestion context containing items and state.

        Returns:
            A list of StepResult objects, one for each executed step.
        """

        if not ctx.store:
            raise ValueError("Metadata store required in context")

        ctx = await self._resolve_resource_single(ctx)
        if not ctx.items:
            ctx.items = [KnowledgeItem(raw_content=ctx.resource.raw_content, content_uri=ctx.resource.content_uri, content_encoding=ctx.resource.content_encoding, metadata=ctx.resource.metadata.copy(), tenant_id=ctx.tenant_id, source_uri=ctx.resource.uri)]
        ctx.run_id = await ctx.store.create_pipeline_run(ctx.tenant_id, self.pipeline_version_id, ctx.resource.uri, ctx.run_id)
        await ctx.store.set_pipeline_checkpoint(ctx.run_id, self._checkpoint(ctx, 0))
        return await self._execute(ctx, output, 0)

    async def resume(self, run_id: UUID, store: MetadataStore, output: Output | None = None) -> tuple[IngestionContext[ResolvedResource], Awaitable[list[StepResult]]]:
        """Resumes a pipeline run from its latest successful-stage checkpoint."""

        checkpoint = await store.get_pipeline_checkpoint(run_id)
        if checkpoint is None:
            raise ValueError(f"Pipeline checkpoint not found: {run_id}")
        if checkpoint.stage > len(self.stages):
            raise ValueError(f"Pipeline has {len(self.stages)} stages, but checkpoint {run_id} completed {checkpoint.stage}; resume with the pipeline YAML used to create this run")
        ctx = IngestionContext(resource=checkpoint.resource, store=store, run_id=run_id, tenant_id=checkpoint.tenant_id, domain_profile=checkpoint.domain_profile, metadata=checkpoint.metadata, items=checkpoint.items)
        await store.update_pipeline_run(run_id, status="running")
        return ctx, self._execute(ctx, output, checkpoint.stage)

    async def _execute(self, ctx: IngestionContext[ResolvedResource], output: Output | None, stage: int) -> list[StepResult]:
        PIPELINE_RUNNERS[ctx.run_id] = self
        try:
            result = []
            async for e in self._run(ctx, stage):
                event.publish(e)
                if isinstance(e, event.PipelineStepEndEvent):
                    result.append(e.step_result)
            success = all(r.status != StepStatus.FAILURE for r in result)
            if success:
                await ctx.store.set_pipeline_checkpoint(ctx.run_id, None)
            if output is not None and success:
                self.output_result = await output.write(ctx)
            return result
        finally:
            del PIPELINE_RUNNERS[ctx.run_id]

    async def flatten(self, ctx: IngestionContext[Any]) -> AsyncIterator[IngestionContext[ResolvedResource]]:
        if isinstance(ctx.resource, ResolvedResource):
            yield ctx
            return

        async for resource in read_resources(str(ctx.resource)):
            yield replace(ctx, run_id=uuid4(), resource=resource)

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

    async def _run(self, ctx: IngestionContext[ResolvedResource], start: int) -> AsyncIterator[event.Event]:
        final_status: Literal["succeeded", "failed"] = "succeeded"
        error_msg = None

        yield event.PipelineBeginEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id, completed_steps=self.steps[:sum(map(len, list(self.stages.values())[:start]))])
        for stage_index, stage in enumerate(self.stages.values()):
            if stage_index < start:
                continue
            step_runs = []
            for step in stage:
                step_name = lookup_step_id(step) or "unknown"
                step_runs.append((await ctx.store.create_step_run(ctx.run_id, step_name), step_name, step, self.step_validations.get(id(step), [])))
                yield event.PipelineStepBeginEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id, step_name=step_name, step=step)
            results = await asyncio.gather(*(self._run_step(ctx, step_name, step, validation) for _, step_name, step, validation in step_runs))
            for (sid, step_name, step, _), res in zip(step_runs, results, strict=True):
                yield event.PipelineStepEndEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id, step_name=step_name, step=step, step_result=res)
                await ctx.store.update_step_run(sid, status=res.status.value, metadata=res.metadata, error=res.error)
            if failed := next((res for res in results if res.status == StepStatus.FAILURE), None):
                final_status = "failed"
                error_msg = failed.error
                break
            await ctx.store.set_pipeline_checkpoint(ctx.run_id, self._checkpoint(ctx, stage_index + 1))

        await ctx.store.update_pipeline_run(ctx.run_id, status=final_status, completed_at=datetime.now(UTC), error=error_msg)
        yield event.PipelineEndEvent(run_id=ctx.run_id, tenant_id=ctx.tenant_id, status=final_status, error=error_msg)

    async def _run_step(self, ctx: IngestionContext[ResolvedResource], step_name: str, step: Step, validation: list[ValidationCheck]) -> StepResult:
        token = _CURRENT_STEP.set((step_name, step))
        try:
            if skipped := self._validate(ctx, validation, {"condition"})[0]:
                return skipped.model_copy(update={"status": StepStatus.SKIPPED})
            failed, stop_validation = self._validate(ctx, validation, {"before", "plain"})
            if failed:
                return failed
            result = await step.run(ctx)
            if result.status == StepStatus.SUCCESS and not stop_validation and (failed := self._validate(ctx, validation, {"after"}, result)[0]):
                return failed
            return result
        except Exception as e:
            traceback.print_exception(e)
            return StepResult(status=StepStatus.FAILURE, error=str(e))
        finally:
            _CURRENT_STEP.reset(token)

    def _validate(self, ctx: IngestionContext[ResolvedResource], validation: list[ValidationCheck], modes: set[ValidationMode], result: StepResult | None = None) -> tuple[StepResult | None, bool]:
        doc: dict[str, Any] = {
            "metadata": ctx.metadata,
            "items": [{"id": str(item.id), "source_uri": item.source_uri, "metadata": item.metadata} for item in ctx.items],
        }
        if result is not None:
            doc["result"] = result.model_dump(mode="json", exclude={"items": {"__all__": {"raw_content"}}})
        for check in validation:
            if check.mode not in modes:
                continue
            if not jq.compile(check.expr).input_value(doc).first():
                return StepResult(status=StepStatus.FAILURE, error=_interpolate_value(check.message, doc)), False
            if check.mode == "plain":
                return None, True
        return None, False

    def _checkpoint(self, ctx: IngestionContext[ResolvedResource], stage: int) -> PipelineCheckpoint:
        return PipelineCheckpoint(stage=stage, resource=ctx.resource, tenant_id=ctx.tenant_id, domain_profile=ctx.domain_profile, metadata=ctx.metadata, items=ctx.items)

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


def create_pipeline_from_config(config_path: Path, factory: StepFactory | None = None, config_args: dict[str, Any] | None = None) -> PipelineRunner:
    """
    Factory function to create a PipelineRunner from a YAML configuration file.

    Args:
        config_path: Path to the .yaml configuration file.
        factory: Factory to use to resolve and load steps from.
    """
    config = _load_pipeline_config(config_path)

    stages: dict[str, list[Step]] = {}
    descriptions = {}
    validations = {}
    for s in _flatten_steps(_resolve_pipeline_steps(config, config_args or {}, config_path), config_path.parent, factory):
        stage = s.stage if s.stage is not None else f"stage-{len(stages) + 1}-{uuid4().hex[:8]}"
        if stage in stages and stage != next(reversed(stages)):
            raise ValueError(f"Stage {stage} must be contiguous")
        cls = get_step_class(s.agent, factory)
        kwargs = _resolve_step_kwargs(cls, s.config, {})
        step = cls(**kwargs)
        if s.description:
            descriptions[id(step)] = s.description
        if s.validation:
            validations[id(step)] = s.validation
        stages.setdefault(stage, []).append(step)
    toolsets = {k: MCPToolset(Client(MCPConfig(mcpServers={k: v}).to_dict())) for k, v in config.tool_vendors.items()}
    return PipelineRunner(steps=stages, toolsets=toolsets, step_descriptions=descriptions, step_validations=validations)


def _load_pipeline_config(path: Path) -> PipelineConfig:
    with path.open() as f:
        return PipelineConfig.model_validate(yaml.safe_load(f))


def _flatten_steps(steps: list[StepConfig], base_dir: Path, factory: StepFactory | None = None, stack: tuple[Path, ...] = ()) -> list[StepConfig]:
    flat = []
    for step in steps:
        if path := _resolve_step_pipeline_path(step.agent, base_dir, factory):
            if step.stage is not None:
                raise ValueError(f"Stage cannot be assigned to pipeline component {step.agent}")
            if step.validation:
                raise ValueError(f"Validation cannot be assigned to pipeline component {step.agent}")
            path = path.resolve()
            if path in stack:
                cycle = " -> ".join(str(p) for p in (*stack, path))
                raise ValueError(f"Pipeline component cycle detected: {cycle}")
            config = _load_pipeline_config(path)
            flat.extend(_flatten_steps(_resolve_pipeline_steps(config, step.config, path), path.parent, factory, (*stack, path)))
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

    profile = settings.profiles_dir / path
    if profile.is_file():
        return profile

    get_step_class(agent_id, factory)
    return None


def _resolve_pipeline_steps(config: PipelineConfig, overrides: dict[str, Any], config_path: Path) -> list[StepConfig]:
    schema = {"type": "object", "additionalProperties": False, **config.parameters}
    values = {k: v["default"] for k, v in schema.get("properties", {}).items() if "default" in v} | overrides
    try:
        validate(instance=values, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for pipeline {config_path}: {e.message}") from e
    return [
        StepConfig(
            agent=step.agent,
            description=step.description,
            stage=step.stage,
            config=_interpolate_value(step.config, {"config": values}),
            validation=[check.model_copy(update={"expr": _interpolate_value(check.expr, {"config": values})}) for check in step.validation],
        )
        for step in config.steps
    ]


def _interpolate_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(v, context) for v in value]
    if not isinstance(value, str):
        return value

    full_match = _INTERPOLATION_RE.fullmatch(value)
    if full_match:
        return jq.compile(full_match.group(1).strip()).input_value(context).first()

    def replace(match: re.Match[str]) -> str:
        return str(jq.compile(match.group(1).strip()).input_value(context).first())

    return _INTERPOLATION_RE.sub(replace, value)


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

    def __init__(self, metadata_store: MetadataStore, vector_store: VectorStore, embedder: Embedder | None = None):
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.embedder = embedder or Embedder(settings.default_embedding_model)

    async def list_items(self, tenant_id: UUID, domain_profile: str | None = None, **filters) -> list[KnowledgeItem]:
        return await self.metadata_store.list_knowledge_items(tenant_id=tenant_id, domain_profile=domain_profile, **filters)

    async def query(self, tenant_id: UUID, query: list[float], k: PositiveInt = 5, domain_profile: str | None = None) -> list[KnowledgeItem]:
        return await self.vector_store.search(query, k=k, tenant_id=tenant_id)

    async def get(self, item_id: UUID) -> KnowledgeItem | None:
        return await self.metadata_store.get_knowledge_item(item_id)

    async def query_text(self, tenant_id: UUID, query: str, k: PositiveInt = 5, domain_profile: str | None = None) -> list[KnowledgeItem]:
        try:
            res = await self.embedder.embed_query(query)
            emb = list(res.embeddings[0])
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e
        return await self.query(tenant_id, emb, k=k, domain_profile=domain_profile)
