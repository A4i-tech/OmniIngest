"""Output abstractions for finalized ingestion contexts."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..agent.deduplication import DeduplicationStrategyConfig, NormalizedTextHashDeduplicationStrategy
from .model import IngestionContext, ResolvedResource, StepStatus


class OutputResult(BaseModel):
    status: StepStatus = Field(...)
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: str | None = Field(None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Output(ABC):
    @abstractmethod
    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        ...


class JsonFileOutput(BaseModel, Output):
    path: Path

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        self.path.write_text(json.dumps({
            "resource": ctx.resource.model_dump(mode="json"),
            "metadata": ctx.metadata,
            "items": [item.model_dump(mode="json") for item in ctx.items],
        }, indent=2))
        return OutputResult(status=StepStatus.SUCCESS, output_paths={"json": str(self.path)})


class VectorStoreOutput(BaseModel, Output):
    store: Any = Field(..., description="The physical database where data will be saved")
    deduplication_strategy: DeduplicationStrategyConfig = Field(default_factory=NormalizedTextHashDeduplicationStrategy, description="Strategy used to skip duplicate chunks before indexing")

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        if not ctx.items:
            return OutputResult(status=StepStatus.SKIPPED, error="No items")

        deduplication_result = await self.deduplication_strategy.deduplicate(ctx.items, ctx)
        items = deduplication_result.kept_items
        if items:
            await self.store.upsert(items)
            if ctx.store and (pid := ctx.metadata.get("knowledge_item_id")):
                for item in items:
                    await ctx.store.create_chunk(
                        chunk_id=item.id,
                        knowledge_item_id=pid,
                        tenant_id=ctx.tenant_id,
                        text=item.decode(),
                        vector_store_kind=self.store.kind,
                        vector_store_space=self.store.space,
                        vector_store_ref_id=str(item.id),
                        metadata=item.metadata,
                    )

        return OutputResult(
            status=StepStatus.SUCCESS,
            metadata={
                "count": len(items),
                "input_count": deduplication_result.input_count,
                "deduplicated_count": deduplication_result.dropped_count,
                "deduplication_strategy": deduplication_result.strategy,
                "deduplication_scope": deduplication_result.scope,
            },
        )


class GraphStoreOutput(BaseModel, Output):
    store: Any = Field(..., description="The GraphStore implementation to use")

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        kg_data = ctx.metadata.get("kg_data")
        if not kg_data or not ctx.items:
            return OutputResult(status=StepStatus.SKIPPED, error="No KG data found in context")

        if nodes := kg_data.get("nodes"):
            await self.store.upsert_nodes(nodes)
        if edges := kg_data.get("edges"):
            await self.store.upsert_edges(edges)
        return OutputResult(status=StepStatus.SUCCESS, metadata={"graph_nodes": len(kg_data.get("nodes", []))})
