"""Output abstractions for finalized ingestion contexts."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from base64 import b64encode
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..agent.deduplication import DeduplicationStrategyConfig, NormalizedTextHashDeduplicationStrategy
from .model import ByteContent, ResolvedResource, StepStatus
from .protocol import GraphStore, VectorStore

if TYPE_CHECKING:
    from .pipeline import IngestionContext


def _json_dump_content(obj: ByteContent) -> dict[str, Any]:
    data = obj.model_dump(mode="json", exclude={"content"})
    try:
        data["content"] = obj.content.decode(obj.content_encoding)
    except (LookupError, UnicodeDecodeError):
        data["content"] = b64encode(obj.content).decode("ascii")
        data["content_encoding"] = "base64"
    return data


class OutputResult(BaseModel):
    status: StepStatus = Field(...)
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Output(ABC):
    @abstractmethod
    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        ...


class JsonFileOutput(Output):
    """Writes finalized ingestion output to a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        self.path.write_text(json.dumps({
            "resource": _json_dump_content(ctx.resource),
            "metadata": ctx.metadata,
            "items": [_json_dump_content(item) for item in ctx.items],
        }, indent=2))
        return OutputResult(status=StepStatus.SUCCESS, output_paths={"json": str(self.path)})


class VectorStoreOutput(Output):
    """Writes finalized knowledge items to a vector store.

    Attributes:
        store: The physical database where data will be saved
        deduplication_strategy: Strategy used to skip duplicate chunks before indexing
    """

    def __init__(self, store: VectorStore, deduplication_strategy: DeduplicationStrategyConfig | None = None) -> None:
        self.store = store
        self.deduplication_strategy = deduplication_strategy or NormalizedTextHashDeduplicationStrategy()


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


class GraphStoreOutput(Output):
    """Writes finalized graph data to a graph store.

    Attributes:
        store: The GraphStore implementation to use
    """

    def __init__(self, store: GraphStore) -> None:
        self.store = store


    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        kg_data = ctx.metadata.get("kg_data")
        if not kg_data or not ctx.items:
            return OutputResult(status=StepStatus.SKIPPED, error="No KG data found in context")

        if nodes := kg_data.get("nodes"):
            await self.store.upsert_nodes(nodes)
        if edges := kg_data.get("edges"):
            await self.store.upsert_edges(edges)
        return OutputResult(status=StepStatus.SUCCESS, metadata={"graph_nodes": len(kg_data.get("nodes", []))})
