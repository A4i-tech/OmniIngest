"""Output abstractions for finalized ingestion contexts."""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
from abc import ABC, abstractmethod
from base64 import b64encode
from typing import TYPE_CHECKING, Any

import fsspec  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from ..agent.deduplication import DeduplicationStrategyConfig, NormalizedTextHashDeduplicationStrategy
from .config import settings
from .model import ByteContent, ResolvedResource, StepStatus
from .protocol import GraphStore, VectorStore

if TYPE_CHECKING:
    from .pipeline import IngestionContext


async def _json_dump_content(obj: ByteContent, ctx: IngestionContext[ResolvedResource]) -> dict[str, Any]:
    data = obj.model_dump(mode="json", exclude={"raw_content"})
    content = await obj.content(ctx)
    try:
        data["content"] = content.decode(obj.content_encoding)
    except (LookupError, UnicodeDecodeError):
        data["content"] = b64encode(content).decode("ascii")
        data["content_encoding"] = "base64"
    return data


async def json_dump_context(ctx: IngestionContext[ResolvedResource]) -> dict[str, Any]:
    return {
        "resource": await _json_dump_content(ctx.resource, ctx),
        "metadata": ctx.metadata,
        "items": [await _json_dump_content(item, ctx) for item in ctx.items],
    }


class OutputResult(BaseModel):
    status: StepStatus = Field(...)
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Output(ABC):
    @abstractmethod
    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        ...

    async def write_many(self, ctxs: list[IngestionContext[ResolvedResource]]) -> OutputResult:
        result = OutputResult(status=StepStatus.SUCCESS)
        for ctx in ctxs:
            result = await self.write(ctx)
        return result


class FileOutput(Output):

    def __init__(self, path: str) -> None:
        self.path = path

    def _write(self, data: str | bytes, mode: str) -> None:
        with fsspec.open(self.path, mode) as f:
            f.write(data)


class JsonFileOutput(FileOutput):
    """Writes finalized ingestion output to a JSON file."""

    async def dump(self, ctxs: list[IngestionContext[ResolvedResource]]) -> str:
        data = await json_dump_context(ctxs[0]) if len(ctxs) == 1 else {"runs": [await json_dump_context(ctx) for ctx in ctxs]}
        return json.dumps(data, indent=2)

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        await asyncio.to_thread(self._write, await self.dump([ctx]), mode="wt")
        return OutputResult(status=StepStatus.SUCCESS, output_paths={"json": self.path})

    async def write_many(self, ctxs: list[IngestionContext[ResolvedResource]]) -> OutputResult:
        await asyncio.to_thread(self._write, await self.dump(ctxs), mode="wt")
        return OutputResult(status=StepStatus.SUCCESS, output_paths={"json": self.path})


class BrfFileOutput(JsonFileOutput):
    """Writes finalized ingestion output to a BRF (Braille Ready Format) file."""

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        json_str = await self.dump([ctx])
        # Simplification: BRF is uppercase-only computer braille ASCII; a full Grade-1/Grade-2 braille translation is out of scope for this ticket.
        brf_str = json_str.upper()
        await asyncio.to_thread(self._write, brf_str, mode="wt")
        return OutputResult(status=StepStatus.SUCCESS, output_paths={"brf": self.path})

    async def write_many(self, ctxs: list[IngestionContext[ResolvedResource]]) -> OutputResult:
        json_str = await self.dump(ctxs)
        # Simplification: BRF is uppercase-only computer braille ASCII; a full Grade-1/Grade-2 braille translation is out of scope for this ticket.
        brf_str = json_str.upper()
        await asyncio.to_thread(self._write, brf_str, mode="wt")
        return OutputResult(status=StepStatus.SUCCESS, output_paths={"brf": self.path})


class PickleFileOutput(FileOutput):
    """Writes finalized ingestion output to a pickle file."""

    async def write_many(self, ctxs: list[IngestionContext[ResolvedResource]]) -> OutputResult:
        runs: list[dict[str, Any]] = [{
            "resource": {**ctx.resource.model_dump(mode="python", exclude={"raw_content"}), "content": await ctx.resource.content(ctx)},
            "metadata": ctx.metadata,
            "items": [{**item.model_dump(mode="python", exclude={"raw_content"}), "content": await item.content(ctx)} for item in ctx.items],
        } for ctx in ctxs]
        data = runs[0] if len(runs) == 1 else {"runs": runs}
        await asyncio.to_thread(self._write, pickle.dumps(data), mode="wb")
        return OutputResult(status=StepStatus.SUCCESS, output_paths={"pkl": self.path})

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        return await self.write_many([ctx])


class StdoutOutput(Output):
    """Writes finalized ingestion output to stdout as JSON."""

    def __init__(self) -> None:
        self._inner = JsonFileOutput(os.devnull)

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        print(await self._inner.dump([ctx]), file=sys.stdout)
        return OutputResult(status=StepStatus.SUCCESS)

    async def write_many(self, ctxs: list[IngestionContext[ResolvedResource]]) -> OutputResult:
        print(await self._inner.dump(ctxs), file=sys.stdout)
        return OutputResult(status=StepStatus.SUCCESS)


class VectorStoreOutput(Output):
    """Writes finalized knowledge items to a vector store.

    Attributes:
        store: The physical database where data will be saved
        deduplication_strategy: Strategy used to skip duplicate items before indexing
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
            await self.store.upsert(items, ctx)
            if ctx.store and (pid := ctx.metadata.get("knowledge_item_id")):
                for item in items:
                    metadata = {
                        **item.metadata,
                        "kind": "indexed",
                        "parent_knowledge_item_id": str(pid),
                        "vector_store": {
                            "kind": self.store.kind,
                            "space": self.store.space,
                            "ref_id": str(item.id),
                        },
                    }
                    await ctx.store.create_knowledge_item(
                        tenant_id=ctx.tenant_id,
                        domain_profile=ctx.domain_profile,
                        modality="text",
                        source_uri=item.source_uri or ctx.resource.uri,
                        content=await item.content(ctx),
                        metadata=metadata,
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


class MongoDocumentOutput(Output):
    """Writes KnowledgeItems as flat BSON documents to a plain MongoDB collection.

    URI format:
      mongodb://host:port/database/collection
    """

    def __init__(self, uri: str) -> None:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(uri)
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) < 2:
            raise ValueError(
                f"MongoDB output URI must include database and collection in the path: "
                f"mongodb://host/database/collection — got: {uri}"
            )
        client_uri = urlunparse(parsed._replace(path=f"/{path_parts[0]}"))

        try:
            from pymongo import AsyncMongoClient, ReplaceOne
            from pymongo.uri_parser import parse_uri
            self._ReplaceOne = ReplaceOne
        except ModuleNotFoundError as exc:
            raise ImportError(
                "MongoDocumentOutput requires pymongo. Install omni-ingest[storage] or add pymongo to the environment."
            ) from exc

        if parse_uri(client_uri)["database"] is None:
            raise ValueError(f"MongoDB output URI must include a database name — got: {uri}")
        self._collection: Any = AsyncMongoClient(client_uri).get_default_database()[path_parts[1]]

    async def write(self, ctx: IngestionContext[ResolvedResource]) -> OutputResult:
        if not ctx.items:
            return OutputResult(status=StepStatus.SKIPPED, error="No items")

        ops = [
            self._ReplaceOne(
                {"_id": str(item.id)},
                {"_id": str(item.id), "source_uri": item.source_uri, "content": await item.decode(ctx), "metadata": item.metadata},
                upsert=True,
            )
            for item in ctx.items
        ]
        await self._collection.bulk_write(ops, ordered=False)
        return OutputResult(status=StepStatus.SUCCESS, metadata={"count": len(ctx.items)})


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


def parse(value: str, format: str | None = None) -> Output:
    if value.startswith(("mongodb://", "mongodb+srv://")):
        return MongoDocumentOutput(value)

    format = format or {
        "-": "json",
        "vector:default": "vector",
        "graph:default": "graph",
    }.get(value, value.rsplit(".", 1)[-1].lower())

    if format == "vector":
        return VectorStoreOutput(settings.vector_store.build())
    if format == "graph":
        return GraphStoreOutput(settings.graph_store.build())
    if format == "json":
        return StdoutOutput() if value == "-" else JsonFileOutput(value)
    if format == "pkl":
        return PickleFileOutput(value)
    if format == "brf":
        return BrfFileOutput(value)
    raise ValueError(f"Unsupported output format: {format}")
