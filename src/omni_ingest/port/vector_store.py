"""Implementations for storing and querying high-dimensional vector embeddings."""

import json
import tempfile
from typing import Any
from uuid import UUID

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
from pydantic import PositiveInt
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from ..core.config import settings
from ..core.model import KnowledgeItem
from ..core.protocol import VectorStore


class LanceDBVectorStore(VectorStore):
    """
    Vector storage and semantic search powered by LanceDB.
    """

    def __init__(self, table_name: str, db_path: str | None = None, vector_size: PositiveInt = 1536):
        self.table_name, self.vector_size = table_name, vector_size
        self.db_path = db_path or settings.lancedb_path.resolve()
        self._db, self._table = None, None

    async def _get_table(self) -> Any:
        if self._table is None:
            db = lancedb.connect(self.db_path)
            self._table = db.create_table(self.table_name, schema=pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.vector_size)),
                pa.field("content", pa.string()),
                pa.field("tenant_id", pa.string(), nullable=True),
                pa.field("source_uri", pa.string(), nullable=True),
                pa.field("metadata", pa.string()),
            ]), exist_ok=True)
            self._db = db
        return self._table

    async def upsert(self, items: list[KnowledgeItem]) -> None:
        t, data = await self._get_table(), []
        for i in items:
            if emb := i.metadata.get("embedding"):
                data.append(
                    {
                        "id": str(i.id),
                        "vector": emb,
                        "content": i.content,
                        "tenant_id": str(i.tenant_id) if i.tenant_id else None,
                        "source_uri": i.source_uri,
                        "metadata": json.dumps(i.metadata),
                    }
                )
        if data:
            t.add(data, mode="overwrite")

    async def search(self, query: list[float], k: PositiveInt = 5, tenant_id: UUID | None = None) -> list[KnowledgeItem]:
        t = await self._get_table()
        q = t.search(query).limit(k)
        if tenant_id:
            q = q.where(f"tenant_id = '{tenant_id}'")
        res = q.to_list()
        return [
            KnowledgeItem(
                id=UUID(r["id"]),
                content=r["content"],
                tenant_id=UUID(r["tenant_id"]) if r.get("tenant_id") else None,
                source_uri=r["source_uri"],
                metadata=json.loads(r["metadata"]),
            )
            for r in res
        ]

    async def count(self) -> int:
        return int(len(await self._get_table()))

    async def close(self):
        pass


class InMemoryVectorStore(LanceDBVectorStore):
    """
    Volatile, fast vector storage suitable for testing and ephemeral processing.
    """

    def __init__(self, vector_size: PositiveInt = 1536):
        self._temp = tempfile.TemporaryDirectory()
        super().__init__(table_name="temp_vectors", db_path=self._temp.name, vector_size=vector_size)

    async def close(self):
        await super().close()
        self._temp.cleanup()


class QdrantVectorStore(VectorStore):
    """
    Production-grade vector search using Qdrant.
    """

    def __init__(self, col: str, url: str | None = None, key: str | None = None, size: PositiveInt = 1536):
        self.col, self.size = col, size
        self._client = AsyncQdrantClient(
            url=url or str(settings.qdrant_url),
            api_key=key or (settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
        )
        self._init = False

    async def _ensure(self):
        if not self._init:
            if self.col not in [c.name for c in (await self._client.get_collections()).collections]:
                await self._client.create_collection(self.col, VectorParams(size=self.size, distance=Distance.COSINE))
            self._init = True

    async def upsert(self, items: list[KnowledgeItem]) -> None:
        await self._ensure()
        pts = [
            PointStruct(
                id=i.id,
                vector=i.metadata["embedding"],
                payload={
                    "content": i.content,
                    "tenant_id": str(i.tenant_id) if i.tenant_id else None,
                    "source_uri": i.source_uri,
                    **i.metadata,
                },
            )
            for i in items
            if i.metadata.get("embedding")
        ]
        if pts:
            await self._client.upsert(self.col, pts)

    async def search(self, query: list[float], k: PositiveInt = 5, tenant_id: UUID | None = None) -> list[KnowledgeItem]:
        await self._ensure()
        filt = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id)))]) if tenant_id else None
        res = await self._client.search(self.col, query, query_filter=filt, limit=k)  # type: ignore
        return [
            KnowledgeItem(
                id=r.id,
                content=r.payload.get("content", ""),
                tenant_id=r.payload.get("tenant_id"),
                source_uri=r.payload.get("source_uri"),
                metadata={k: v for k, v in r.payload.items() if k not in ("content", "source_uri", "tenant_id")},
            )
            for r in res
        ]

    async def count(self) -> int:
        await self._ensure()
        return (await self._client.get_collection(self.col)).points_count or 0

    async def close(self):
        await self._client.close()
