"""Implementations for storing and querying high-dimensional vector embeddings."""

import json
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
from pydantic import PositiveInt
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from ..core.model import ByteContent, KnowledgeItem
from ..core.protocol import VectorStore

DOCUMENT_ID_FIELD = "id"
DOCUMENT_VECTOR_FIELD = "vector"
DOCUMENT_CONTENT_FIELD = "content"
DOCUMENT_TENANT_ID_FIELD = "tenant_id"
DOCUMENT_SOURCE_URI_FIELD = "source_uri"
DOCUMENT_METADATA_FIELD = "metadata_json"


class LanceDBVectorStore(VectorStore):
    """
    Vector storage and semantic search powered by LanceDB.
    """

    def __init__(self, table_name: str, db_path: str | None = None, vector_size: PositiveInt = 1536):
        self.table_name, self.vector_size = table_name, vector_size
        self.db_path = str(Path(db_path or "./lancedb_data").resolve())
        self._db, self._table = None, None

    async def __aenter__(self):
        await self._get_table()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _get_table(self) -> lancedb.Table:
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

    async def upsert(self, items: list[KnowledgeItem], ctx: Any) -> None:
        t, data = await self._get_table(), []
        for i in items:
            if emb := i.metadata.get("embedding"):
                data.append(
                    {
                        "id": str(i.id),
                        "vector": emb,
                        "content": await i.content.decode(ctx),
                        "tenant_id": "" if i.tenant_id is None else str(i.tenant_id),
                        "source_uri": i.source_uri or "",
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
        items: list[KnowledgeItem] = []
        for row in res:
            tenant_value = row.get("tenant_id")
            source_uri = row.get("source_uri")
            metadata = row["metadata"]
            items.append(
                KnowledgeItem(
                    id=UUID(str(row["id"])),
                    content=ByteContent.from_bytes(str(row["content"] or "").encode(), content_type="text/plain"),
                    tenant_id=None if not tenant_value else UUID(str(tenant_value)),
                    source_uri=None if not source_uri else str(source_uri),
                    metadata=json.loads(metadata) if isinstance(metadata, str) else dict(metadata),
                )
            )
        return items

    async def count(self) -> int:
        return int(len(await self._get_table()))

    async def close(self):
        pass

    @property
    def kind(self) -> str:
        return "lancedb"

    @property
    def space(self) -> str:
        return self.table_name


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

    @property
    def kind(self) -> str:
        return "in_memory"


class QdrantVectorStore(VectorStore):
    """
    Production-grade vector search using Qdrant.
    """

    def __init__(self, col: str, url: str | None = None, key: str | None = None, size: PositiveInt = 1536):
        self.col, self.size = col, size
        self._client = AsyncQdrantClient(
            url=url or "http://localhost:6333",
            api_key=key,
        )
        self._init = False

    async def __aenter__(self):
        await self._ensure()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _ensure(self):
        if not self._init:
            if self.col not in [c.name for c in (await self._client.get_collections()).collections]:
                await self._client.create_collection(self.col, VectorParams(size=self.size, distance=Distance.COSINE))
            self._init = True

    async def upsert(self, items: list[KnowledgeItem], ctx: Any) -> None:
        await self._ensure()
        pts = [
            PointStruct(
                id=str(i.id),
                vector=i.metadata["embedding"],
                payload={
                    DOCUMENT_CONTENT_FIELD: await i.content.decode(ctx),
                    DOCUMENT_TENANT_ID_FIELD: "" if i.tenant_id is None else str(i.tenant_id),
                    DOCUMENT_SOURCE_URI_FIELD: i.source_uri or "",
                    DOCUMENT_METADATA_FIELD: json.dumps(i.metadata),
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
        items: list[KnowledgeItem] = []
        for row in res:
            tenant_value = row.payload.get(DOCUMENT_TENANT_ID_FIELD)
            source_uri = row.payload.get(DOCUMENT_SOURCE_URI_FIELD)
            metadata = row.payload.get(DOCUMENT_METADATA_FIELD, "{}")
            items.append(
                KnowledgeItem(
                    id=UUID(str(row.id)),
                    content=ByteContent.from_bytes(str(row.payload.get(DOCUMENT_CONTENT_FIELD, "") or "").encode(), content_type="text/plain"),
                    tenant_id=None if not tenant_value else UUID(str(tenant_value)),
                    source_uri=None if not source_uri else str(source_uri),
                    metadata=json.loads(metadata) if isinstance(metadata, str) else dict(metadata),
                )
            )
        return items

    async def count(self) -> int:
        await self._ensure()
        return (await self._client.get_collection(self.col)).points_count or 0

    async def close(self):
        await self._client.close()

    @property
    def kind(self) -> str:
        return "qdrant"

    @property
    def space(self) -> str:
        return self.col


class MongoVectorStore(VectorStore):
    """
    MongoDB Atlas vector search backed by a pre-provisioned search index.
    """

    def __init__(self, uri: str, database_name: str, collection_name: str, index_name: str, num_candidates: PositiveInt = 100):
        try:
            from pymongo import AsyncMongoClient
        except ModuleNotFoundError as exc:
            raise ImportError("MongoVectorStore requires pymongo. Install omni-ingest[storage] or add pymongo to the environment.") from exc

        self.database_name = database_name
        self.collection_name = collection_name
        self.index_name = index_name
        self.num_candidates = num_candidates
        self._client: Any = AsyncMongoClient(uri)
        self._collection = self._client[self.database_name][self.collection_name]

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def upsert(self, items: list[KnowledgeItem], ctx: Any) -> None:
        try:
            from pymongo import ReplaceOne
        except ModuleNotFoundError as exc:
            raise ImportError("MongoVectorStore requires pymongo. Install omni-ingest[storage] or add pymongo to the environment.") from exc

        operations = []
        for item in items:
            embedding = item.metadata.get("embedding")
            if embedding is None:
                continue
            operations.append(
                ReplaceOne(
                    {DOCUMENT_ID_FIELD: str(item.id)},
                    {
                        DOCUMENT_ID_FIELD: str(item.id),
                        DOCUMENT_VECTOR_FIELD: embedding,
                        DOCUMENT_CONTENT_FIELD: await item.content.decode(ctx),
                        DOCUMENT_TENANT_ID_FIELD: "" if item.tenant_id is None else str(item.tenant_id),
                        DOCUMENT_SOURCE_URI_FIELD: item.source_uri or "",
                        DOCUMENT_METADATA_FIELD: json.dumps(item.metadata),
                    },
                    upsert=True,
                )
            )

        if operations:
            await self._collection.bulk_write(operations, ordered=False)

    async def search(self, query: list[float], k: PositiveInt = 5, tenant_id: UUID | None = None) -> list[KnowledgeItem]:
        vector_stage: dict[str, Any] = {
            "index": self.index_name,
            "path": DOCUMENT_VECTOR_FIELD,
            "queryVector": query,
            "numCandidates": max(int(self.num_candidates), int(k)),
            "limit": int(k),
        }
        if tenant_id is not None:
            vector_stage["filter"] = {DOCUMENT_TENANT_ID_FIELD: str(tenant_id)}

        cursor = await self._collection.aggregate([
            {"$vectorSearch": vector_stage},
            {
                "$project": {
                    DOCUMENT_ID_FIELD: 1,
                    DOCUMENT_CONTENT_FIELD: 1,
                    DOCUMENT_TENANT_ID_FIELD: 1,
                    DOCUMENT_SOURCE_URI_FIELD: 1,
                    DOCUMENT_METADATA_FIELD: 1,
                }
            },
        ])
        results: list[KnowledgeItem] = []
        async for document in cursor:
            tenant_value = document.get(DOCUMENT_TENANT_ID_FIELD)
            source_uri = document.get(DOCUMENT_SOURCE_URI_FIELD)
            metadata = document.get(DOCUMENT_METADATA_FIELD, "{}")
            results.append(
                KnowledgeItem(
                    id=UUID(str(document[DOCUMENT_ID_FIELD])),
                    content=ByteContent.from_bytes(str(document.get(DOCUMENT_CONTENT_FIELD, "") or "").encode(), content_type="text/plain"),
                    tenant_id=None if not tenant_value else UUID(str(tenant_value)),
                    source_uri=None if not source_uri else str(source_uri),
                    metadata=json.loads(metadata) if isinstance(metadata, str) else dict(metadata),
                )
            )
        return results

    async def count(self) -> int:
        return await self._collection.count_documents({})

    async def close(self):
        await self._client.close()

    @property
    def kind(self) -> str:
        return "mongo"

    @property
    def space(self) -> str:
        return f"{self.database_name}.{self.collection_name}"


class AzureVectorSearchStore(VectorStore):
    """
    Azure AI Search vector retrieval backed by a pre-created index.
    """

    def __init__(self, endpoint: str, api_key: str, index_name: str,):
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents.aio import SearchClient
        except ModuleNotFoundError as exc:
            raise ImportError(
                "AzureVectorSearchStore requires azure-search-documents. Install omni-ingest[storage] or add azure-search-documents to the environment."
            ) from exc

        self.index_name = index_name
        self._client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def upsert(self, items: list[KnowledgeItem], ctx: Any) -> None:
        documents = []
        for item in items:
            embedding = item.metadata.get("embedding")
            if embedding is None:
                continue
            documents.append(
                {
                    DOCUMENT_ID_FIELD: str(item.id),
                    DOCUMENT_VECTOR_FIELD: embedding,
                    DOCUMENT_CONTENT_FIELD: await item.content.decode(ctx),
                    DOCUMENT_TENANT_ID_FIELD: "" if item.tenant_id is None else str(item.tenant_id),
                    DOCUMENT_SOURCE_URI_FIELD: item.source_uri or "",
                    DOCUMENT_METADATA_FIELD: json.dumps(item.metadata),
                }
            )

        if documents:
            await self._client.merge_or_upload_documents(documents)

    async def search(self, query: list[float], k: PositiveInt = 5, tenant_id: UUID | None = None) -> list[KnowledgeItem]:
        try:
            from azure.search.documents.models import VectorizedQuery
        except ModuleNotFoundError as exc:
            raise ImportError(
                "AzureVectorSearchStore requires azure-search-documents. Install omni-ingest[storage] or add azure-search-documents to the environment."
            ) from exc

        vector_query = VectorizedQuery(vector=query, k_nearest_neighbors=int(k), fields=DOCUMENT_VECTOR_FIELD)
        filter_expression = None
        if tenant_id is not None:
            filter_expression = f"{DOCUMENT_TENANT_ID_FIELD} eq '{tenant_id}'"

        results = await self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expression,
            select=[
                DOCUMENT_ID_FIELD,
                DOCUMENT_CONTENT_FIELD,
                DOCUMENT_TENANT_ID_FIELD,
                DOCUMENT_SOURCE_URI_FIELD,
                DOCUMENT_METADATA_FIELD,
            ],
            top=int(k),
        )
        items: list[KnowledgeItem] = []
        async for document in results:
            tenant_value = document.get(DOCUMENT_TENANT_ID_FIELD)
            source_uri = document.get(DOCUMENT_SOURCE_URI_FIELD)
            metadata = document.get(DOCUMENT_METADATA_FIELD, "{}")
            items.append(
                KnowledgeItem(
                    id=UUID(str(document[DOCUMENT_ID_FIELD])),
                    content=ByteContent.from_bytes(str(document.get(DOCUMENT_CONTENT_FIELD, "") or "").encode(), content_type="text/plain"),
                    tenant_id=None if not tenant_value else UUID(str(tenant_value)),
                    source_uri=None if not source_uri else str(source_uri),
                    metadata=json.loads(metadata) if isinstance(metadata, str) else dict(metadata),
                )
            )
        return items

    async def count(self) -> int:
        return await self._client.get_document_count()

    async def close(self):
        await self._client.close()

    @property
    def kind(self) -> str:
        return "azure_vector_search"

    @property
    def space(self) -> str:
        return self.index_name


class ChromaVectorStore(VectorStore):
    """
    Persistent ChromaDB storage for local vector retrieval.
    """

    def __init__(self, collection_name: str, path: str):
        try:
            import chromadb
        except ModuleNotFoundError as exc:
            raise ImportError("ChromaVectorStore requires chromadb. Install omni-ingest[storage] or add chromadb to the environment.") from exc

        self.collection_name = collection_name
        self.path = str(Path(path).resolve())
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection: Any = self._client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def upsert(self, items: list[KnowledgeItem], ctx: Any) -> None:
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str]] = []

        for item in items:
            embedding = item.metadata.get("embedding")
            if embedding is None:
                continue
            ids.append(str(item.id))
            documents.append(await item.content.decode(ctx))
            embeddings.append(embedding)
            metadatas.append(
                {
                    DOCUMENT_TENANT_ID_FIELD: "" if item.tenant_id is None else str(item.tenant_id),
                    DOCUMENT_SOURCE_URI_FIELD: item.source_uri or "",
                    DOCUMENT_METADATA_FIELD: json.dumps(item.metadata),
                }
            )

        if ids:
            self._collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    async def search(self, query: list[float], k: PositiveInt = 5, tenant_id: UUID | None = None) -> list[KnowledgeItem]:
        where = None
        if tenant_id is not None:
            where = {DOCUMENT_TENANT_ID_FIELD: str(tenant_id)}

        result = self._collection.query(query_embeddings=[query], n_results=int(k), where=where)
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        items: list[KnowledgeItem] = []
        for item_id, content, metadata in zip(ids, documents, metadatas, strict=True):
            tenant_value = metadata.get(DOCUMENT_TENANT_ID_FIELD, "")
            source_uri = metadata.get(DOCUMENT_SOURCE_URI_FIELD, "")
            raw_metadata = metadata.get(DOCUMENT_METADATA_FIELD, "{}")
            items.append(
                KnowledgeItem(
                    id=UUID(str(item_id)),
                    content=ByteContent.from_bytes(str(content or "").encode(), content_type="text/plain"),
                    tenant_id=None if not tenant_value else UUID(str(tenant_value)),
                    source_uri=None if not source_uri else str(source_uri),
                    metadata=json.loads(raw_metadata) if isinstance(raw_metadata, str) else dict(raw_metadata),
                )
            )
        return items

    async def count(self) -> int:
        return int(self._collection.count())

    async def close(self):
        pass

    @property
    def kind(self) -> str:
        return "chroma"

    @property
    def space(self) -> str:
        return self.collection_name
