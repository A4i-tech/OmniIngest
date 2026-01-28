"""Agents for generating embeddings and persisting data to vector and graph stores."""

import json
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, PrivateAttr

from ..core.config import settings
from ..core.model import IngestionContext, Step, StepResult, StepStatus, handle_step_error
from ..core.pipeline import step


@step("embedding")
class EmbeddingAgent(BaseModel, Step):
    """
    Generates vector embeddings for KnowledgeItems to facilitate semantic search.

    This agent converts textual content into numerical vector representations
    (embeddings). These vectors enable similarity-based retrieval operations
    within a vector database.
    """

    model_config = {"arbitrary_types_allowed": True}
    provider: str = Field("openai", description="The AI service used to generate vectors (e.g., 'openai')")
    model: str = Field("text-embedding-3-small", description="The specific mathematical model to use")

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            if self.provider == "azure":
                key = settings.azure_openai_api_key.get_secret_value() if settings.azure_openai_api_key else None
                self._client = AsyncAzureOpenAI(
                    api_key=key,
                    api_version="2024-02-01",
                    azure_endpoint=str(settings.azure_openai_endpoint) if settings.azure_openai_endpoint else "",
                )
            else:
                key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
                self._client = AsyncOpenAI(api_key=key)
        return self._client

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to embed")

        client = self._get_client()
        texts = [i.content for i in ctx.items]
        res = await client.embeddings.create(model=self.model, input=texts)
        for item, emb in zip(ctx.items, res.data, strict=True):
            item.metadata["embedding"] = emb.embedding

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"embedded_count": len(ctx.items)})


@step("indexing")
class VectorIndexingAgent(BaseModel, Step):
    """
    Persists KnowledgeItems into a vector storage system.

    This agent is responsible for the upsertion of processed and enriched
    data into a configured vector database, ensuring that information
    is indexed and available for retrieval by AI applications.
    """

    model_config = {"arbitrary_types_allowed": True}
    store: Any = Field(..., description="The physical database where data will be saved")
    vector_store_kind: str = Field("lancedb", description="The type of database (e.g., 'lancedb' or 'qdrant')")
    vector_store_space: str = Field("default", description="The 'room' or collection name where data lives")

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items")
        await self.store.upsert(ctx.items)
        if ctx.store and (pid := ctx.metadata.get("knowledge_item_id")):
            for i in ctx.items:
                await ctx.store.create_chunk(
                    chunk_id=str(i.id),
                    knowledge_item_id=pid,
                    tenant_id=ctx.tenant_id,
                    text=i.content,
                    vector_store_kind=self.vector_store_kind,
                    vector_store_space=self.vector_store_space,
                    vector_store_ref_id=str(i.id),
                    metadata=i.metadata,
                )
        return StepResult(status=StepStatus.SUCCESS, metadata={"count": len(ctx.items)})


@step("keyword_indexing")
class KeywordIndexingAgent(BaseModel, Step):
    """
    Identifies and indexes searchable keywords from KnowledgeItems.

    This agent utilizes an LLM to extract primary keywords and phrases
    from text content, facilitating keyword-based search and filtering
    operations.
    """

    model_config = {"arbitrary_types_allowed": True}
    provider: str = Field("openai", description="The AI service to use")
    model: str = Field("gpt-4o", description="The AI model version used for keyword extraction")

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            if self.provider == "azure":
                key = settings.azure_openai_api_key.get_secret_value() if settings.azure_openai_api_key else None
                self._client = AsyncAzureOpenAI(
                    api_key=key,
                    api_version="2024-02-01",
                    azure_endpoint=str(settings.azure_openai_endpoint) if settings.azure_openai_endpoint else "",
                )
            else:
                key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
                self._client = AsyncOpenAI(api_key=key)
        return self._client

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for keywords")

        client = self._get_client()
        prompt_tmpl = "Extract 5-10 descriptive keywords from the text. Return JSON array of strings.\n\nContent: {text}"

        for item in ctx.items:
            res = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_tmpl.format(text=item.content)}],
                response_format={"type": "json_object"},
            )
            data = json.loads(res.choices[0].message.content)
            keywords = data.get("keywords") or list(data.values())[0]
            if isinstance(keywords, list):
                item.metadata["keywords"] = list(set(item.metadata.get("keywords", []) + keywords))

        return StepResult(status=StepStatus.SUCCESS, metadata={"items_processed": len(ctx.items)})


@step("graph_indexing")
class GraphIndexingAgent(BaseModel, Step):
    """
    Indexes extracted knowledge graph data into a graph database.
    """

    graph_store: Any = Field(..., description="The GraphStore implementation to use")

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        kg_data = ctx.metadata.get("kg_data")
        if not kg_data or not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No KG data found in context")

        if nodes := kg_data.get("nodes"):
            await self.graph_store.upsert_nodes(nodes)
        if edges := kg_data.get("edges"):
            await self.graph_store.upsert_edges(edges)

        for item in ctx.items:
            item.structured_views = item.structured_views or {}
            item.structured_views["knowledge_graph"] = kg_data

        return StepResult(status=StepStatus.SUCCESS, metadata={"graph_nodes": len(kg_data.get("nodes", []))})
