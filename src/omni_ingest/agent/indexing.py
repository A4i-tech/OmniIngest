"""Agents for generating embeddings and persisting data to vector and graph stores."""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.embeddings import Embedder, KnownEmbeddingModelName

from ..core.model import AgentMixin, IngestionContext, Step, StepResult, StepStatus
from ..core.pipeline import step


@step("embedding")
class EmbeddingAgent(BaseModel, Step):
    """
    Generates vector embeddings for KnowledgeItems to facilitate semantic search.

    This agent converts textual content into numerical vector representations
    (embeddings). These vectors enable similarity-based retrieval operations
    within a vector database.
    """

    model: KnownEmbeddingModelName = Field("openai:text-embedding-3-small", description="Model to use for embedding")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to embed")

        texts = [i.content for i in ctx.items]
        res = await Embedder(self.model).embed_documents(texts)
        for item, emb in zip(ctx.items, res.embeddings, strict=True):
            item.metadata["embedding"] = emb

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"embedded_count": len(ctx.items)})


@step("indexing")
class VectorIndexingAgent(BaseModel, Step):
    """
    Persists KnowledgeItems into a vector storage system.

    This agent is responsible for the upsertion of processed and enriched
    data into a configured vector database, ensuring that information
    is indexed and available for retrieval by AI applications.
    """

    store: Any = Field(..., description="The physical database where data will be saved")
    vector_store_kind: str = Field("lancedb", description="The type of database (e.g., 'lancedb' or 'qdrant')")
    vector_store_space: str = Field("default", description="The 'room' or collection name where data lives")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items")
        await self.store.upsert(ctx.items)
        if ctx.store and (pid := ctx.metadata.get("knowledge_item_id")):
            for i in ctx.items:
                await ctx.store.create_chunk(
                    chunk_id=i.id,
                    knowledge_item_id=pid,
                    tenant_id=ctx.tenant_id,
                    text=i.content,
                    vector_store_kind=self.vector_store_kind,
                    vector_store_space=self.vector_store_space,
                    vector_store_ref_id=str(i.id),
                    metadata=i.metadata,
                )
        return StepResult(status=StepStatus.SUCCESS, metadata={"count": len(ctx.items)})


class Keywords(BaseModel):
    keywords: list[str] = Field(..., description="Extracted keywords from content")


@step("keyword_indexing")
class KeywordIndexingAgent(BaseModel, Step, AgentMixin):
    """
    Identifies and indexes searchable keywords from KnowledgeItems.

    This agent utilizes an LLM to extract primary keywords and phrases
    from text content, facilitating keyword-based search and filtering
    operations.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for keywords")

        prompt_tmpl = "Extract 5-10 descriptive keywords from the text. Content: {text}"

        agent = self._agent(ctx)
        for item in ctx.items:
            res = await agent.run(prompt_tmpl.format(text=item.content), output_type=Keywords)
            keywords = res.output.keywords
            item.metadata["keywords"] = list(set(item.metadata.get("keywords", []) + keywords))

        return StepResult(status=StepStatus.SUCCESS, metadata={"items_processed": len(ctx.items)})


@step("graph_indexing")
class GraphIndexingAgent(BaseModel, Step):
    """
    Indexes extracted knowledge graph data into a graph database.
    """

    graph_store: Any = Field(..., description="The GraphStore implementation to use")

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
