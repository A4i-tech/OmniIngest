"""Agents for generating embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jq
from pydantic import BaseModel, Field
from pydantic_ai.embeddings import Embedder, KnownEmbeddingModelName

from ..core.config import settings
from ..core.model import JqExpression, Step, StepResult, StepStatus
from .enrichment import _assign, _doc

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

class EmbeddingAgent(BaseModel, Step):
    """
    Generates vector embeddings for KnowledgeItems to facilitate semantic search.

    This agent converts textual content into numerical vector representations
    (embeddings). These vectors enable similarity-based retrieval operations
    within a vector database.
    """

    model: KnownEmbeddingModelName | str = Field(default=settings.default_embedding_model, description="Model to use for embedding")
    foreach: JqExpression = Field(default=".items[]", description="jq expression selecting items to embed")
    input: JqExpression = Field(default=".text", description="jq expression selecting embedding input")
    path: JqExpression = Field(default=".metadata.embedding", description="jq path where embedding output is written")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to embed")

        doc = await _doc(ctx, text=True)
        item_by_id = {str(item.id): item for item in ctx.items}
        selected = jq.compile(self.foreach).input_value(doc).all()
        if not selected:
            return StepResult(status=StepStatus.SKIPPED, error="No items selected for embedding")
        values = [jq.compile(self.input).input_value(item).first() for item in selected]
        if any(not isinstance(value, str) for value in values):
            raise ValueError("Embedding input must resolve to text")
        res = await Embedder(self.model).embed_documents(values)
        for item_doc, embedding in zip(selected, res.embeddings, strict=True):
            item = item_by_id[item_doc["id"]]
            item.metadata = _assign({"metadata": item.metadata}, self.path, embedding)["metadata"]

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"embedded_count": len(selected), "path": self.path})


class VectorStoreSinkAgent(BaseModel, Step):
    """
    Writes embedded KnowledgeItems to the configured vector store.

    This agent reads the backend from ``Settings.vector_store``. A profile
    never names a backend, so the same profile writes to whatever backend a
    deployment configures. This agent writes through the ``VectorStore``
    port in ``port/vector_store.py``. It never calls a specific backend
    client directly.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to write")

        items = [item for item in ctx.items if item.metadata.get("embedding")]
        if not items:
            return StepResult(status=StepStatus.SKIPPED, error="No items carry an embedding")

        await ctx.vector_store.upsert(items, ctx)
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"written_count": len(items)})
