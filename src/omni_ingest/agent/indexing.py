"""Agents for generating embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai.embeddings import Embedder, KnownEmbeddingModelName

from ..core.config import settings

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import Step, StepResult, StepStatus


class EmbeddingAgent(BaseModel, Step):
    """
    Generates vector embeddings for KnowledgeItems to facilitate semantic search.

    This agent converts textual content into numerical vector representations
    (embeddings). These vectors enable similarity-based retrieval operations
    within a vector database.
    """

    model: KnownEmbeddingModelName | str = Field(default=settings.default_embedding_model, description="Model to use for embedding")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to embed")

        texts = [await i.decode(ctx) for i in ctx.items]
        res = await Embedder(self.model).embed_documents(texts)
        for item, emb in zip(ctx.items, res.embeddings, strict=True):
            item.metadata["embedding"] = emb

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"embedded_count": len(ctx.items)})
