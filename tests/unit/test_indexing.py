import pytest
from pydantic_ai.embeddings import Embedder, TestEmbeddingModel

import omni_ingest.agent.indexing as indexing
from omni_ingest.agent.indexing import EmbeddingAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext


@pytest.mark.asyncio
async def test_embedding_agent_uses_jq_selection_input_and_path(monkeypatch):
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")))
    ctx.items = [
        KnowledgeItem(content=ByteContent.from_bytes(b"keep"), metadata={"kind": "paragraph"}),
        KnowledgeItem(content=ByteContent.from_bytes(b"skip"), metadata={"kind": "image"}),
    ]
    embedder = Embedder(TestEmbeddingModel(dimensions=2))
    monkeypatch.setattr(indexing, "Embedder", lambda _model: embedder)
    agent = EmbeddingAgent(
        foreach='.items[] | select(.metadata.kind == "paragraph")',
        input=".text",
        path=".metadata.vector",
    )

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert len(ctx.items[0].metadata["vector"]) == 2
    assert "vector" not in ctx.items[1].metadata
