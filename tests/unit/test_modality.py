import pytest

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import IngestionContext, KnowledgeItem, ResolvedResource, StepStatus


@pytest.mark.asyncio
async def test_text_agent_encodes_text_bytes_and_recomputes_content_type():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="data:text/plain,hello", content=b"hello"),
        items=[KnowledgeItem(content=b"world")],
    )

    result = await TextAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.resource.content == b"hello"
    assert ctx.resource.content_type.startswith("text/")
    assert ctx.items[0].content == b"world"
    assert ctx.items[0].content_type.startswith("text/")
