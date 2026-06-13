import io
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import IngestionContext, KnowledgeItem, ResolvedResource, StepStatus

RESOURCES_DIR = Path(__file__).parent.parent / "resources"


@pytest.mark.asyncio
async def test_text_agent_encodes_text_bytes_and_recomputes_content_type():
    res = ResolvedResource(uri="data:text/plain,hello", content=b"hello")
    ctx = IngestionContext(resource=res, items=[KnowledgeItem(content=b"world")])
    result = await TextAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.resource.content == b"hello"
    assert ctx.resource.content_type.startswith("text/")
    assert ctx.items[0].content == b"world"
    assert ctx.items[0].content_type.startswith("text/")


@pytest.mark.parametrize("uri, parser, expected", [
    (str(RESOURCES_DIR / "sample.mp3"), "parse_audio", "audio"),
    (str(RESOURCES_DIR / "when-tomorrow-was-yesterday.pdf"), "parse_document", "document"),
    ("/tmp/sample.txt", "parse_text", "text"),
])
@pytest.mark.asyncio
async def test_text_agent_parse_routes_by_mime_type(uri, parser, expected, monkeypatch):
    mock_parse = AsyncMock(return_value=("", {}))
    monkeypatch.setattr(f"omni_ingest.agent.modality.{parser}", mock_parse)

    ctx = IngestionContext(resource=ResolvedResource(uri=uri, content=b""))
    _, metadata = await TextAgent().parse(ctx, uri, io.BytesIO())

    assert metadata["selected_parser"] == expected
    mock_parse.assert_awaited_once()


@pytest.mark.asyncio
async def test_text_agent_parse_rejects_unsupported_mime_type():
    with pytest.raises(ValueError, match="Unsupported data URI MIME type: 'font/ttf'"):
        ctx = IngestionContext(resource=ResolvedResource(uri="/tmp/sample.ttf", content=b""))
        await TextAgent().parse(ctx, "/tmp/sample.ttf", io.BytesIO())
