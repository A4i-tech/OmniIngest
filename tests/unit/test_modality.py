import pytest

import omni_ingest.agent.modality as modality
from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext


@pytest.mark.asyncio
async def test_text_agent_encodes_text_bytes_and_recomputes_content_type():
    res = ResolvedResource(uri="data:text/plain,hello", content=ByteContent.from_bytes(b"hello"))
    ctx = IngestionContext(resource=res, items=[KnowledgeItem(content=ByteContent.from_bytes(b"world"))])
    result = await TextAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert await ctx.resource.content.read(ctx) == b"hello"
    assert (await ctx.resource.content.content_type(ctx)).startswith("text/")
    assert await ctx.items[0].content.read(ctx) == b"world"
    assert (await ctx.items[0].content.content_type(ctx)).startswith("text/")


@pytest.mark.asyncio
async def test_text_agent_replaces_cached_content_type(monkeypatch):
    ctx = IngestionContext(resource=ResolvedResource(uri="sample.webp", content=ByteContent.from_bytes(b"image")))
    item = KnowledgeItem(content=ByteContent.from_bytes(b"image", content_type="image/webp"))
    ctx.items = [item]

    async def parse(*_args):
        return "shelf", {"selected_parser": "image"}

    monkeypatch.setattr(TextAgent, "parse", parse)
    await TextAgent().run(ctx)

    assert await item.content.decode(ctx) == "shelf"
    assert await item.content.content_type(ctx) == "text/plain"
    assert item.metadata["selected_parser"] == "image"


@pytest.mark.parametrize(
    "uri, content_type, parser, expected",
    [
        ("sample.mp3", "audio/mpeg", "parse_audio", "audio"),
        ("sample.pdf", "application/pdf", "parse_document", "document"),
        ("sample.txt", "text/plain", "parse_text", "text"),
        ("sample.brf", "text/plain", "parse_brf", "brf"),
    ],
)
@pytest.mark.asyncio
async def test_text_agent_parse_routes_by_mime_type(uri, content_type, parser, expected, monkeypatch):
    called = False

    async def parse(*_args, **_kwargs):
        nonlocal called
        called = True
        return "", {}

    monkeypatch.setattr(modality, parser, parse)
    ctx = IngestionContext(resource=ResolvedResource(uri=uri, content=ByteContent.from_bytes(b"")))
    monkeypatch.setattr(ctx, "stt_factory", lambda *_: None)
    obj = ByteContent.from_bytes(b"", content_type=content_type)

    _, metadata = await TextAgent().parse(ctx, obj, uri)

    assert metadata["selected_parser"] == expected
    assert called


@pytest.mark.asyncio
async def test_text_agent_parse_rejects_unsupported_mime_type():
    with pytest.raises(ValueError, match="Unsupported data URI MIME type: 'font/ttf'"):
        ctx = IngestionContext(resource=ResolvedResource(uri="/tmp/sample.ttf", content=ByteContent.from_bytes(b"")))
        await TextAgent().parse(ctx, ByteContent.from_bytes(b"", content_type="font/ttf"), "/tmp/sample.ttf")
