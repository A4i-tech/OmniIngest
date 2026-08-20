from unittest.mock import AsyncMock

import pytest

import omni_ingest.agent.modality as modality
from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext


@pytest.mark.asyncio
async def test_text_agent_encodes_text_bytes_and_recomputes_content_type():
    res = ResolvedResource(uri="data:text/plain,hello", raw_content=b"hello")
    ctx = IngestionContext(resource=res, items=[KnowledgeItem(raw_content=b"world")])
    result = await TextAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert await ctx.resource.content(ctx) == b"hello"
    assert (await ctx.resource.content_type(ctx)).startswith("text/")
    assert await ctx.items[0].content(ctx) == b"world"
    assert (await ctx.items[0].content_type(ctx)).startswith("text/")


@pytest.mark.asyncio
async def test_text_agent_replaces_cached_content_type(monkeypatch):
    ctx = IngestionContext(resource=ResolvedResource(uri="sample.webp", raw_content=b"image"))
    item = KnowledgeItem(raw_content=b"image", metadata={"content_type": "image/webp"})

    async def parse(*_args):
        return "shelf", {"selected_parser": "image"}

    monkeypatch.setattr(TextAgent, "parse", parse)
    await TextAgent().convert(ctx, item, "sample.webp")

    assert await item.decode(ctx) == "shelf"
    assert await item.content_type(ctx) == "text/plain"


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
    ctx = IngestionContext(resource=ResolvedResource(uri=uri, raw_content=b""))
    obj = ByteContent.from_bytes(b"", content_type=content_type)

    _, metadata = await TextAgent().parse(ctx, obj, uri)

    assert metadata["selected_parser"] == expected
    assert called


@pytest.mark.asyncio
async def test_text_agent_parse_rejects_unsupported_mime_type():
    with pytest.raises(ValueError, match="Unsupported data URI MIME type: 'font/ttf'"):
        ctx = IngestionContext(resource=ResolvedResource(uri="/tmp/sample.ttf", raw_content=b""))
        await TextAgent().parse(ctx, ByteContent.from_bytes(b"", content_type="font/ttf"), "/tmp/sample.ttf")


@pytest.mark.asyncio
async def test_text_agent_translates_audio_when_dst_set(monkeypatch):
    mock_parse_audio = AsyncMock(return_value=("hola mundo", {"source_text": "hello world"}))
    monkeypatch.setattr("omni_ingest.agent.modality.parse_audio", mock_parse_audio)

    item = KnowledgeItem(raw_content=b"fake-audio-bytes", metadata={"content_type": "audio/mpeg"})
    ctx = IngestionContext(resource=ResolvedResource(uri="sample.mp3", raw_content=b""), items=[item])

    result = await TextAgent(dst="es").run(ctx)

    assert result.status == StepStatus.SUCCESS
    mock_parse_audio.assert_awaited_once()
    assert mock_parse_audio.call_args.args[0] == "sample.mp3"
    assert mock_parse_audio.call_args.kwargs == {"src": None, "dst": "es"}
    assert await item.decode(ctx) == "hola mundo"
    assert item.metadata["source_text"] == "hello world"


@pytest.mark.asyncio
async def test_text_agent_transcribes_audio_without_dst(monkeypatch):
    mock_parse_audio = AsyncMock(return_value=("hello world", {}))
    monkeypatch.setattr("omni_ingest.agent.modality.parse_audio", mock_parse_audio)

    item = KnowledgeItem(raw_content=b"fake-audio-bytes", metadata={"content_type": "audio/mpeg"})
    ctx = IngestionContext(resource=ResolvedResource(uri="sample.mp3", raw_content=b""), items=[item])

    result = await TextAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    mock_parse_audio.assert_awaited_once()
    assert mock_parse_audio.call_args.args[0] == "sample.mp3"
    assert mock_parse_audio.call_args.kwargs == {"src": None, "dst": None}
    assert await item.decode(ctx) == "hello world"
