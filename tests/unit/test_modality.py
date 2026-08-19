import io
from unittest.mock import AsyncMock

import pytest

from omni_ingest.agent.enrichment import TextToSpeechAgent
from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.port.tts import TtsOutputFormat


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
    agent = TextAgent()
    monkeypatch.setattr(TextAgent, "parse", AsyncMock(return_value=("shelf", {"selected_parser": "image"})))

    await agent.convert(ctx, item, "sample.webp")

    assert await item.decode(ctx) == "shelf"


@pytest.mark.parametrize("uri, content_type, parser, expected", [
    ("sample.mp3", "audio/mpeg", "parse_audio", "audio"),
    ("when-tomorrow-was-yesterday.pdf", "application/pdf", "parse_document", "document"),
    ("/tmp/sample.txt", "text/plain", "parse_text", "text"),
])
@pytest.mark.asyncio
async def test_text_agent_parse_routes_by_mime_type(uri, content_type, parser, expected, monkeypatch, resources_dir):
    uri = str(resources_dir / uri)
    mock_parse = AsyncMock(return_value=("", {}))
    monkeypatch.setattr(f"omni_ingest.agent.modality.{parser}", mock_parse)

    ctx = IngestionContext(resource=ResolvedResource(uri=uri, raw_content=b""))
    _, metadata = await TextAgent().parse(ctx, uri, io.BytesIO(), content_type=content_type)

    assert metadata["selected_parser"] == expected
    mock_parse.assert_awaited_once()


@pytest.mark.asyncio
async def test_text_agent_parse_rejects_unsupported_mime_type():
    with pytest.raises(ValueError, match="Unsupported data URI MIME type: 'font/ttf'"):
        ctx = IngestionContext(resource=ResolvedResource(uri="/tmp/sample.ttf", raw_content=b""))
        await TextAgent().parse(ctx, "/tmp/sample.ttf", io.BytesIO(), content_type="font/ttf")


@pytest.mark.parametrize("output_format, mime_type, audio_bytes", [
    (TtsOutputFormat.MP3, "audio/mpeg", b"mp3-bytes"),
    (TtsOutputFormat.WAV, "audio/wav", b"wav-bytes"),
])
@pytest.mark.asyncio
async def test_tts_agent_writes_audio_bytes_and_content_type(output_format, mime_type, audio_bytes, monkeypatch):
    mock_synthesizer = AsyncMock()
    mock_synthesizer.synthesize = AsyncMock(return_value=audio_bytes)
    monkeypatch.setattr("omni_ingest.agent.enrichment.create_synthesizer", lambda model: mock_synthesizer)

    ctx = IngestionContext(resource=ResolvedResource(uri="data:text/plain,hello", raw_content=b""), items=[KnowledgeItem(raw_content=b"hello", metadata={"content_type": "text/plain"})])
    result = await TextToSpeechAgent(format=output_format).run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.items[0].raw_content == audio_bytes
    assert ctx.items[0].metadata["content_type"] == mime_type
    mock_synthesizer.synthesize.assert_awaited_once_with("hello", voice=None, format=output_format)


@pytest.mark.asyncio
async def test_tts_agent_skips_when_no_items_selected():
    ctx = IngestionContext(resource=ResolvedResource(uri="data:text/plain,hello", raw_content=b""))
    result = await TextToSpeechAgent().run(ctx)

    assert result.status == StepStatus.SKIPPED
