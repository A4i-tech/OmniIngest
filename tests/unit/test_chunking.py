import io

import pytest
from pypdf import PdfReader, PdfWriter

from omni_ingest.agent.chunking import PageChunkingAgent, SentenceChunkingAgent
from omni_ingest.core.model import ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext


@pytest.mark.asyncio
async def test_chunking_with_default_params():
    agent = SentenceChunkingAgent()

    assert agent.chunk_size == 512
    assert agent.overlap == 50

    ctx = IngestionContext(resource=ResolvedResource(uri="", content=b"Hello world. This is a test. Another sentence."))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["chunk_size"] == 512
    assert len(result.items) >= 1


@pytest.mark.asyncio
async def test_chunking_with_custom_params():
    agent = SentenceChunkingAgent(chunk_size=100, overlap=20)

    assert agent.chunk_size == 100
    assert agent.overlap == 20

    long_text = "This is a sentence. " * 20
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=long_text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] > 1


@pytest.mark.asyncio
async def test_chunking_respects_sentence_boundaries():
    agent = SentenceChunkingAgent(chunk_size=50, overlap=10)

    text = "First sentence here. Second sentence follows. Third one too."
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert not item.content.startswith(b" ")


@pytest.mark.asyncio
async def test_chunking_short_text():
    agent = SentenceChunkingAgent(chunk_size=500, overlap=50)

    short_text = "This is a short text."
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=short_text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 1
    assert result.items[0].content == short_text.encode()


@pytest.mark.asyncio
async def test_chunking_includes_position_metadata():
    agent = SentenceChunkingAgent(chunk_size=50, overlap=0)

    text = "First part. Second part. Third part."
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert "source_idx" in item.metadata
        assert "char_start" in item.metadata


@pytest.mark.asyncio
async def test_chunking_empty_text():
    agent = SentenceChunkingAgent()

    ctx = IngestionContext(resource=ResolvedResource(uri="", content=b""))
    result = await agent.run(ctx)
    assert result.status == StepStatus.SKIPPED


@pytest.mark.asyncio
async def test_chunking_whitespace_normalization():
    agent = SentenceChunkingAgent()

    text = "Hello    world.\n\n\nThis   has   extra    spaces."
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert b"  " not in item.content


@pytest.mark.asyncio
async def test_page_chunking_pdf_mode_emits_single_page_pdfs():
    with PdfWriter() as writer, io.BytesIO() as pdf:
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.write(pdf)
        ctx = IngestionContext(resource=ResolvedResource(uri="", content=pdf.getvalue()))

    result = await PageChunkingAgent(mode="pdf").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata == {"page_count": 2, "mode": "pdf"}
    assert [item.metadata["page"] for item in result.items] == [1, 2]
    assert all(item.metadata["content_type"] == "application/pdf" for item in result.items)
    assert all(len(PdfReader(io.BytesIO(item.content)).pages) == 1 for item in result.items)
