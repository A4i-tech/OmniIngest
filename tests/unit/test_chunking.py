import io

import pytest
from pypdf import PdfReader, PdfWriter

from omni_ingest.agent.chunking import PageChunkingAgent, PageStitchingAgent, SentenceChunkingAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext


@pytest.mark.asyncio
async def test_chunking_with_default_params():
    agent = SentenceChunkingAgent()

    assert agent.chunk_size == 512
    assert agent.overlap == 50

    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b"Hello world. This is a test. Another sentence."))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["chunk_size"] == 512
    assert len(result.items) >= 1


@pytest.mark.asyncio
async def test_byte_content_resolves_inline_content():
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b"context"))
    item = KnowledgeItem(raw_content=b"inline")

    assert await item.content(ctx) == b"inline"
    assert await item.decode(ctx) == "inline"


@pytest.mark.asyncio
async def test_byte_content_rejects_unknown_omni_ingest_route():
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b"context"))
    item = KnowledgeItem(content_uri="omni-ingest://resource/unknown/1", metadata={"content_type": "application/pdf"})

    with pytest.raises(ValueError, match="Unsupported content URI"):
        await item.content(ctx)


@pytest.mark.asyncio
async def test_chunking_with_custom_params():
    agent = SentenceChunkingAgent(chunk_size=100, overlap=20)

    assert agent.chunk_size == 100
    assert agent.overlap == 20

    long_text = "This is a sentence. " * 20
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=long_text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] > 1


@pytest.mark.asyncio
async def test_chunking_respects_sentence_boundaries():
    agent = SentenceChunkingAgent(chunk_size=50, overlap=10)

    text = "First sentence here. Second sentence follows. Third one too."
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert not (await item.content(ctx)).startswith(b" ")


@pytest.mark.asyncio
async def test_chunking_short_text():
    agent = SentenceChunkingAgent(chunk_size=500, overlap=50)

    short_text = "This is a short text."
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=short_text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 1
    assert await result.items[0].content(ctx) == short_text.encode()


@pytest.mark.asyncio
async def test_chunking_includes_position_metadata():
    agent = SentenceChunkingAgent(chunk_size=50, overlap=0)

    text = "First part. Second part. Third part."
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert "source_idx" in item.metadata
        assert "char_start" in item.metadata


@pytest.mark.asyncio
async def test_chunking_empty_text():
    agent = SentenceChunkingAgent()

    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""))
    result = await agent.run(ctx)
    assert result.status == StepStatus.SKIPPED


@pytest.mark.asyncio
async def test_chunking_whitespace_normalization():
    agent = SentenceChunkingAgent()

    text = "Hello    world.\n\n\nThis   has   extra    spaces."
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=text.encode()))
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert b"  " not in await item.content(ctx)


@pytest.mark.asyncio
async def test_page_chunking_pdf_mode_emits_virtual_page_items():
    with PdfWriter() as writer, io.BytesIO() as pdf:
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.write(pdf)
        ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=pdf.getvalue()))

    result = await PageChunkingAgent(mode="pdf").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata == {"page_count": 2, "mode": "pdf"}
    assert [item.metadata["page"] for item in result.items] == [1, 2]
    assert all(item.metadata["content_type"] == "application/pdf" for item in result.items)
    assert [item.content_uri for item in result.items] == ["omni-ingest://resource/page/1", "omni-ingest://resource/page/2"]
    assert all(item.raw_content == b"" for item in result.items)
    assert all([len(PdfReader(io.BytesIO(await item.content(ctx))).pages) == 1 for item in result.items])


@pytest.mark.asyncio
async def test_page_stitching_agent_combines_selected_page_items():
    items = []
    for page in range(1, 4):
        with PdfWriter() as writer, io.BytesIO() as pdf:
            writer.add_blank_page(width=72, height=72)
            writer.write(pdf)
            items.append(KnowledgeItem(raw_content=pdf.getvalue(), metadata={"page": page}))
    ids = [str(item.id) for item in items]
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""), items=items)
    agent = PageStitchingAgent(
        foreach=f'{{"record": {{"name": "chapter"}}, "ids": ["{ids[0]}", "{ids[2]}"]}}',
        input=".ids",
        metadata='{record: .record, page_count: (.ids | length)}',
    )

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata == {"count": 1, "replace": True}
    assert ctx.items[0].metadata == {"record": {"name": "chapter"}, "page_count": 2}
    assert len(PdfReader(io.BytesIO(await ctx.items[0].content(ctx))).pages) == 2


@pytest.mark.asyncio
async def test_page_stitching_agent_combines_virtual_page_items():
    with PdfWriter() as writer, io.BytesIO() as pdf:
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.write(pdf)
        ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=pdf.getvalue()))

    await PageChunkingAgent(mode="pdf").run(ctx)
    ids = [str(item.id) for item in ctx.items]
    agent = PageStitchingAgent(
        foreach=f'{{"record": {{"name": "chapter"}}, "ids": ["{ids[0]}", "{ids[2]}"]}}',
        input=".ids",
        metadata='{record: .record, page_count: (.ids | length)}',
    )

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata == {"count": 1, "replace": True}
    assert len(PdfReader(io.BytesIO(await ctx.items[0].content(ctx))).pages) == 2
