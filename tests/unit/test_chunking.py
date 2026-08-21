import io
import json
from urllib.parse import parse_qs, urlparse

import pytest
from pypdf import PdfReader, PdfWriter

from omni_ingest.agent.chunking import PageChunkingAgent, PageStitchingAgent, SentenceChunkingAgent, TimedChunkingAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext


def create_ctx(content: bytes) -> IngestionContext[ResolvedResource]:
    resource = ResolvedResource(uri="", raw_content=content)
    return IngestionContext(resource=resource, items=[KnowledgeItem(raw_content=content)])


@pytest.mark.asyncio
async def test_chunking_with_default_params():
    agent = SentenceChunkingAgent()

    assert agent.chunk_size == 512
    assert agent.overlap == 50

    ctx = create_ctx(b"Hello world. This is a test. Another sentence.")
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["chunk_size"] == 512
    assert len(result.items) >= 1


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
    ctx = create_ctx(long_text.encode())
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] > 1


@pytest.mark.asyncio
async def test_chunking_respects_sentence_boundaries():
    agent = SentenceChunkingAgent(chunk_size=50, overlap=10)

    text = "First sentence here. Second sentence follows. Third one too."
    ctx = create_ctx(text.encode())
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert not (await item.content(ctx)).startswith(b" ")


@pytest.mark.asyncio
async def test_chunking_short_text():
    agent = SentenceChunkingAgent(chunk_size=500, overlap=50)

    short_text = "This is a short text."
    ctx = create_ctx(short_text.encode())
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 1
    assert await result.items[0].content(ctx) == short_text.encode()


@pytest.mark.asyncio
async def test_chunking_includes_position_metadata():
    agent = SentenceChunkingAgent(chunk_size=50, overlap=0)

    text = "First part. Second part. Third part."
    ctx = create_ctx(text.encode())
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert "source_idx" in item.metadata
        assert "char_start" in item.metadata


@pytest.mark.asyncio
async def test_chunking_empty_text():
    agent = SentenceChunkingAgent()

    ctx = create_ctx(b"")
    result = await agent.run(ctx)
    assert result.status == StepStatus.SKIPPED


@pytest.mark.asyncio
async def test_chunking_whitespace_normalization():
    agent = SentenceChunkingAgent()

    text = "Hello    world.\n\n\nThis   has   extra    spaces."
    ctx = create_ctx(text.encode())
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    for item in result.items:
        assert b"  " not in await item.content(ctx)


@pytest.mark.asyncio
async def test_timed_chunking_from_word_timestamps():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", raw_content=b""),
        items=[KnowledgeItem(metadata={
            "word_timestamps": [
                {"Word": "Hello", "Offset": 10000000, "Duration": 2000000},
                {"Word": "world", "Offset": 15000000, "Duration": 2000000},
                {"Word": "this", "Offset": 400000000, "Duration": 2000000},
                {"Word": "is", "Offset": 410000000, "Duration": 2000000},
                {"Word": "audio", "Offset": 420000000, "Duration": 2000000},
            ]
        })],
    )

    result = await TimedChunkingAgent(segment_duration="00:00:10").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert len(result.items) == 2
    assert await result.items[0].decode(ctx) == "Hello world"
    assert result.items[0].metadata["start_offset_ms"] == 1000
    assert await result.items[1].decode(ctx) == "this is audio"
    assert result.items[1].metadata["start_offset_ms"] == 40000


@pytest.mark.asyncio
async def test_timed_chunking_accepts_custom_token_schema():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", raw_content=b""),
        items=[KnowledgeItem(metadata={"tokens": [{"value": "Hello", "start": 1.0, "length": 0.2}, {"value": "world", "start": 1.5, "length": 0.2}]})],
    )
    agent = TimedChunkingAgent(tokens=".metadata.tokens[]", text=".value", start=".start", duration=".length", units_per_second=1)

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert await result.items[0].decode(ctx) == "Hello world"
    assert result.items[0].metadata["start_offset_ms"] == 1000


@pytest.mark.asyncio
async def test_sentence_chunking_selects_items_and_custom_boundaries():
    selected = KnowledgeItem(raw_content=b"Section one; Section two; Section three", metadata={"kind": "paragraph"})
    ignored = KnowledgeItem(raw_content=b"Ignore this", metadata={"kind": "caption"})
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""), items=[selected, ignored])

    result = await SentenceChunkingAgent(chunk_size=20, overlap=0, foreach='.items[] | select(.metadata.kind == "paragraph")', separators=["; "]).run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert all(item.metadata["kind"] == "paragraph" for item in result.items)
    assert b"Ignore" not in b" ".join([await item.content(ctx) for item in result.items])


@pytest.mark.asyncio
async def test_page_chunking_pdf_mode_emits_virtual_page_items():
    with PdfWriter() as writer, io.BytesIO() as pdf:
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.write(pdf)
        ctx = create_ctx(pdf.getvalue())

    result = await PageChunkingAgent(mode="pdf").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata == {"page_count": 2, "mode": "pdf"}
    assert [item.metadata["page"] for item in result.items] == [1, 2]
    assert all(item.metadata["content_type"] == "application/pdf" for item in result.items)
    assert [item.content_uri for item in result.items] == ["omni-ingest://resource/pages/1", "omni-ingest://resource/pages/2"]
    assert all(item.raw_content == b"" for item in result.items)
    assert all([len(PdfReader(io.BytesIO(await item.content(ctx))).pages) == 1 for item in result.items])


@pytest.mark.asyncio
async def test_page_chunking_real_pdf(resources_dir):
    ctx = create_ctx((resources_dir / "when-tomorrow-was-yesterday.pdf").read_bytes())

    result = await PageChunkingAgent().run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert len(ctx.items) == 12


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
        ctx = create_ctx(pdf.getvalue())

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
    assert ctx.items[0].raw_content == b""
    assert ctx.items[0].content_uri == "omni-ingest://resource/pages/1,3"
    assert len(PdfReader(io.BytesIO(await ctx.items[0].content(ctx))).pages) == 2


@pytest.mark.asyncio
async def test_page_stitching_agent_chains_virtual_stitches():
    with PdfWriter() as writer, io.BytesIO() as pdf:
        for _ in range(4):
            writer.add_blank_page(width=72, height=72)
        writer.write(pdf)
        ctx = create_ctx(pdf.getvalue())

    await PageChunkingAgent(mode="pdf").run(ctx)
    ids = [str(item.id) for item in ctx.items]
    await PageStitchingAgent(
        foreach=f'[{{"ids": ["{ids[0]}", "{ids[1]}"]}}, {{"ids": ["{ids[2]}", "{ids[3]}"]}}][]',
        input=".ids",
    ).run(ctx)
    stitched_ids = [str(item.id) for item in ctx.items]

    await PageStitchingAgent(
        foreach=f'{{"ids": ["{stitched_ids[0]}", "{stitched_ids[1]}"]}}',
        input=".ids",
    ).run(ctx)

    item = ctx.items[0]
    assert item.content_uri is not None
    recipe = json.loads(parse_qs(urlparse(item.content_uri).query)["recipe"][0])
    assert item.raw_content == b""
    assert recipe == {"op": "pdf.concat", "inputs": ["omni-ingest://resource/pages/1,2", "omni-ingest://resource/pages/3,4"]}
    assert len(PdfReader(io.BytesIO(await item.content(ctx))).pages) == 4
