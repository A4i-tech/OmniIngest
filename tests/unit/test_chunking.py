import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
from pypdf import PdfReader, PdfWriter

from omni_ingest.agent.chunking import PageChunkingAgent, PageStitchingAgent, TimedChunkingAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, create_pipeline_from_config
from omni_ingest.port.metadata_store import NullMetadataStore

PROFILE = Path(__file__).resolve().parents[2] / "src" / "omni_ingest" / "profiles" / "sentence_chunking.yaml"


def create_ctx(content: bytes) -> IngestionContext[ResolvedResource]:
    resource = ResolvedResource(uri="", content=ByteContent.from_bytes(content))
    return IngestionContext(resource=resource, items=[KnowledgeItem(content=ByteContent.from_bytes(content))])


@pytest.mark.asyncio
async def test_byte_content_rejects_unknown_omni_ingest_route():
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"context")))
    item = KnowledgeItem(content=ByteContent.from_uri("omni-ingest://resource/unknown/1", content_type="application/pdf"))

    with pytest.raises(ValueError, match="Unsupported content URI"):
        await item.content.read(ctx)


@pytest.mark.asyncio
async def test_timed_chunking_from_word_timestamps():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")),
        items=[KnowledgeItem(content=ByteContent.from_bytes(b""), metadata={
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
    assert await result.items[0].content.decode(ctx) == "Hello world"
    assert result.items[0].metadata["start_offset_ms"] == 1000
    assert await result.items[1].content.decode(ctx) == "this is audio"
    assert result.items[1].metadata["start_offset_ms"] == 40000


@pytest.mark.asyncio
async def test_timed_chunking_accepts_custom_token_schema():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")),
        items=[KnowledgeItem(content=ByteContent.from_bytes(b""), metadata={"tokens": [{"value": "Hello", "start": 1.0, "length": 0.2}, {"value": "world", "start": 1.5, "length": 0.2}]})],
    )
    agent = TimedChunkingAgent(tokens=".metadata.tokens[]", text=".value", start=".start", duration=".length", units_per_second=1)

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert await result.items[0].content.decode(ctx) == "Hello world"
    assert result.items[0].metadata["start_offset_ms"] == 1000


@pytest.mark.asyncio
async def test_sentence_chunking_profile_hard_splits_unpunctuated_table_row():
    runner = create_pipeline_from_config(PROFILE, config_args={"chunk_size": 50})
    row = "|" + "|".join(f"cell{i}" for i in range(40)) + "|"
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")),
        items=[KnowledgeItem(content=ByteContent.from_bytes(row.encode()), metadata={"content_type": "text/plain"})],
        store=NullMetadataStore(),
    )

    results = await runner.run(ctx)

    assert results[0].status == StepStatus.SUCCESS
    assert len(ctx.items) > 1
    for item in ctx.items:
        assert len(await item.content.read(ctx)) <= 50


@pytest.mark.asyncio
async def test_sentence_chunking_profile_keeps_text_under_chunk_size_in_one_chunk():
    runner = create_pipeline_from_config(PROFILE, config_args={"chunk_size": 500})
    text = "First sentence. Second sentence."
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")),
        items=[KnowledgeItem(content=ByteContent.from_bytes(text.encode()), metadata={"content_type": "text/plain"})],
        store=NullMetadataStore(),
    )

    results = await runner.run(ctx)

    assert results[0].status == StepStatus.SUCCESS
    assert len(ctx.items) == 1
    assert await ctx.items[0].content.decode(ctx) == text


async def run_profile(items, chunk_size=50):
    runner = create_pipeline_from_config(PROFILE, config_args={"chunk_size": chunk_size})
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=items, store=NullMetadataStore())
    return await runner.run(ctx), ctx


@pytest.mark.asyncio
async def test_sentence_chunking_profile_keeps_tenant_and_content_type():
    tenant_id = uuid4()
    item = KnowledgeItem(content=ByteContent.from_bytes(b"a\nb\nc\n", content_type="text/markdown"), tenant_id=tenant_id, source_uri="s3://x", metadata={"k": 1})

    _, ctx = await run_profile([item], chunk_size=4)

    assert len(ctx.items) == 2
    for chunk in ctx.items:
        assert chunk.id != item.id
        assert (chunk.tenant_id, chunk.source_uri, chunk.metadata) == (tenant_id, "s3://x", {"k": 1})
        assert await chunk.content.content_type(ctx) == "text/markdown"


@pytest.mark.asyncio
async def test_sentence_chunking_profile_passes_non_text_item_through():
    image = KnowledgeItem(content=ByteContent.from_bytes(b"PNG", content_type="image/png"))
    text = KnowledgeItem(content=ByteContent.from_bytes(b"hello", content_type="text/plain"))

    _, ctx = await run_profile([image, text])

    assert ctx.items[0] is image
    assert await ctx.items[1].content.decode(ctx) == "hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [0, -1])
async def test_sentence_chunking_profile_rejects_non_positive_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="Invalid parameters"):
        create_pipeline_from_config(PROFILE, config_args={"chunk_size": chunk_size})


@pytest.mark.asyncio
async def test_sentence_chunking_profile_chunks_non_ascii_text_within_limit():
    text = "नमस्ते दुनिया।" * 10 + "你好，世界。" * 10
    item = KnowledgeItem(content=ByteContent.from_bytes(text.encode(), content_type="text/plain"))

    _, ctx = await run_profile([item], chunk_size=20)

    chunks = [await chunk.content.decode(ctx) for chunk in ctx.items]
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert "".join(chunks) == text


@pytest.mark.asyncio
async def test_sentence_chunking_profile_skips_when_no_chunks():
    item = KnowledgeItem(content=ByteContent.from_bytes(b"", content_type="text/plain"))

    results, _ = await run_profile([item])

    assert results[0].status == StepStatus.SKIPPED


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
    assert all([await item.content.content_type(ctx) == "application/pdf" for item in result.items])
    assert [item.content.uri for item in result.items] == ["omni-ingest://resource/pages/1", "omni-ingest://resource/pages/2"]
    assert all(item.content.data == b"" for item in result.items)
    assert all([len(PdfReader(io.BytesIO(await item.content.read(ctx))).pages) == 1 for item in result.items])


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
            items.append(KnowledgeItem(content=ByteContent.from_bytes(pdf.getvalue()), metadata={"page": page}))
    ids = [str(item.id) for item in items]
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=items)
    agent = PageStitchingAgent(
        foreach=f'{{"record": {{"name": "chapter"}}, "ids": ["{ids[0]}", "{ids[2]}"]}}',
        input=".ids",
        metadata='{record: .record, page_count: (.ids | length)}',
    )

    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata == {"count": 1, "replace": True}
    assert ctx.items[0].metadata == {"record": {"name": "chapter"}, "page_count": 2}
    assert len(PdfReader(io.BytesIO(await ctx.items[0].content.read(ctx))).pages) == 2


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
    assert ctx.items[0].content.data == b""
    assert ctx.items[0].content.uri == "omni-ingest://resource/pages/1,3"
    assert len(PdfReader(io.BytesIO(await ctx.items[0].content.read(ctx))).pages) == 2


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
    assert item.content.uri is not None
    recipe = json.loads(parse_qs(urlparse(item.content.uri).query)["recipe"][0])
    assert item.content.data == b""
    assert recipe == {"op": "pdf.concat", "inputs": ["omni-ingest://resource/pages/1,2", "omni-ingest://resource/pages/3,4"]}
    assert len(PdfReader(io.BytesIO(await item.content.read(ctx))).pages) == 4
