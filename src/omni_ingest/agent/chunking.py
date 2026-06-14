"""Agents for segmenting large documents and transcripts into manageable chunks."""

from __future__ import annotations

import io
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import jq
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, field_validator
from pypdf import PdfReader, PdfWriter

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import KnowledgeItem, ResolvedResource, Step, StepResult, StepStatus


class SentenceChunkingAgent(BaseModel, Step):
    """
    Segments text into discrete chunks while maintaining sentence boundaries.

    Large documents are partitioned into smaller segments of a specified
    character count. The agent prioritizes breaks at sentence endings
    to preserve contextual integrity for subsequent search operations.
    """

    chunk_size: PositiveInt = Field(default=512, description="Maximum number of characters per chunk")
    overlap: NonNegativeInt = Field(default=50, description="Character overlap between adjacent chunks")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        if not ctx.resource.content:
            return StepResult(status=StepStatus.SKIPPED, error="No text")
        encoding = ctx.resource.content_encoding
        text, chunks, pos = " ".join(ctx.resource.decode().split()), [], 0
        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            if end < len(text):
                chunk = text[pos:end]
                breaks = [chunk.rfind(p) + len(p) - 1 for p in [". ", "! ", "? ", ".\n", "!\n", "?\n"] if chunk.rfind(p) > self.chunk_size // 2]
                if breaks:
                    end = pos + max(breaks) + 1
            chunks.append((text[pos:end].strip(), pos))
            pos = end - self.overlap if end < len(text) and end - self.overlap > pos else end
        ctx.items = [KnowledgeItem(content=c.encode(encoding), tenant_id=ctx.tenant_id, content_encoding=encoding, metadata={"source_idx": i, "char_start": p}) for i, (c, p) in enumerate(chunks) if c]
        return StepResult(
            status=StepStatus.SUCCESS,
            items=ctx.items,
            metadata={"count": len(ctx.items), "chunk_size": self.chunk_size, "overlap": self.overlap},
        )


class PageChunkingAgent(BaseModel, Step):
    """
    Partitions PDF documents based on their original page layout.

    This agent creates distinct KnowledgeItems for each page identified
    in a PDF source, maintaining the hierarchical structure of the
    input document.
    """

    mode: Literal["text", "pdf"] = Field(default="text", description="Whether to emit extracted page text or single-page PDFs")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        if (ct := ctx.resource.content_type) != "application/pdf":
            raise ValueError(f"Expected a document, got {ct}")

        enc = ctx.resource.content_encoding
        items = []
        with io.BytesIO(ctx.resource.content) as s, PdfReader(stream=s) as reader:
            if self.mode == "pdf":
                for i, page in enumerate(reader.pages):
                    with io.BytesIO() as s, PdfWriter() as w:
                        w.add_page(page)
                        w.write(s)
                        items.append(KnowledgeItem(content=s.getvalue(), content_encoding=enc, metadata={"page": i + 1, "content_type": ct}))
            else:
                items.extend(
                    KnowledgeItem(content=text.encode(enc), content_encoding=enc, metadata={"page": i + 1, "content_type": "text/plain"})
                    for i, p in enumerate(reader.pages) if (text := p.extract_text()).strip()
                )
        ctx.items = items
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"page_count": len(ctx.items), "mode": self.mode})


class PageStitchingAgent(BaseModel, Step):
    """
    Combines selected PDF page items into larger PDF items.

    jq expressions define the stitch specs and selected page item ids, making
    this reusable for chapters, sections, ranges, or any metadata-driven slices.
    """

    foreach: str = Field(default=".", description="jq expression producing stitch specs")
    input: str = Field(..., description="jq expression selecting page item ids for each stitch spec")
    metadata: str = Field(default=".", description="jq expression producing metadata for each stitched item")
    replace: bool = Field(default=True, description="Replace ctx.items with stitched PDFs")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        item_docs = [{"id": str(item.id), "metadata": item.metadata, "source_uri": item.source_uri} for item in ctx.items]
        item_by_id = {item_doc["id"]: item for item_doc, item in zip(item_docs, ctx.items, strict=True)}
        doc = {"metadata": ctx.metadata, "items": item_docs}
        stitched = []

        for spec in jq.compile(self.foreach).input_value(doc).all():
            ids = jq.compile(self.input).input_value(spec).first()
            if not isinstance(ids, list) or not ids or any(item_id not in item_by_id for item_id in ids):
                raise ValueError("PageStitchingAgent input must resolve to known item ids")

            with io.BytesIO() as s, PdfWriter() as writer:
                for item_id in ids:
                    with io.BytesIO(item_by_id[item_id].content) as page_pdf, PdfReader(page_pdf) as reader:
                        writer.append(reader)
                writer.write(s)
                stitched.append(KnowledgeItem(content=s.getvalue(), content_encoding="binary", metadata=jq.compile(self.metadata).input_value(spec).first()))

        if self.replace:
            ctx.items = stitched
        else:
            ctx.items.extend(stitched)
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"count": len(stitched), "replace": self.replace})


class TimedChunkingAgent(BaseModel, Step):
    """
    Segments audio transcription data into temporal blocks.

    This agent divides indexed audio text into intervals based on
    timestamps, facilitating precise time-based search and retrieval within
    audiovisual datasets.
    """

    segment_duration: timedelta = Field(default=timedelta(seconds=30), description="Target duration of each time block")
    overlap: timedelta = Field(default=timedelta(0), description="Overlap between adjacent time blocks")
    pause_threshold: timedelta | None = Field(default=None, description="Silence threshold to split blocks")

    @field_validator("segment_duration")
    def _validate_segment_duration(cls, value: timedelta) -> timedelta:
        if value <= timedelta(0):
            raise ValueError("segment_duration must be positive")
        return value

    @field_validator("overlap")
    def _validate_overlap(cls, value: timedelta) -> timedelta:
        if value < timedelta(0):
            raise ValueError("overlap must be non-negative")
        return value

    @field_validator("pause_threshold")
    def _validate_pause_threshold(cls, value: timedelta | None) -> timedelta | None:
        if value is not None and value <= timedelta(0):
            raise ValueError("pause_threshold must be positive")
        return value

    @property
    def segment_duration_ticks(self) -> int:
        return int(self.segment_duration.total_seconds() * 10_000_000)

    @property
    def overlap_ticks(self) -> int:
        return int(self.overlap.total_seconds() * 10_000_000)

    @property
    def pause_threshold_ticks(self) -> int | None:
        return int(self.pause_threshold.total_seconds() * 10_000_000) if self.pause_threshold else None

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        words = ctx.metadata.get("word_timestamps")
        if not words:
            return StepResult(status=StepStatus.SKIPPED, error="No word_timestamps found in metadata")

        chunks: list[KnowledgeItem] = []
        current_chunk_words: list[dict[str, Any]] = []
        start_tick = 0

        for w in words:
            offset, duration = w["Offset"], w["Duration"]
            if not current_chunk_words:
                start_tick = offset

            if (offset + duration - start_tick) > self.segment_duration_ticks and current_chunk_words:
                text = " ".join([cw["Word"] for cw in current_chunk_words])
                metadata = {"start_offset_ms": start_tick // 10000, "end_offset_ms": (current_chunk_words[-1]["Offset"] + current_chunk_words[-1]["Duration"]) // 10000, "word_count": len(current_chunk_words)}
                chunks.append(KnowledgeItem(content=text.encode(ctx.resource.content_encoding), content_encoding=ctx.resource.content_encoding, metadata=metadata))
                current_chunk_words = [w]
                start_tick = offset
            else:
                current_chunk_words.append(w)

        if current_chunk_words:
            text = " ".join([cw["Word"] for cw in current_chunk_words])
            metadata = {"start_offset_ms": start_tick // 10000, "end_offset_ms": (current_chunk_words[-1]["Offset"] + current_chunk_words[-1]["Duration"]) // 10000, "word_count": len(current_chunk_words)}
            chunks.append(KnowledgeItem(content=text.encode(ctx.resource.content_encoding), content_encoding=ctx.resource.content_encoding, metadata=metadata))

        ctx.items = chunks
        return StepResult(status=StepStatus.SUCCESS, items=chunks, metadata={"chunk_count": len(chunks)})
