"""Agents for segmenting large documents and transcripts into manageable chunks."""

import io
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, field_validator
from pypdf import PdfReader

from ..core.model import IngestionContext, KnowledgeItem, Step, StepResult, StepStatus


class SentenceChunkingAgent(BaseModel, Step):
    """
    Segments text into discrete chunks while maintaining sentence boundaries.

    Large documents are partitioned into smaller segments of a specified
    character count. The agent prioritizes breaks at sentence endings
    to preserve contextual integrity for subsequent search operations.
    """

    chunk_size: PositiveInt = Field(512, description="Maximum number of characters per chunk")
    overlap: NonNegativeInt = Field(50, description="Character overlap between adjacent chunks")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not (text := ctx.metadata.get("text")):
            return StepResult(status=StepStatus.SKIPPED, error="No text")
        text, chunks, pos = " ".join(text.split()), [], 0
        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            if end < len(text):
                chunk = text[pos:end]
                breaks = [chunk.rfind(p) + len(p) - 1 for p in [". ", "! ", "? ", ".\n", "!\n", "?\n"] if chunk.rfind(p) > self.chunk_size // 2]
                if breaks:
                    end = pos + max(breaks) + 1
            chunks.append((text[pos:end].strip(), pos))
            pos = end - self.overlap if end < len(text) and end - self.overlap > pos else end
        ctx.items = [KnowledgeItem(content=c, tenant_id=ctx.tenant_id, metadata={"source_idx": i, "char_start": p}) for i, (c, p) in enumerate(chunks) if c]
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

    async def run(self, ctx: IngestionContext) -> StepResult:
        path, raw = ctx.metadata.get("file_path"), ctx.metadata.get("raw_data")
        if not path and not raw:
            return StepResult(status=StepStatus.SKIPPED, error="No PDF data")
        if path:
            reader = PdfReader(path)
        elif raw is not None:
            reader = PdfReader(io.BytesIO(raw))
        else:
            return StepResult(status=StepStatus.SKIPPED, error="No PDF data")
        ctx.items = [KnowledgeItem(content=p.extract_text(), metadata={"page": i + 1}) for i, p in enumerate(reader.pages) if p.extract_text().strip()]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"page_count": len(ctx.items)})


class TimedChunkingAgent(BaseModel, Step):
    """
    Segments audio transcription data into temporal blocks.

    This agent divides indexed audio text into intervals based on
    timestamps, facilitating precise time-based search and retrieval within
    audiovisual datasets.
    """

    segment_duration: timedelta = Field(timedelta(seconds=30), description="Target duration of each time block")
    overlap: timedelta = Field(timedelta(0), description="Overlap between adjacent time blocks")
    pause_threshold: timedelta | None = Field(None, description="Silence threshold to split blocks")

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

    async def run(self, ctx: IngestionContext) -> StepResult:
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
                chunks.append(KnowledgeItem(content=text, metadata=metadata))
                current_chunk_words = [w]
                start_tick = offset
            else:
                current_chunk_words.append(w)

        if current_chunk_words:
            text = " ".join([cw["Word"] for cw in current_chunk_words])
            metadata = {"start_offset_ms": start_tick // 10000, "end_offset_ms": (current_chunk_words[-1]["Offset"] + current_chunk_words[-1]["Duration"]) // 10000, "word_count": len(current_chunk_words)}
            chunks.append(KnowledgeItem(content=text, metadata=metadata))

        ctx.items = chunks
        return StepResult(status=StepStatus.SUCCESS, items=chunks, metadata={"chunk_count": len(chunks)})
