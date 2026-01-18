"""Agents for segmenting large documents and transcripts into manageable chunks."""

import io
from typing import Any

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt
from pypdf import PdfReader

from ..core.model import IngestionContext, KnowledgeItem, Step, StepResult, StepStatus
from ..core.pipeline import register_step


@register_step("chunking")
class SentenceChunkingAgent(BaseModel, Step):
    """
    Segments text into discrete chunks while maintaining sentence boundaries.

    Large documents are partitioned into smaller segments of a specified
    character count. The agent prioritizes breaks at sentence endings
    to preserve contextual integrity for subsequent search operations.
    """

    name: str = Field("chunking", description="The internal name for this chunking step")
    chunk_size: PositiveInt = Field(512, description="The maximum number of characters we should put in one piece")
    overlap: NonNegativeInt = Field(50, description="How many characters should carry over from one piece to the next?")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not (text := ctx.metadata.get("text")):
            return StepResult(status=StepStatus.SKIPPED, error="No text")
        text, chunks, pos = " ".join(text.split()), [], 0
        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            if end < len(text):
                chunk = text[pos:end]
                breaks = [
                    chunk.rfind(p) + len(p) - 1
                    for p in [". ", "! ", "? ", ".\n", "!\n", "?\n"]
                    if chunk.rfind(p) > self.chunk_size // 2
                ]
                if breaks:
                    end = pos + max(breaks) + 1
            chunks.append((text[pos:end].strip(), pos))
            pos = end - self.overlap if end < len(text) and end - self.overlap > pos else end
        ctx.items = [
            KnowledgeItem(content=c, tenant_id=ctx.tenant_id, metadata={"source_idx": i, "char_start": p})
            for i, (c, p) in enumerate(chunks)
            if c
        ]
        return StepResult(
            status=StepStatus.SUCCESS,
            items=ctx.items,
            metadata={"count": len(ctx.items), "chunk_size": self.chunk_size, "overlap": self.overlap},
        )


@register_step("page_chunking")
class PageChunkingAgent(BaseModel, Step):
    """
    Partitions PDF documents based on their original page layout.

    This agent creates distinct KnowledgeItems for each page identified
    in a PDF source, maintaining the hierarchical structure of the
    input document.
    """

    name: str = Field("page_chunking", description="Internal step name")

    async def run(self, ctx: IngestionContext) -> StepResult:
        path, raw = ctx.metadata.get("file_path"), ctx.metadata.get("raw_data")
        if not path and not raw:
            return StepResult(status=StepStatus.SKIPPED, error="No PDF data")
        try:
            if path:
                reader = PdfReader(path)
            elif raw is not None:
                reader = PdfReader(io.BytesIO(raw))
            else:
                return StepResult(status=StepStatus.SKIPPED, error="No PDF data")
            ctx.items = [
                KnowledgeItem(content=p.extract_text(), metadata={"page": i + 1})
                for i, p in enumerate(reader.pages)
                if p.extract_text().strip()
            ]
            return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"page_count": len(ctx.items)})
        except Exception as e:
            return StepResult(status=StepStatus.FAILURE, error=str(e))


@register_step("timed_chunking")
class TimedChunkingAgent(BaseModel, Step):
    """
    Segments audio transcription data into temporal blocks.

    This agent divides indexed audio text into intervals based on
    timestamps, facilitating precise time-based search and retrieval within
    audiovisual datasets.
    """

    name: str = Field("timed_chunking", description="Internal step name")
    segment_duration_ms: PositiveInt = Field(30000, description="Target length of time block in ms")
    overlap_ms: NonNegativeInt = Field(0, description="Should time blocks overlap at the edges?")
    pause_threshold_ms: PositiveInt | None = Field(None, description="Optional: break block if silence for this long")

    @property
    def segment_duration_ticks(self) -> int:
        return self.segment_duration_ms * 10000

    @property
    def overlap_ticks(self) -> int:
        return self.overlap_ms * 10000

    @property
    def pause_threshold_ticks(self) -> int | None:
        return self.pause_threshold_ms * 10000 if self.pause_threshold_ms else None

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
