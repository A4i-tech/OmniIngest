"""Agents for modality transformation."""

from __future__ import annotations

import io
import mimetypes
from typing import TYPE_CHECKING, Any, BinaryIO

from pydantic import BaseModel, Field
from pydantic_ai import BinaryContent

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import AgentMixin, ByteContent, ResolvedResource, Step, StepResult, StepStatus
from ..parser import parse_audio, parse_document, parse_text


class TextAgent(BaseModel, Step, AgentMixin):
    """
    Transforms incoming modality into textual content.

    This agent normalizes resources and existing KnowledgeItems to text. This
    is useful when downstream processing expects textual content. If the content
    is already text, it only records the encoding on the object.

    Note that this agent infers content type using libmagic to check for `text/`
    prefix. The output content may be of any valid `text/` type, not necessarily
    `text/plain`.
    """

    encoding: str = Field(default="utf-8", description="Encoding used for decoded text bytes")

    image_conversion_prompt: str = Field(default=(
        "Convert the image into a complete, accessibility-focused text description for someone who cannot see it. "
        "Capture all visible text exactly, describe the layout, objects, people, actions, relationships, colors, symbols, charts, tables, UI elements, and any important visual context. "
        "Preserve reading order where possible. "
        "Be factual, avoid guessing, and clearly mark anything uncertain. "
        "The output should let a visually impaired person understand and reason about the image without needing to view it."
    ), description="The prompt to use when transforming image to text.")


    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        for item in ctx.items:
            await self.convert(ctx, item, item.source_uri or ctx.resource.uri)
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"content_types": [await item.content_type(ctx) for item in ctx.items]})


    async def convert(self, ctx: IngestionContext[ResolvedResource], obj: ByteContent, uri: str) -> None:
        if not (await obj.content_type(ctx)).startswith("text/"):
            text, metadata = await self.parse(ctx, uri, io.BytesIO(await obj.content(ctx)), encoding=self.encoding)
            obj.metadata.update(metadata)
            obj.raw_content = text.encode(self.encoding)
            obj.content_uri = None
            if not (ct := await obj.content_type(ctx)).startswith("text/") and ct != "application/x-empty":
                raise ValueError(f"Could not transform content to text, resulting content-type was {ct}")
        obj.content_encoding = self.encoding


    async def parse(self, ctx: IngestionContext[ResolvedResource], path: str, stream: BinaryIO, *, encoding: str = "utf-8") -> tuple[str, dict[str, Any]]:
        content_type, _ = mimetypes.guess_type(path)
        content_type = content_type or "application/octet-stream"

        if content_type.startswith("audio/"):
            content, metadata = await parse_audio(path, stream)
            metadata["selected_parser"] = "audio"
            return content, metadata
        if content_type.startswith("text/"):
            content, metadata = await parse_text(stream, encoding=encoding)
            metadata["selected_parser"] = "text"
            return content, metadata
        if content_type.startswith("application/"):
            content, metadata = await parse_document(stream, encoding=encoding)
            metadata["selected_parser"] = "document"
            return content, metadata
        if content_type.startswith("image/"):
            async with self._agent(ctx) as agent:
                result = await agent.run([BinaryContent(data=stream.read(), media_type=content_type)], instructions=self.image_conversion_prompt)
            return result.output, {"selected_parser": "image"}

        raise ValueError(f"Unsupported data URI MIME type: {content_type!r}")
