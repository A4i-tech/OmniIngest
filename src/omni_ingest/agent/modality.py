"""Agents for modality transformation."""

import io

from pydantic import BaseModel, Field

from ..core.model import ByteContent, IngestionContext, ResolvedResource, Step, StepResult, StepStatus
from ..parser import parse


class TextAgent(BaseModel, Step):
    """
    Transforms incoming modality into textual content.

    This agent normalizes resources and existing KnowledgeItems to text. This
    is useful when downstream processing expects textual content. If the content
    is already text, it only records the encoding on the object.

    Note that this agent infers content type using libmagic to check for `text/`
    prefix. The output content may be of any valid `text/` type, not necessarily
    `text/plain`.
    """

    encoding: str = Field("utf-8", description="Encoding used for decoded text bytes")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        await self.convert(ctx.resource, ctx.resource.uri)
        for item in ctx.items:
            await self.convert(item, item.source_uri or ctx.resource.uri)
        return StepResult(status=StepStatus.SUCCESS, metadata={"content_type": ctx.resource.content_type})

    async def convert(self, obj: ByteContent, uri: str) -> None:
        if not obj.content_type.startswith("text/"):
            text, metadata = await parse(uri, io.BytesIO(obj.content), encoding=self.encoding)
            obj.metadata.update(metadata)
            obj.content = text.encode(self.encoding)
            if not (ct := obj.content_type).startswith("text/") and ct != "application/x-empty":
                raise ValueError(f"Could not transform content to text, resulting content-type was {ct}")
        obj.content_encoding = self.encoding
