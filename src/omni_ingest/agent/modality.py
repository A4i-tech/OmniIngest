"""Agents for converting modality bytes into text bytes."""

import io

from pydantic import BaseModel, Field

from ..core.model import ByteContent, IngestionContext, ResolvedResource, Step, StepResult, StepStatus
from ..parser import parse


class TextAgent(BaseModel, Step):
    encoding: str = Field("utf-8")

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
        obj.content_encoding = self.encoding
