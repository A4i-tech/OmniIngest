"""Agents for refining and standardizing text content using LLMs and heuristics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import AgentMixin, ResolvedResource, Step, StepResult, StepStatus


class MarkdownCleaningAgent(BaseModel, Step, AgentMixin):
    """
    Standardizes Markdown formatting and removes non-essential content.

    This agent utilizes a large language model (LLM) to refine extracted
    text by correcting formatting inconsistencies and eliminating
    boilerplate or redundant informational elements.
    """

    prompt_template: str = Field(
        "Clean the following markdown text by fixing formatting issues, removing boilerplate, and ensuring consistency. Output ONLY the cleaned markdown.\n\n{text}",
        description="The instructions we send to the AI editor",
    )

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        text = ctx.resource.decode()
        resp = await self._agent(ctx).run(self.prompt_template.format(text=text))
        cleaned = resp.output or ""
        ctx.resource.content = cleaned.encode(ctx.resource.content_encoding)
        return StepResult(status=StepStatus.SUCCESS, metadata={"original_len": len(text), "cleaned_len": len(cleaned)})


class TranscriptionPostProcessorAgent(BaseModel, Step, AgentMixin):
    """
    Refines automated speech-to-text transcripts for improved accuracy.

    This agent employs an LLM to correct grammatical errors and
    common phonological misinterpretations in raw audio transcripts,
    enhancing the overall readability and semantic precision of the data.
    """

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        if not ctx.resource.content:
            return StepResult(status=StepStatus.SKIPPED, error="No transcript")
        prompt = f"Improve the following audio transcript by fixing punctuation and correcting common STT errors. Output ONLY the improved text.\n\n{ctx.resource.decode()}"
        resp = await self._agent(ctx).run(prompt)
        ctx.resource.content = resp.output.encode(ctx.resource.content_encoding)
        return StepResult(status=StepStatus.SUCCESS)
