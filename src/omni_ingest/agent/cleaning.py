"""Agents for refining and standardizing text content using LLMs and heuristics."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

from ..core.model import IngestionContext, Step, StepResult, StepStatus, handle_step_error
from ..core.pipeline import step


@step("markdown_cleaning")
class MarkdownCleaningAgent(BaseModel, Step):
    """
    Standardizes Markdown formatting and removes non-essential content.

    This agent utilizes a large language model (LLM) to refine extracted
    text by correcting formatting inconsistencies and eliminating
    boilerplate or redundant informational elements.
    """

    model: KnownModelName = Field("openai:gpt-4o", description="Model to use for cleaning")
    prompt_template: str = Field(
        "Clean the following markdown text by fixing formatting issues, removing boilerplate, and ensuring consistency. Output ONLY the cleaned markdown.\n\n{text}",
        description="The instructions we send to the AI editor",
    )

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        text = ctx.metadata.get("text")
        if not text:
            return StepResult(status=StepStatus.SKIPPED, error="No text to clean")
        resp = await Agent(self.model).run(self.prompt_template.format(text=text))
        cleaned = resp.output or ""
        ctx.metadata["text"] = cleaned
        return StepResult(status=StepStatus.SUCCESS, metadata={"original_len": len(text), "cleaned_len": len(cleaned)})


@step("transcription_post_processor")
class TranscriptionPostProcessorAgent(BaseModel, Step):
    """
    Refines automated speech-to-text transcripts for improved accuracy.

    This agent employs an LLM to correct grammatical errors and
    common phonological misinterpretations in raw audio transcripts,
    enhancing the overall readability and semantic precision of the data.
    """

    model: KnownModelName = Field("openai:gpt-4o", description="Model to use for post-processing")

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        text = ctx.metadata.get("text")
        if not text:
            return StepResult(status=StepStatus.SKIPPED, error="No transcript")
        prompt = f"Improve the following audio transcript by fixing punctuation and correcting common STT errors. Output ONLY the improved text.\n\n{text}"
        resp = await Agent(self.model).run(prompt)
        ctx.metadata["text"] = resp.output
        return StepResult(status=StepStatus.SUCCESS)
