"""Agents for refining and standardizing text content using LLMs and heuristics."""

from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, PrivateAttr

from ..core.config import settings
from ..core.model import IngestionContext, Step, StepResult, StepStatus
from ..core.pipeline import step


def _get_openai_client(provider: str):
    if provider == "azure":
        key = settings.azure_openai_api_key.get_secret_value() if settings.azure_openai_api_key else None
        return AsyncAzureOpenAI(
            api_key=key,
            api_version="2024-02-01",
            azure_endpoint=str(settings.azure_openai_endpoint) if settings.azure_openai_endpoint else "",
        )
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return AsyncOpenAI(api_key=key)


@step("markdown_cleaning")
class MarkdownCleaningAgent(BaseModel, Step):
    """
    Standardizes Markdown formatting and removes non-essential content.

    This agent utilizes a large language model (LLM) to refine extracted
    text by correcting formatting inconsistencies and eliminating
    boilerplate or redundant informational elements.
    """

    model_config = {"arbitrary_types_allowed": True}
    provider: str = Field("openai", description="The AI service to use (e.g., 'openai' or 'azure')")
    model: str = Field("gpt-4o", description="The specific AI model version to use")
    prompt_template: str = Field(
        "Clean the following markdown text by fixing formatting issues, removing "
        "boilerplate, and ensuring consistency. Output ONLY the cleaned markdown.\n\n{text}",
        description="The instructions we send to the AI editor",
    )

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            self._client = _get_openai_client(self.provider)
        return self._client

    async def run(self, ctx: IngestionContext) -> StepResult:
        text = ctx.metadata.get("text")
        if not text:
            return StepResult(status=StepStatus.SKIPPED, error="No text to clean")
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": self.prompt_template.format(text=text)}]
            )
            cleaned = resp.choices[0].message.content
            ctx.metadata["text"] = cleaned
            return StepResult(
                status=StepStatus.SUCCESS, metadata={"original_len": len(text), "cleaned_len": len(cleaned)}
            )
        except Exception as e:
            return StepResult(status=StepStatus.FAILURE, error=str(e))


@step("transcription_post_processor")
class TranscriptionPostProcessorAgent(BaseModel, Step):
    """
    Refines automated speech-to-text transcripts for improved accuracy.

    This agent employs an LLM to correct grammatical errors and
    common phonological misinterpretations in raw audio transcripts,
    enhancing the overall readability and semantic precision of the data.
    """

    model_config = {"arbitrary_types_allowed": True}
    provider: str = Field("openai", description="The AI service to use")
    model: str = Field("gpt-4o", description="The AI model used for proofreading")

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            self._client = _get_openai_client(self.provider)
        return self._client

    async def run(self, ctx: IngestionContext) -> StepResult:
        text = ctx.metadata.get("text")
        if not text:
            return StepResult(status=StepStatus.SKIPPED, error="No transcript")
        prompt = (
            "Improve the following audio transcript by fixing punctuation and correcting "
            f"common STT errors. Output ONLY the improved text.\n\n{text}"
        )
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            ctx.metadata["text"] = resp.choices[0].message.content
            return StepResult(status=StepStatus.SUCCESS)
        except Exception as e:
            return StepResult(status=StepStatus.FAILURE, error=str(e))
