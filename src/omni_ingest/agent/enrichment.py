"""Agents for structured metadata extraction and enrichment."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

import jq
from datamodel_code_generator import GenerateConfig
from datamodel_code_generator.dynamic import generate_dynamic_models
from pydantic import BaseModel, Field, PositiveInt
from pydantic_ai import BinaryContent, ModelRetry, NativeOutput, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.config import KnownSpeechModelName, KnownTranslationModelName, settings
from ..core.model import AgentMixin, ByteContent, JqExpression, KnowledgeItem, Step, StepResult, StepStatus
from ..core.speech import SpeechFormat, TTSBuilderParams
from ..port.translator import create_translator


def _assign(doc: dict[str, Any], path: str, output: Any) -> dict[str, Any]:
    return cast(dict[str, Any], jq.compile(f"{path} |= if type == \"object\" and ($output | type) == \"object\" then . + $output else $output end", args={"output": output}).input_value(doc).first())


async def _doc(ctx: IngestionContext, text: bool = False) -> dict[str, Any]:
    return {
        "metadata": ctx.metadata,
        "items": [
            {"id": str(item.id), "source_uri": item.source_uri, "metadata": item.metadata}
            | ({"text": await item.content.decode(ctx) if (await item.content.content_type(ctx)).startswith("text/") else None} if text else {})
            for item in ctx.items
        ],
    }


def _item_from_doc(doc: dict[str, Any]) -> KnowledgeItem:
    encoding = doc.get("content_encoding") or "utf-8"
    return KnowledgeItem(
        id=UUID(doc["id"]) if doc.get("id") else uuid4(),
        content=ByteContent.from_bytes(str(doc.get("text") or "").encode(encoding), content_type="text/plain", content_encoding=encoding),
        tenant_id=UUID(doc["tenant_id"]) if doc.get("tenant_id") else None,
        source_uri=doc.get("source_uri"),
        metadata=doc.get("metadata") or {},
    )


async def _content(item: KnowledgeItem, ctx: IngestionContext) -> str | BinaryContent:
    content_type = await item.content.content_type(ctx)
    return await item.content.decode(ctx) if content_type.startswith("text/") else BinaryContent(data=await item.content.read(ctx), media_type=content_type)


class ExtractAssertion(BaseModel):
    expr: JqExpression = Field(..., description="jq expression that must evaluate truthy against extracted output")
    feedback: str = Field(..., description="Feedback sent to the model when the assertion fails")


@dataclass
class ExtractAgentDeps:
    cursor: int
    buffered_ids: tuple[str, ...]


class ExtractAgent(BaseModel, Step, AgentMixin):
    """Extracts structured output with an LLM and writes it to metadata."""

    prompt: str = Field(..., description="Instruction prompt")
    json_schema: dict[str, Any] | JqExpression = Field(..., alias="schema", description="JSON schema or jq expression producing one")
    path: JqExpression = Field(..., description="jq path where extracted output is written")
    foreach: JqExpression = Field(default=".items[]", description="jq expression selecting documents to extract from")
    input: JqExpression = Field(default=".text", description="jq expression selecting prompt input")
    context: JqExpression | None = Field(default=None, description="jq expression selecting extra prompt context")
    mode: Literal["direct", "buffered"] = Field(default="direct", description="How selected input is supplied to the model")
    assertions: list[ExtractAssertion] = Field(default_factory=list, description="LLM retry assertions")
    concurrency: PositiveInt = Field(default=1, description="Maximum number of selected documents to extract concurrently")

    def _output_type(self, item: Any) -> type[BaseModel]:
        output = self.json_schema if isinstance(self.json_schema, dict) else jq.compile(self.json_schema).input_value(item).first()
        schema = {"type": "object", "properties": {"output": output}, "required": ["output"]}
        return generate_dynamic_models(schema, config=GenerateConfig(formatters=[]))["Model"]

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for extraction")

        doc = await _doc(ctx, text=True)
        item_by_id = {str(item.id): item for item in ctx.items}
        selected_docs = jq.compile(self.foreach).input_value(doc).all()

        progress_total = len(selected_docs)
        if self.mode == "buffered":
            progress_total += sum(len(jq.compile(self.input).input_value(item).first()) for item in selected_docs)

        progress = 0
        def advance(message: str) -> None:
            nonlocal progress
            progress += 1
            ctx.progress(progress, progress_total, message)

        ctx.progress(0, progress_total, "extracting")
        sem = asyncio.Semaphore(self.concurrency)
        for done, task in enumerate(asyncio.as_completed([self._extract(ctx, item_doc, item_by_id, sem, advance) for item_doc in selected_docs]), 1):
            item_doc, output = await task
            if "id" in item_doc:
                item = item_by_id[item_doc["id"]]
                current_item_doc = {
                    "id": str(item.id),
                    "source_uri": item.source_uri,
                    "metadata": item.metadata,
                }
                item.metadata = _assign(current_item_doc, self.path, output)["metadata"]
            else:
                ctx.metadata = _assign({"metadata": ctx.metadata}, self.path, output)["metadata"]
            advance(f"extracted {done}/{len(selected_docs)}")

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"items_processed": len(selected_docs), "path": self.path})

    async def _extract(self, ingestion_ctx: IngestionContext, item_doc: Any, item_by_id: dict[str, KnowledgeItem], sem: asyncio.Semaphore, advance: Callable[[str], None]) -> tuple[dict[str, Any], Any]:
        async with sem:
            item_doc = cast(dict[str, Any], item_doc)
            prompt_input = jq.compile(self.input).input_value(item_doc).first()
            item_ids: list[str] = prompt_input if isinstance(prompt_input, list) and all(item_id in item_by_id for item_id in prompt_input) else []
            prompt_context = jq.compile(self.context).input_value(item_doc).first() if self.context else None

            user_prompt: str | list[Any]
            if self.mode == "buffered":
                user_prompt = self.prompt
            elif item_ids:
                user_prompt = [
                    self.prompt,
                    *([prompt_context] if prompt_context is not None else []),
                    *[await _content(item_by_id[item_id], ingestion_ctx) for item_id in item_ids],
                ]
            else:
                user_prompt = [self.prompt, *([prompt_context] if prompt_context is not None else []), prompt_input]

            hooks = Hooks()

            @hooks.on.after_output_process
            async def validate_output(_, *, output_context, output):
                extracted = output.model_dump(mode="json")["output"]
                failures = [assertion.feedback for assertion in self.assertions if not jq.compile(assertion.expr, args={"item": item_doc}).input_value(extracted).first()]
                if failures:
                    raise ModelRetry("\n".join(failures))
                return output


            async def prepare_see_next(ctx: RunContext[ExtractAgentDeps], tool_def: ToolDefinition):
                return tool_def if self.mode == "buffered" and ctx.deps.cursor < len(ctx.deps.buffered_ids) else None


            agent = self._agent(ingestion_ctx, deps_type=ExtractAgentDeps, output_type=NativeOutput(self._output_type(item_doc), strict=True), capabilities=[hooks])

            @agent.tool(prepare=prepare_see_next, sequential=True)
            async def see_next(ctx: RunContext[ExtractAgentDeps]) -> list[Any]:
                if ctx.deps.cursor >= len(ctx.deps.buffered_ids):
                    return ["No items remain. Return the structured output now."]
                item = item_by_id[ctx.deps.buffered_ids[ctx.deps.cursor]]
                ctx.deps.cursor += 1
                advance(f"read {ctx.deps.cursor}/{len(ctx.deps.buffered_ids)}")
                content = [f"Item {ctx.deps.cursor} of {len(ctx.deps.buffered_ids)}"]
                if ctx.deps.cursor >= len(ctx.deps.buffered_ids):
                    content.append("This is the final item. You can no longer call this tool.")
                return [*content, await _content(item, ingestion_ctx)]

            async with agent:
                res = await agent.run(user_prompt, deps=ExtractAgentDeps(cursor=0, buffered_ids=tuple(item_ids)))
            return item_doc, res.output.model_dump(mode="json")["output"]


def _default_update() -> list[Literal["metadata", "items"]]:
    return ["metadata"]


class TransformAgent(BaseModel, Step):
    """Applies a jq transform to the pipeline document and writes back selected keys."""

    value: JqExpression = Field(..., description="jq expression producing the transformed document")
    update: list[Literal["metadata", "items"]] = Field(default_factory=_default_update, description="Document keys written back to the pipeline")

    async def run(self, ctx: IngestionContext) -> StepResult:
        result = jq.compile(self.value).input_value(await _doc(ctx, text=True)).first()
        if not isinstance(result, dict):
            raise ValueError("Transform value must produce an object")
        if "metadata" in self.update:
            ctx.metadata = result.get("metadata", {})
        if "items" in self.update:
            ctx.items = [_item_from_doc(item) for item in result.get("items", [])]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"update": self.update})


class PromptAgent(BaseModel, Step, AgentMixin):
    """Write model output to pipeline metadata."""

    prompt: str = Field(description="Instructions for rewriting content")
    input: JqExpression = Field(..., description="jq expression selecting prompt input")
    path: JqExpression = Field(..., description="jq path where model output is written")

    async def run(self, ctx: IngestionContext) -> StepResult:
        doc = await _doc(ctx, text=True)
        value = jq.compile(self.input).input_value(doc).first()
        async with self._agent(ctx) as agent:
            result = await agent.run([self.prompt, value])
        ctx.metadata = _assign(doc, self.path, result.output)["metadata"]
        return StepResult(status=StepStatus.SUCCESS)


class TranslateAgent(BaseModel, Step):
    """Translates text or structured metadata in place."""

    src: str | None = Field(..., description="Source language or null for automatic detection")
    dst: str = Field(..., description="Destination language")
    input: JqExpression = Field(..., description="jq expression selecting data to translate")
    path: JqExpression = Field(..., description="jq path where translated data is written")
    skip_fields: list[str] = Field(default_factory=list, description="Field path patterns excluded from translation")
    model: KnownTranslationModelName = Field(default=settings.default_translation_model, description="Translation model")

    async def run(self, ctx: IngestionContext) -> StepResult:
        doc = await _doc(ctx, text=True)
        async with create_translator(self.model) as translator:
            output = await translator.translate(self.src, self.dst, jq.compile(self.input).input_value(doc).first(), self.skip_fields)
        ctx.metadata = _assign(doc, self.path, output)["metadata"]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"path": self.path})


class SpeechAgent(BaseModel, Step):
    """Synthesizes item text into audio, replacing item content in place."""

    foreach: JqExpression = Field(default=".items[]", description="jq expression selecting items to synthesize")
    input: JqExpression = Field(default=".text", description="jq expression selecting synthesis input")
    voice: str | None = Field(default=None, description="Azure Speech voice name")
    format: SpeechFormat = Field(default=SpeechFormat.MP3, description="Synthesized audio output format")
    model: KnownSpeechModelName = Field(default=settings.default_speech_model, description="Speech model")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to synthesize")

        doc = await _doc(ctx, text=True)
        item_by_id = {str(item.id): item for item in ctx.items}
        selected = jq.compile(self.foreach).input_value(doc).all()
        if not selected:
            return StepResult(status=StepStatus.SKIPPED, error="No items selected for synthesis")
        values = [jq.compile(self.input).input_value(item).first() for item in selected]
        if any(not isinstance(value, str) for value in values):
            raise ValueError("Speech input must resolve to text")

        tts_model = ctx.tts_factory(self.model, TTSBuilderParams(ctx=ctx, voice=self.voice))
        ctx.progress(0, len(selected), "synthesizing audio")
        for n, (item_doc, text) in enumerate(zip(selected, values, strict=True), 1):
            item = item_by_id[item_doc["id"]]
            audio = await tts_model(ByteContent.from_bytes(text.encode(), content_type="text/plain"), self.format)
            item.content = audio
            ctx.progress(n, len(selected), f"synthesized {n}/{len(selected)}")

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"synthesized_count": len(selected), "format": self.format.value})
