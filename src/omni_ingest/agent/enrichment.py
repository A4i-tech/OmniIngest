"""Agents for structured metadata extraction and enrichment."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import jq
from datamodel_code_generator import GenerateConfig
from datamodel_code_generator.dynamic import generate_dynamic_models
from pydantic import BaseModel, Field, PositiveInt
from pydantic_ai import BinaryContent, ModelRetry, NativeOutput
from pydantic_ai.capabilities import Hooks

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import AgentMixin, JqExpression, KnowledgeItem, Step, StepResult, StepStatus


def _assign(doc: dict[str, Any], path: str, output: Any) -> dict[str, Any]:
    return cast(dict[str, Any], jq.compile(f"{path} |= if type == \"object\" and ($output | type) == \"object\" then . + $output else $output end", args={"output": output}).input_value(doc).first())


def _doc(ctx: IngestionContext, text: bool = False) -> dict[str, Any]:
    return {
        "metadata": ctx.metadata,
        "items": [
            {"id": str(item.id), "source_uri": item.source_uri, "metadata": item.metadata}
            | ({"text": item.decode() if item.content_type.startswith("text/") else None} if text else {})
            for item in ctx.items
        ],
    }


def _content(item: KnowledgeItem) -> Any:
    return item.decode() if item.content_type.startswith("text/") else BinaryContent(data=item.content, media_type=item.content_type)


class ExtractAssertion(BaseModel):
    expr: JqExpression = Field(..., description="jq expression that must evaluate truthy against extracted output")
    feedback: str = Field(..., description="Feedback sent to the model when the assertion fails")


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

        doc = _doc(ctx, text=True)
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

    async def _extract(self, ctx: IngestionContext, item_doc: Any, item_by_id: dict[str, KnowledgeItem], sem: asyncio.Semaphore, advance: Callable[[str], None]) -> tuple[dict[str, Any], Any]:
        async with sem:
            agent = self._agent(ctx)
            hooks = Hooks()

            @hooks.on.after_output_process
            async def validate_output(_, *, output_context, output):
                extracted = output.model_dump(mode="json")["output"]
                failures = [assertion.feedback for assertion in self.assertions if not jq.compile(assertion.expr, args={"item": item_doc}).input_value(extracted).first()]
                if failures:
                    raise ModelRetry("\n".join(failures))
                return output

            item_doc = cast(dict[str, Any], item_doc)
            prompt_input = jq.compile(self.input).input_value(item_doc).first()
            item_ids: list[str] = prompt_input if isinstance(prompt_input, list) and all(item_id in item_by_id for item_id in prompt_input) else []
            prompt_context = jq.compile(self.context).input_value(item_doc).first() if self.context else None

            if self.mode == "buffered":
                buffered_ids = tuple(item_ids)
                cursor = 0

                @agent.tool_plain
                async def see_next() -> list[Any]:
                    nonlocal cursor
                    if cursor >= len(buffered_ids):
                        raise ModelRetry("You have read all items, you cannot use this tool anymore.")
                    item = item_by_id[buffered_ids[cursor]]
                    cursor += 1
                    advance(f"read {cursor}/{len(buffered_ids)}")
                    return [
                        f"Physical page {item.metadata['page']}",
                        _content(item),
                    ]

                user_prompt: Any = self.prompt
            elif item_ids:
                user_prompt = [
                    self.prompt,
                    *([prompt_context] if prompt_context is not None else []),
                    *(_content(item_by_id[item_id]) for item_id in item_ids),
                ]
            else:
                user_prompt = [self.prompt, *([prompt_context] if prompt_context is not None else []), prompt_input]

            res = await agent.run(user_prompt, output_type=NativeOutput(self._output_type(item_doc), strict=True), capabilities=[hooks])
            return item_doc, res.output.model_dump(mode="json")["output"]

class TransformAgent(BaseModel, Step):
    """Applies jq transforms to root metadata."""

    value: JqExpression = Field(..., description="jq expression producing transformed output")
    path: JqExpression = Field(..., description="jq path where transformed output is written")

    async def run(self, ctx: IngestionContext) -> StepResult:
        doc = _doc(ctx)
        ctx.metadata = _assign(doc, self.path, jq.compile(self.value).input_value(doc).first())["metadata"]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"path": self.path})


class KGNode(BaseModel):
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class KGEdge(BaseModel):
    source: str
    target: str
    type: str


class KGData(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]


class KnowledgeGraphBuilderAgent(BaseModel, Step, AgentMixin):
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for KG extraction")

        agent = self._agent(ctx)
        all_nodes, all_edges = [], []
        for item in ctx.items:
            res = await agent.run("Directly extract entities and their relationships from the following text into a graph format. Content: " + item.decode(), output_type=KGData)
            all_nodes.extend([n.model_dump() for n in res.output.nodes])
            all_edges.extend([e.model_dump() for e in res.output.edges])

        ctx.metadata["kg_data"] = {"nodes": all_nodes, "edges": all_edges}
        return StepResult(status=StepStatus.SUCCESS, metadata={"nodes": len(all_nodes), "edges": len(all_edges)})
