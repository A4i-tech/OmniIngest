"""Generic structured-output metadata extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import jq
from datamodel_code_generator import GenerateConfig
from datamodel_code_generator.dynamic import generate_dynamic_models
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic_ai import BinaryContent, ModelRetry

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import AgentMixin, Step, StepResult, StepStatus


class StructureAgent(BaseModel, Step, AgentMixin):
    """
    Extracts structured metadata from text items using an LLM.

    The supplied JSON schema defines the modeled output. jq expressions select
    prompt input and assign the structured output back to the runtime document.
    """

    prompt: str = Field(..., description="Instruction prompt")
    json_schema: dict[str, Any] = Field(..., description="JSON schema for the structured model output")
    target: str = Field(..., description="jq assignment expression where $output is the structured result")
    mode: Literal["direct", "buffered"] = Field(default="direct", description="How selected input is supplied to the model")
    foreach: str | None = Field(default=".items[]", description="jq expression selecting documents to process; null runs once over the whole context")
    input: str = Field(default=".text", description="jq expression selecting prompt input from each selected document")
    assertions: list[dict[str, str]] = Field(default_factory=list, description="jq assertions evaluated after target assignment")

    _output_key: str = PrivateAttr("output")
    _output_type: type[BaseModel] = PrivateAttr()

    @model_validator(mode="after")
    def build_output_type(self):
        schema = {"type": "object", "properties": {self._output_key: self.json_schema}, "required": [self._output_key]}
        self._output_type = generate_dynamic_models(schema, config=GenerateConfig(formatters=[]))["Model"]
        return self

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for structure extraction")

        agent = self._agent(ctx)
        item_docs = [
            {"id": str(item.id), "source_uri": item.source_uri, "text": item.decode() if item.content_type.startswith("text/") else None, "metadata": item.metadata}
            for item in ctx.items
        ]
        doc: dict[str, Any] = {"metadata": ctx.metadata, "items": item_docs}
        item_by_id = {item_doc["id"]: item for item_doc, item in zip(item_docs, ctx.items, strict=True)}
        selected_docs = [doc] if self.foreach is None else jq.compile(self.foreach).input_value(doc).all()

        processed = 0
        for item_doc in selected_docs:
            item_doc = cast(dict[str, Any], item_doc)
            prompt_input: Any = jq.compile(self.input).input_value(item_doc).first()
            input_ids = prompt_input.get("ids") if isinstance(prompt_input, dict) else prompt_input
            item_ids: list[str] = input_ids if isinstance(input_ids, list) and all(item_id in item_by_id for item_id in input_ids) else []
            user_prompt: Any
            if self.mode == "buffered":
                buffered_ids = tuple(item_ids)
                cursor = 0

                @agent.tool_plain
                def see_next() -> list[Any]:
                    nonlocal cursor
                    if cursor >= len(buffered_ids):  # noqa: B023
                        raise ModelRetry("You have read all items, you cannot use this tool anymore.")
                    item = item_by_id[buffered_ids[cursor]]  # noqa: B023
                    cursor += 1
                    return [
                        f"Page {item.metadata['page']} / {len(ctx.items)}",
                        BinaryContent(data=item.content, media_type=item.content_type) if not item.content_type.startswith("text/") else item.decode(),
                    ]

                user_prompt = self.prompt
            elif item_ids:
                user_prompt = [
                    self.prompt,
                    *([prompt_input["context"]] if isinstance(prompt_input, dict) and "context" in prompt_input else []),
                    *(BinaryContent(data=item_by_id[item_id].content, media_type=item_by_id[item_id].content_type) for item_id in item_ids),
                ]
            else:
                user_prompt = [self.prompt, prompt_input]

            res = await agent.run(user_prompt, output_type=self._output_type)
            data = res.output.model_dump(mode="json") if isinstance(res.output, BaseModel) else res.output
            output = data[self._output_key]
            updates_root = self.foreach is None or "id" not in item_doc
            updated = cast(dict[str, Any], jq.compile(self.target, args={self._output_key: output, "item": item_doc}).input_value(doc if updates_root else item_doc).first())

            if updates_root:
                doc.update(updated)
                ctx.metadata.clear()
                ctx.metadata.update(doc["metadata"])
            else:
                item_by_id[updated["id"]].metadata.clear()
                item_by_id[updated["id"]].metadata.update(updated["metadata"])

            for assertion in self.assertions:
                if not jq.compile(assertion["expr"], args={self._output_key: output}).input_value(updated).first():
                    metadata = {self._output_key: output} if self.foreach is None else {"items_processed": processed, self._output_key: output}
                    return StepResult(status=StepStatus.FAILURE, error=assertion.get("message", f"Assertion failed: {assertion['expr']}"), items=ctx.items, metadata=metadata)
            processed += 1

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"target": self.target} if self.foreach is None else {"items_processed": processed})
