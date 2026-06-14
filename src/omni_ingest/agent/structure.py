"""Generic structured-output metadata extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jq
from datamodel_code_generator import GenerateConfig
from datamodel_code_generator.dynamic import generate_dynamic_models
from pydantic import BaseModel, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import AgentMixin, Step, StepResult, StepStatus


class StructureAgent(BaseModel, Step, AgentMixin):
    """
    Extracts structured metadata from text items using an LLM.

    The supplied JSON schema defines the modeled output. jq expressions select
    prompt input and assign the structured output back to the runtime document.
    """

    prompt: str = Field(..., description="Prompt template. Use {text} for item text")
    json_schema: dict[str, Any] = Field(..., description="JSON schema for the structured model output")
    target: str = Field(..., description="jq assignment expression where $output is the structured result")
    foreach: str | None = Field(default=".items[]", description="jq expression selecting documents to process; null runs once over the whole context")
    input: str = Field(default=".text", description="jq expression selecting text from each selected document")
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
        doc: dict[str, Any] = {
            "metadata": ctx.metadata,
            "items": [
                {"id": str(item.id), "source_uri": item.source_uri, "text": item.decode(), "metadata": item.metadata}
                for item in ctx.items
            ]
        }

        if self.foreach is None:
            text = jq.compile(self.input).input_value(doc).first()
            res = await agent.run(self.prompt.format(text=text), output_type=self._output_type)
            data = res.output.model_dump() if isinstance(res.output, BaseModel) else res.output
            output = data[self._output_key]
            doc.update(cast(dict[str, Any], jq.compile(self.target, args={self._output_key: output}).input_value(doc).first()))
            ctx.metadata.clear()
            ctx.metadata.update(doc["metadata"])
            if error := _assertion_error(self.assertions, doc, output):
                return StepResult(status=StepStatus.FAILURE, error=error, items=ctx.items, metadata={self._output_key: output})
            return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"target": self.target})

        item_by_id = {item_doc["id"]: item for item_doc, item in zip(cast(list[dict[str, Any]], doc["items"]), ctx.items, strict=True)}
        processed = 0
        for item_doc in jq.compile(self.foreach).input_value(doc).all():
            item_doc = cast(dict[str, Any], item_doc)
            text = jq.compile(self.input).input_value(item_doc).first()
            res = await agent.run(self.prompt.format(text=text), output_type=self._output_type)
            data = res.output.model_dump() if isinstance(res.output, BaseModel) else res.output
            output = data[self._output_key]
            updated = cast(dict[str, Any], jq.compile(self.target, args={self._output_key: output}).input_value(item_doc).first())
            item_by_id[updated["id"]].metadata.clear()
            item_by_id[updated["id"]].metadata.update(updated["metadata"])
            if error := _assertion_error(self.assertions, updated, output):
                return StepResult(status=StepStatus.FAILURE, error=error, items=ctx.items, metadata={"items_processed": processed, self._output_key: output})
            processed += 1

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"items_processed": processed})


def _assertion_error(assertions: list[dict[str, str]], doc: dict[str, Any], output: Any) -> str | None:
    for assertion in assertions:
        if not jq.compile(assertion["expr"], args={"output": output}).input_value(doc).first():
            return assertion.get("message", f"Assertion failed: {assertion['expr']}")
    return None
