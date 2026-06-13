"""Generic structured-output metadata extraction."""

from typing import Any, Literal

from datamodel_code_generator import GenerateConfig
from datamodel_code_generator.dynamic import generate_dynamic_models
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from ..core.model import AgentMixin, IngestionContext, Step, StepResult, StepStatus


class StructureAgent(BaseModel, Step, AgentMixin):
    prompt: str = Field(...)
    json_schema: dict[str, Any] = Field(...)
    metadata_key: str | None = Field(None)
    method: Literal["set", "append", "update"] = Field("set")

    _output_key: str = PrivateAttr("output")
    _output_type: type[BaseModel] = PrivateAttr()

    @model_validator(mode="after")
    def build_output_type(self):
        schema = self.json_schema
        if self.method == "update":
            if schema.get("type") != "object": raise ValueError("update requires object schema")
        else:
            if self.metadata_key is None: raise ValueError(f"{self.method} requires metadata_key")
            if self.method == "append" and schema.get("type") != "array": raise ValueError(f"{self.method} requires array schema")
            schema = {"type": "object", "properties": {self._output_key: schema}, "required": [self._output_key]}

        self._output_type = generate_dynamic_models(schema, config=GenerateConfig(formatters=[]))["Model"]
        return self

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for structure extraction")

        agent = self._agent(ctx)

        for item in ctx.items:
            res = await agent.run(self.prompt.format(text=item.decode()), output_type=self._output_type)
            data = res.output.model_dump() if isinstance(res.output, BaseModel) else res.output
            if self.method == "update":
                item.metadata.update(data)
            elif self.method == "set":
                assert self.metadata_key is not None
                item.metadata[self.metadata_key] = data[self._output_key]
            else:
                assert self.metadata_key is not None
                item.metadata[self.metadata_key] = [*item.metadata.get(self.metadata_key, []), *data[self._output_key]]

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"items_processed": len(ctx.items)})
