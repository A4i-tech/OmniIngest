"""Other miscellaneous agents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import Step, StepResult


class PipelineAgent(BaseModel, Step):
    """Executes another pipeline configuration as a single parent step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    pipeline: Any = Field(description="The pipeline to run as a step. This can also be path to a pipeline file.")
    config: dict[str, str] = Field(default_factory=dict, description="Substitution parameters for the pipeline")

    @model_validator(mode="before")
    @classmethod
    def resolve_pipeline(cls, data: dict) -> dict:
        from ..core.pipeline import PipelineRunner, create_pipeline_from_config

        pipeline = data.get("pipeline")
        if pipeline is not None and not isinstance(pipeline, PipelineRunner):
            path = Path(pipeline)
            if not path.is_file():
                raise ValueError(f"Pipeline config file not found: {path}")
            data["pipeline"] = create_pipeline_from_config(path, config_args=TypeAdapter(dict[str, str]).validate_python(data.get("config", {})))
        return data

    async def run(self, ctx: IngestionContext) -> StepResult:
        from ..core.pipeline import PipelineRunner
        assert isinstance(self.pipeline, PipelineRunner)
        results = await self.pipeline.run(ctx)
        return results[-1]
