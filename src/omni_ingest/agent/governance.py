from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, Field, PositiveInt, confloat

from ..core.model import AgentMixin, IngestionContext, Step, StepResult, StepStatus
from ..core.pipeline import step


@step("ingestion_logger")
class IngestionLoggerAgent(BaseModel, Step):
    """
    Records operational metadata for ingestion pipeline execution.

    This agent logs audit information—including execution timestamps
    and item counts—to the metadata store, providing a record for
    monitoring and auditing purposes.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.store or not ctx.run_id:
            return StepResult(status=StepStatus.SKIPPED, error="No store/run context")

        audit_entry = {
            "event": "pipeline_checkpoint",
            "timestamp": str(datetime.now(UTC)),
            "items_in_context": len(ctx.items) if ctx.items else 0,
        }
        return StepResult(status=StepStatus.SUCCESS, metadata=audit_entry)


@step("lineage")
class DataLineageAgent(BaseModel, Step):
    """
    Maintains data provenance and lineage information for KnowledgeItems.

    This agent attaches metadata documenting the origin and transformation
    history of each KnowledgeItem, ensuring traceability from the source
    data to the finalized index.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        lineage = {
            "run_id": str(ctx.run_id),
            "tenant_id": str(ctx.tenant_id),
            "source_uri": ctx.metadata.get("source_uri") or ctx.metadata.get("file_path"),
            "items_count": len(ctx.items) if ctx.items else 0,
            "timestamp": str(datetime.now(UTC)),
        }
        if ctx.items:
            for item in ctx.items:
                item.lineage = lineage

        return StepResult(status=StepStatus.SUCCESS, metadata=lineage)


class QualityReport(BaseModel):
    overall_score: Annotated[float, confloat(ge=0, le=1)] = Field(..., description="Quality score")
    issues: list[str] = Field(..., description="List of quality issues found")


@step("quality")
class QualityVerifierAgent(BaseModel, Step, AgentMixin):
    """
    Evaluates the quality of extracted text based on predefined criteria.

    This agent samples generated knowledge items and utilizes an LLM to
    provide a quality score. It identifies potential issues and can be
    configured to halt the pipeline if the score falls below a specified
    threshold.
    """

    sample_size: PositiveInt = Field(3, description="Number of items to sample for quality verification")
    min_quality_score: Annotated[float, confloat(ge=0, le=1)] = Field(0.7, description="Minimum acceptable quality score")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to verify")

        samples = ctx.items[: self.sample_size]

        prompt = """Evaluate the quality of the following extracted text samples.
Return a JSON object with:
- "overall_score": float between 0 and 1
- "issues": list of strings describing any problems found

Samples:
"""
        for i, item in enumerate(samples):
            prompt += f"\n--- Sample {i + 1} ---\n{item.content[:500]}\n"

        agent = self._agent(ctx)
        res = await agent.run(prompt, output_type=QualityReport)
        data = res.output
        score = data.overall_score
        passed = score >= self.min_quality_score

        if not passed:
            error_msg = f"Quality check failed: score {score:.2f} < {self.min_quality_score}"
            return StepResult(
                status=StepStatus.FAILURE,
                error=error_msg,
                metadata={"quality_score": score, "issues": data.issues},
            )

        ctx.metadata["quality_report"] = data.model_dump()
        return StepResult(
            status=StepStatus.SUCCESS,
            items=ctx.items,
            metadata={"quality_score": score, "samples_checked": len(samples), "issues": data.issues},
        )
