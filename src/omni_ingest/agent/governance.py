from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import ResolvedResource, Step, StepResult, StepStatus


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


class DataLineageAgent(BaseModel, Step):
    """
    Maintains data provenance and lineage information for KnowledgeItems.

    This agent attaches metadata documenting the origin and transformation
    history of each KnowledgeItem, ensuring traceability from the source
    data to the finalized index.
    """

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        lineage = {
            "run_id": str(ctx.run_id),
            "tenant_id": str(ctx.tenant_id),
            "source_uri": ctx.resource.uri,
            "items_count": len(ctx.items) if ctx.items else 0,
            "timestamp": str(datetime.now(UTC)),
        }
        return StepResult(status=StepStatus.SUCCESS, metadata=lineage)

