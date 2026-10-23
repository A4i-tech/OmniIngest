from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .core import event
from .core.model import Resource, StepResult, StepStatus
from .core.output import Output, parse
from .core.pipeline import IngestionContext, create_pipeline_from_config

logger = logging.getLogger(__name__)


async def run(pipeline: Path | str, resource: Resource | str, *, output: Output | str | None = None, params: dict[str, Any] | None = None) -> list[StepResult]:
    """Runs a pipeline YAML against a resource and returns the step results of all runs. Outputs are written only for runs without a failed step. Safe to call concurrently."""
    runner = create_pipeline_from_config(Path(pipeline), config_args=params)
    results, done = [], []
    async for ctx in runner.flatten(IngestionContext(resource=resource)):
        try:
            run_results = await runner.run(ctx)
        except Exception:
            logger.exception("Run %s failed", ctx.run_id)
            continue
        finally:
            event.shutdown(ctx.run_id)
        results += run_results
        if all(r.status != StepStatus.FAILURE for r in run_results):
            done.append(ctx)
    if output is not None and done:
        await (parse(output) if isinstance(output, str) else output).write_many(done)
    return results
