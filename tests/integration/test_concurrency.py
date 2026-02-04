import asyncio

import pytest

from omni_ingest.agent.chunking import SentenceChunkingAgent
from omni_ingest.core.model import IngestionContext, KnowledgeItem, StepStatus
from omni_ingest.core.pipeline import PipelineRunner


@pytest.mark.asyncio
async def test_parallel_pipeline_execution(metadata_store, tenant):
    num_runs = 5

    async def run_pipeline(idx):
        ctx = IngestionContext(
            tenant_id=tenant.id,
            items=[KnowledgeItem(content=f"Parallel content {idx}", source_uri=f"file://{idx}.txt")],
            store=metadata_store,
            metadata={"text": f"Parallel content {idx}"},
        )
        runner = PipelineRunner(steps=[SentenceChunkingAgent(chunk_size=10, overlap=0)])
        results = await runner.run(ctx)
        return results

    tasks = [run_pipeline(i) for i in range(num_runs)]
    all_results = await asyncio.gather(*tasks)

    assert len(all_results) == num_runs
    for results in all_results:
        assert results[0].status == StepStatus.SUCCESS
        assert len(results[0].items) > 0
    async with metadata_store.session_factory() as s:
        from sqlalchemy import func, select

        from omni_ingest.port.metadata_store import PipelineStepRun

        res = await s.execute(select(func.count(PipelineStepRun.id)))
        count = res.scalar()
        assert count == num_runs
