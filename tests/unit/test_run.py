import pytest

import omni_ingest
from omni_ingest.core.model import StepResult, StepStatus


class FakeRunner:
    async def flatten(self, ctx):
        for i in range(3):
            yield type("Ctx", (), {"run_id": i})()

    async def run(self, ctx):
        if ctx.run_id == 1:
            raise RuntimeError("boom")
        return [StepResult(status=StepStatus.SUCCESS)]


class FakeOutput:
    async def write_many(self, ctxs):
        self.written = [c.run_id for c in ctxs]


@pytest.mark.asyncio
async def test_failing_run_keeps_other_runs_results(monkeypatch):
    monkeypatch.setattr(omni_ingest, "create_pipeline_from_config", lambda *a, **k: FakeRunner())
    output = FakeOutput()

    results = await omni_ingest.run("p.yaml", "r", output=output)

    assert len(results) == 2
    assert output.written == [0, 2]
