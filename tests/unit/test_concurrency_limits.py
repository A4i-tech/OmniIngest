"""Concurrency limits of the OCR and rasterize steps.

`concurrency` (OCR) and `workers` (rasterize) must bound the work in flight, and
rasterize must read an item's bytes only once it holds a worker slot. Every test
asserts both directions: the limit is never exceeded and it is actually reached,
so a step that runs everything serially fails as well as one that runs everything
at once.
"""

import asyncio
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import omni_ingest.port.ocr as ocr_port
from omni_ingest.agent.document import RasterizeAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource
from omni_ingest.core.ocr import OcrBuilderParams, OcrOutputFormat
from omni_ingest.core.pipeline import IngestionContext

PDF = Path(__file__).parent.parent / "resources" / "pdf" / "one_image.pdf"
ITEMS = 8


def _ctx(items: list[KnowledgeItem] | None = None) -> IngestionContext:
    return IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=items or [])


class _InFlight:
    """Counts overlapping calls; `peak` is the largest number running at once."""

    def __init__(self) -> None:
        self.current = self.peak = 0

    async def hold(self, seconds: float = 0.02) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)
        await asyncio.sleep(seconds)
        self.current -= 1


async def _bytes(n: int):
    for i in range(n):
        yield ByteContent.from_bytes(f"doc{i}".encode(), "application/pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
async def test_llm_ocr_never_exceeds_concurrency_and_uses_it(limit):
    calls = _InFlight()

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, **_):
            await calls.hold()
            return SimpleNamespace(output="text")

    class FakeMixin:
        def _agent(self, ctx):
            return FakeAgent()

    with patch.object(ocr_port, "_DefaultAgentMixin", FakeMixin):
        ocr = ocr_port.llm_ocr_builder(OcrBuilderParams(ctx=_ctx(), format=OcrOutputFormat.MARKDOWN, concurrency=limit))
        results = [pair async for pair in ocr(_bytes(ITEMS))]

    assert len(results) == ITEMS  # every item was processed
    assert calls.peak == limit


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
async def test_sarvam_ocr_never_exceeds_concurrency_and_uses_it(limit):
    jobs = _InFlight()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("upload/upload.md", "text")

    class FakeDocAi:
        """A job is in flight from `digitise` until its result has been downloaded."""

        def __init__(self) -> None:
            self.polls: dict[str, int] = {}

        def digitise(self, **_):
            jobs.current += 1
            jobs.peak = max(jobs.peak, jobs.current)
            job_id = f"job{len(self.polls)}"
            self.polls[job_id] = 0
            return SimpleNamespace(job_id=job_id)

        def get_status(self, job_id):
            self.polls[job_id] += 1
            status = "running" if self.polls[job_id] == 1 else "completed"  # forces one real wait per job
            return SimpleNamespace(status=status, usage=SimpleNamespace(pages_processed=0, pages_total=1))

        def get_download_url(self, job_id):
            return SimpleNamespace(url="https://example.invalid/result.zip", headers={})

    async def fake_read_resources(url, headers=None):
        await asyncio.sleep(0.01)
        jobs.current -= 1  # result downloaded, the job is finished
        yield SimpleNamespace(content=ByteContent.from_bytes(archive.getvalue()))

    real_sleep = asyncio.sleep

    async def short_sleep(_):
        await real_sleep(0.02)

    with (
        patch.object(ocr_port, "SarvamAI", lambda: SimpleNamespace(doc_ai=FakeDocAi())),
        patch.object(ocr_port, "read_resources", fake_read_resources),
        patch.object(ocr_port.asyncio, "sleep", short_sleep),
    ):
        ocr = ocr_port.sarvam_ocr_builder(OcrBuilderParams(ctx=_ctx(), format=OcrOutputFormat.MARKDOWN, concurrency=limit))
        results = [pair async for pair in ocr(_bytes(ITEMS))]

    assert len(results) == ITEMS
    assert jobs.peak == limit


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [1, 2])
async def test_rasterize_reads_content_only_while_holding_a_worker_slot(workers):
    data = PDF.read_bytes()
    items = [KnowledgeItem(content=ByteContent.from_bytes(data, "application/pdf")) for _ in range(ITEMS)]
    read_but_unfinished = 0  # items whose bytes are in memory and whose job has not finished
    peak_in_memory = 0
    original_read, original_job = ByteContent.read, RasterizeAgent._rasterize_pdf

    async def tracked_read(self, ctx):
        nonlocal read_but_unfinished, peak_in_memory
        content = await original_read(self, ctx)
        await asyncio.sleep(0.05)  # a slow read, so every task that may read does so before the first job finishes
        read_but_unfinished += 1
        peak_in_memory = max(peak_in_memory, read_but_unfinished)
        return content

    def tracked_job(self, item, content):
        nonlocal read_but_unfinished
        try:
            return original_job(self, item, content)
        finally:
            read_but_unfinished -= 1

    with patch.object(ByteContent, "read", tracked_read), patch.object(RasterizeAgent, "_rasterize_pdf", tracked_job):
        result = await RasterizeAgent(dpi=72, workers=workers).run(_ctx(items))

    assert result.metadata["count"] == ITEMS  # all items were rasterized
    assert peak_in_memory == workers  # bounded by the worker slots, and the slots are used
