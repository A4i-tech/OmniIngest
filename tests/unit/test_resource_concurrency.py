"""Concurrency limit across the resources of one run (`run_flattened` and the CLI).

A glob or directory can match many resources. Each test asserts both directions:
the number of runs in flight never exceeds the limit and actually reaches it, so a
cap that serialises everything fails as well as one that lets everything through.
"""

import asyncio
import json
import sys

import pytest

from omni_ingest import cli
from omni_ingest.core import pipeline
from omni_ingest.core.model import Step, StepResult, StepStatus
from omni_ingest.core.pipeline import DEFAULT_MAX_CONCURRENT_RESOURCES, IngestionContext, PipelineRunner
from omni_ingest.port.metadata_store import NullMetadataStore

FILES = 12


class _Probe(Step):
    """Counts runs in flight across every instance."""

    current = peak = 0

    async def run(self, ctx):
        type(self).current += 1
        type(self).peak = max(type(self).peak, type(self).current)
        await asyncio.sleep(0.05)
        type(self).current -= 1
        return StepResult(status=StepStatus.SUCCESS, metadata={})


@pytest.fixture(autouse=True)
def _reset_probe(monkeypatch):
    _Probe.current = _Probe.peak = 0
    monkeypatch.setitem(pipeline._STEP_FACTORY, "probe", _Probe)


@pytest.fixture
def files(tmp_path):
    for i in range(FILES * 2):
        (tmp_path / f"{i}.txt").write_text(f"doc {i}")
    return tmp_path


async def _flatten(files, **kwargs):
    ctx = IngestionContext(store=NullMetadataStore(), resource=str(files / "*.txt"))
    return [r async for r in PipelineRunner(steps=[_Probe()]).run_flattened(ctx, **kwargs)]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
async def test_run_flattened_never_exceeds_max_concurrent_and_uses_it(files, limit):
    results = await _flatten(files, max_concurrent=limit)

    assert len(results) == FILES * 2  # every resource was processed
    assert _Probe.peak == limit


@pytest.mark.asyncio
async def test_run_flattened_default_limit_is_the_documented_default(files):
    results = await _flatten(files)

    assert len(results) == FILES * 2
    assert 1 < _Probe.peak <= DEFAULT_MAX_CONCURRENT_RESOURCES  # concurrent, but bounded


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_run_flattened_rejects_a_limit_below_one(files, limit):
    with pytest.raises(ValueError, match="at least 1"):
        await asyncio.wait_for(_flatten(files, max_concurrent=limit), timeout=5)  # 0 must fail, not wait forever


def _run_cli(monkeypatch, tmp_path, files, *flags):
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text("steps:\n- agent: probe\n")
    monkeypatch.setattr(sys, "argv", ["omni-ingest", str(pipeline_file), "--input", str(files / "*.txt"), "--output", "-", "--format", "json", "-q", *flags])
    return asyncio.run(cli._main())


@pytest.mark.parametrize("limit", [1, 3])
def test_cli_flag_caps_resources_in_flight(monkeypatch, tmp_path, capsys, limit):
    files = tmp_path / "in"
    files.mkdir()
    for i in range(FILES):
        (files / f"{i}.txt").write_text(f"doc {i}")

    assert _run_cli(monkeypatch, tmp_path, files, "--max-concurrent-resources", str(limit)) == 0

    assert _Probe.peak == limit
    assert len(json.loads(capsys.readouterr().out)["runs"]) == FILES  # nothing was dropped


def test_cli_default_limit_bounds_resources_in_flight(monkeypatch, tmp_path, capsys):
    files = tmp_path / "in"
    files.mkdir()
    for i in range(FILES * 2):
        (files / f"{i}.txt").write_text(f"doc {i}")

    assert _run_cli(monkeypatch, tmp_path, files) == 0

    assert 1 < _Probe.peak <= DEFAULT_MAX_CONCURRENT_RESOURCES


@pytest.mark.parametrize("limit", ["0", "-2"])
def test_cli_rejects_a_limit_below_one(monkeypatch, tmp_path, capsys, limit):
    files = tmp_path / "in"
    files.mkdir()
    (files / "a.txt").write_text("doc")

    with pytest.raises(SystemExit) as exit_info:
        _run_cli(monkeypatch, tmp_path, files, "--max-concurrent-resources", limit)

    assert exit_info.value.code == 2
    assert "at least 1" in capsys.readouterr().err
