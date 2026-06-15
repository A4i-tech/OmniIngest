"""Command-line entrypoint for running Omni Ingest pipeline YAML files."""

import argparse
import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import logfire
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

load_dotenv(Path.cwd() / ".env")  # must run before importing modules that instantiate settings

from omni_ingest.core import event  # noqa: E402
from omni_ingest.core.model import PipelineConfig  # noqa: E402
from omni_ingest.core.output import parse as parse_output  # noqa: E402
from omni_ingest.core.pipeline import IngestionContext, create_pipeline_from_config, lookup_step_id  # noqa: E402


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("pipeline", nargs="?")
    known, _ = bootstrap.parse_known_args()
    try:
        parameters = PipelineConfig.model_validate(yaml.safe_load(Path(known.pipeline).read_text())).parameters if known.pipeline else {}
    except Exception:
        parameters = {}
    properties = parameters.get("properties", {})

    parser = argparse.ArgumentParser(prog="omni-ingest", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("-q", "--quiet", action="store_true")
    for name, schema in properties.items():
        kwargs = {
            "default": schema.get("default", argparse.SUPPRESS),
            "help": schema.get("description"),
            "required": name in parameters.get("required", []),
        }
        if "enum" in schema:
            kwargs["choices"] = schema["enum"]
        if schema.get("type") == "boolean":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["metavar"] = schema.get("type", "VALUE").upper()
            kwargs["type"] = {"integer": int, "number": float, "string": str}.get(schema.get("type"), json.loads)
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    args = parser.parse_args()
    args.params = {name: getattr(args, name) for name in properties if hasattr(args, name)}
    return args


async def _events(progress: Progress, tasks: dict[int, Any], labels: dict[int, str]) -> None:
    async for raw in event.subscribe(populate_with_buffered=False):
        e = cast(Any, raw)
        if e.identifier in {"pipeline_step_begin", "pipeline_step_progress", "pipeline_step_end"}:
            step_id, task_id = id(e.step), tasks[id(e.step)]
            label = labels[step_id]
        else:
            continue

        if e.identifier == "pipeline_step_begin":
            progress.start_task(task_id)
            progress.update(task_id, completed=0, description=f"{label} [cyan]running[/cyan]")
        elif e.identifier == "pipeline_step_progress":
            suffix = f" [dim]{e.message}[/dim]" if e.message else ""
            progress.update(task_id, total=e.total or 1, completed=e.completed, description=f"{label} [cyan]running[/cyan]{suffix}")
        elif e.identifier == "pipeline_step_end":
            color = "green" if e.step_result.status.value == "success" else "red"
            progress.update(task_id, total=1, completed=1, description=f"{label} [{color}]{e.step_result.status.value}[/{color}]")
            progress.stop_task(task_id)


async def _run_input(runner: Any, input_path: Path) -> tuple[IngestionContext, bool]:
    ctx = IngestionContext(resource=input_path)
    results = await runner.run(ctx)
    return ctx, any(r.status.value == "failure" for r in results)


async def _main() -> int:
    logfire.configure(console=False, send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()

    args = _parse_args()
    output = parse_output(args.output)
    runners = [(input_path, create_pipeline_from_config(args.pipeline, config_args=args.params)) for input_path in args.input]
    progress = nullcontext() if args.quiet else Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
        redirect_stdout=True,
        redirect_stderr=True,
    )

    with progress as p:
        listener = None
        if p:
            labels = {
                id(step): f"{input_path.name} {i:02d} {runner.step_descriptions.get(id(step)) or lookup_step_id(step) or type(step).__name__}"
                for input_path, runner in runners
                for i, step in enumerate(runner.steps, 1)
            }
            tasks = {step_id: p.add_task(f"{label} [dim]pending[/dim]", total=1, start=False) for step_id, label in labels.items()}
            listener = asyncio.create_task(_events(p, tasks, labels))
            await asyncio.sleep(0)
        try:
            runs = await asyncio.gather(*[_run_input(runner, input_path) for input_path, runner in runners])
        finally:
            event.shutdown()
            if listener:
                await listener

    await output.write_many([ctx for ctx, _ in runs])
    return int(any(failed for _, failed in runs))


def main() -> None:
    raise SystemExit(asyncio.run(_main()))
