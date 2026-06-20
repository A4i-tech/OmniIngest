"""Command-line entrypoint for running Omni Ingest pipeline YAML files."""

import argparse
import asyncio
import json
from pathlib import Path

import logfire
import yaml
from dotenv import load_dotenv
from pydantic import TypeAdapter
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn

load_dotenv(Path.cwd() / ".env")  # must run before importing modules that instantiate settings

from omni_ingest.core import event  # noqa: E402
from omni_ingest.core.model import PipelineConfig, Resource, StepResult, StepStatus  # noqa: E402
from omni_ingest.core.output import parse as parse_output  # noqa: E402
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner, create_pipeline_from_config, lookup_step_id  # noqa: E402


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("pipeline", nargs="?", type=Path)
    known, _ = bootstrap.parse_known_args()
    try:
        parameters = PipelineConfig.model_validate(yaml.safe_load(known.pipeline.read_text())).parameters if known.pipeline else {}
    except Exception:
        parameters = {}
    properties = parameters.get("properties", {})

    parser = argparse.ArgumentParser(prog="omni-ingest", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--input", nargs="+", required=True, type=TypeAdapter(Resource).validate_python)
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


async def _events(progress: Progress, tasks: dict[int, tuple[TaskID, str]]) -> None:
    async for e in event.subscribe(populate_with_buffered=False):
        if not isinstance(e, event._StepMixin):
            continue

        task_id, label = tasks[id(e.step)]
        if isinstance(e, event.PipelineStepBeginEvent):
            progress.start_task(task_id)
            progress.update(task_id, completed=0, description=f"{label} [cyan]running[/cyan]")
        elif isinstance(e, event.PipelineStepProgressEvent):
            suffix = f" [dim]{e.message}[/dim]" if e.message else ""
            progress.update(task_id, total=e.total or 1, completed=e.completed, description=f"{label} [cyan]running[/cyan]{suffix}")
        elif isinstance(e, event.PipelineStepEndEvent):
            color = "green" if e.step_result.status == StepStatus.SUCCESS else "red"
            progress.update(task_id, total=1, completed=1, description=f"{label} [{color}]{e.step_result.status.value}[/{color}]")
            progress.stop_task(task_id)


async def _run(runner: PipelineRunner, ctx: IngestionContext) -> tuple[IngestionContext, list[StepResult]]:
    return ctx, await runner.run(ctx)


async def _main() -> int:
    logfire.configure(console=False, send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()

    args = _parse_args()
    output = parse_output(args.output)
    resolvers = ((input_path, create_pipeline_from_config(args.pipeline, config_args=args.params)) for input_path in args.input)
    runners = [
        (ctx, create_pipeline_from_config(args.pipeline, config_args=args.params))
        for input_path, resolver in resolvers
        async for ctx in resolver.flatten(IngestionContext(resource=input_path))
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        redirect_stdout=True,
        redirect_stderr=True,
        disable=args.quiet
    ) as p:
        listener = None
        if not args.quiet:
            labels = {
                id(step): f"{Path(ctx.resource.uri).name} {i:02d} {runner.step_descriptions.get(id(step)) or lookup_step_id(step) or type(step).__name__}"
                for ctx, runner in runners
                for i, step in enumerate(runner.steps, 1)
            }
            tasks = {step_id: (p.add_task(f"{label} [dim]pending[/dim]", total=1, start=False), label) for step_id, label in labels.items()}
            listener = asyncio.create_task(_events(p, tasks))
            await asyncio.sleep(0)
        try:
            runs = await asyncio.gather(*(_run(runner, ctx) for ctx, runner in runners))
        finally:
            event.shutdown()
            if listener:
                await listener

    await output.write_many([ctx for ctx, _ in runs])
    return int(any(r.status == StepStatus.FAILURE for _, results in runs for r in results))


def main() -> None:
    raise SystemExit(asyncio.run(_main()))
