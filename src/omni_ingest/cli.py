"""Command-line entrypoint for running Omni Ingest pipeline YAML files."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

import logfire
import yaml
from dotenv import load_dotenv
from pydantic import TypeAdapter
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn

load_dotenv(Path.cwd() / ".env")  # must run before importing modules that instantiate settings

from omni_ingest.core import event  # noqa: E402
from omni_ingest.core.model import PipelineConfig, ResolvedResource, Resource, StepStatus  # noqa: E402
from omni_ingest.core.output import parse as parse_output  # noqa: E402
from omni_ingest.core.pipeline import IngestionContext, create_pipeline_from_config, default_metadata_store, lookup_step_id  # noqa: E402


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
    parser.add_argument("pipeline", type=Path, help="Pipeline YAML file")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", type=TypeAdapter(Resource).validate_python, help="Resource URI, local path, or '-' for stdin")
    source.add_argument("--resume", nargs="+", type=UUID, help="Pipeline run ID to resume")
    parser.add_argument("--input-name", help="Filename hint for stdin input")
    parser.add_argument("--output", required=True, help="Output URI, '-' for stdout, or configured store target")
    parser.add_argument("--format", choices=["json"], help="Output format, overrides filename extension")
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide progress display")
    for name, schema in properties.items():
        kwargs = {"default": schema.get("default", argparse.SUPPRESS), "help": schema.get("description"), "required": name in parameters.get("required", [])}
        if "enum" in schema:
            kwargs["choices"] = schema["enum"]
        if schema.get("type") == "boolean":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs.update(metavar=schema.get("type", "VALUE").upper(), type={"integer": int, "number": float, "string": str}.get(schema.get("type"), json.loads))
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    args = parser.parse_args()
    stdin = args.input and any(str(resource) == "-" for resource in args.input)
    if stdin and len(args.input) != 1:
        parser.error("stdin input '-' cannot be combined with other inputs")
    if args.input_name and not stdin:
        parser.error("--input-name requires --input -")
    args.params = {name: getattr(args, name) for name in properties if hasattr(args, name)}
    return args


def _read_stdin(name: str | None) -> ResolvedResource:
    return ResolvedResource(uri=name or "stdin://", raw_content=sys.stdin.buffer.read())


async def _events(progress: Progress, tasks: dict[int, tuple[TaskID, str]]) -> None:
    async for e in event.subscribe(populate_with_buffered=False):
        if isinstance(e, event.PipelineBeginEvent):
            for step in e.completed_steps:
                task_id, label = tasks[id(step)]
                progress.update(task_id, completed=1, description=f"{label} [green]success[/green]")
                progress.stop_task(task_id)

        if not isinstance(e, event._StepMixin):
            continue

        task_id, label = tasks[id(e.step)]
        if isinstance(e, event.PipelineStepBeginEvent):
            progress.start_task(task_id)
            progress.update(task_id, completed=0, description=f"{label} [cyan]running[/cyan]")
        elif isinstance(e, event.PipelineStepProgressEvent):
            progress.update(task_id, total=e.total or 1, completed=e.completed, description=f"{label} [cyan]running[/cyan]" + (f" [dim]{e.message}[/dim]" if e.message else ""))
        elif isinstance(e, event.PipelineStepEndEvent):
            if e.step_result.status == StepStatus.SUCCESS:
                progress.update(task_id, total=1, completed=1, description=f"{label} [green]success[/green]")
            else:
                progress.update(task_id, description=f"{label} [red]{e.step_result.status.value}[/red]")
            progress.stop_task(task_id)


async def _main() -> int:
    logfire.configure(console=False, send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()

    args = _parse_args()
    output = parse_output(args.output, args.format)
    if args.resume:
        store = default_metadata_store()
        jobs = [
            (str(run_id), runner := create_pipeline_from_config(args.pipeline, config_args=args.params), *(await runner.resume(run_id, store)))
            for run_id in args.resume
        ]
    else:
        inputs = [_read_stdin(args.input_name)] if str(args.input[0]) == "-" else args.input
        resolvers = ((input_path, create_pipeline_from_config(args.pipeline, config_args=args.params)) for input_path in inputs)
        jobs = [
            (Path(ctx.resource.uri).name, runner := create_pipeline_from_config(args.pipeline, config_args=args.params), ctx, runner.run(ctx))
            for input_path, resolver in resolvers
            async for ctx in resolver.flatten(IngestionContext(resource=input_path))
        ]

    console = Console(stderr=True)
    console.print("\n".join(f"Pipeline run [bold]{ctx.run_id}[/bold] ({name})" for name, _, ctx, _ in jobs))
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), TextColumn("ETA"), TimeRemainingColumn(), console=console, redirect_stdout=True, redirect_stderr=True, disable=args.quiet) as p:
        labels = {id(step): f"{name} {i:02d} {runner.step_descriptions.get(id(step)) or lookup_step_id(step) or type(step).__name__}" for name, runner, _, _ in jobs for i, step in enumerate(runner.steps, 1)}
        tasks = {step_id: (p.add_task(f"{label} [dim]pending[/dim]", total=1, start=False), label) for step_id, label in labels.items()}
        listener = None if args.quiet else asyncio.create_task(_events(p, tasks))
        if listener:
            await asyncio.sleep(0)
        try:
            runs = await asyncio.gather(*(job[3] for job in jobs))
        finally:
            event.shutdown()
            if listener:
                await listener

    await output.write_many([job[2] for job in jobs])
    failures = [r for run in runs for r in run if r.status == StepStatus.FAILURE]
    if failures:
        console.print("\n".join(f"[red]{r.error}[/red]" for r in failures if r.error))
    return int(any(failures))


def main() -> None:
    raise SystemExit(asyncio.run(_main()))
