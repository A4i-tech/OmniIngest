"""FastAPI backend for the OmniIngest pipeline UI."""

import asyncio
import inspect
import re
import tempfile
import time
import typing
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import fsspec
import yaml
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse

OMNI_INGEST_ROOT = Path(r"C:\a4i\OmniIngest")
load_dotenv(OMNI_INGEST_ROOT / ".env")  # must run before importing omni_ingest

_IMPORT_ERROR: str | None = None
try:
    from omni_ingest.core import event
    from omni_ingest.core.model import JqExpression, PipelineConfig
    from omni_ingest.core.output import Output, parse as parse_output
    from omni_ingest.core.pipeline import _STEP_FACTORY, IngestionContext, PipelineRunner, create_pipeline_from_config, default_metadata_store, lookup_step_id
except Exception as e:  # keep app importable when omni_ingest / its deps are missing
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"

PROFILES_DIR = OMNI_INGEST_ROOT / "profiles"
TESTS_PROFILES_DIR = OMNI_INGEST_ROOT / "tests" / "integration" / "profiles"
LOCAL_PIPELINES_DIR = Path(__file__).resolve().parent / "pipelines"
PIPELINE_DIRS: list[tuple[Path, str]] = [
    (PROFILES_DIR, "profiles"),
    (TESTS_PROFILES_DIR, "tests"),
    (LOCAL_PIPELINES_DIR, "local"),
]

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SYNTHETIC_STAGE_RE = re.compile(r"^stage-\d+-[0-9a-f]{8}$")  # create_pipeline_from_config's auto-generated stage keys, not real stages

_UNSUPPORTED_ERROR = "OmniIngest package is not installed. Run: uv sync --project server (it installs omni-ingest from ../OmniIngest in editable mode)."


def _display_stage(name: str) -> str | None:
    return None if _SYNTHETIC_STAGE_RE.match(name) else name


def _validate_pipeline_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise HTTPException(400, detail=f"'{name}' is not a usable pipeline name. Use only letters, numbers, hyphens, and underscores, up to 64 characters, then try again.")


# --- API models ---

class AgentField(BaseModel):
    name: str
    type: str
    required: bool
    default: Any = None
    description: str | None = None
    choices: list[Any] | None = None


class AgentSpec(BaseModel):
    id: str
    cls: str
    kind: Literal["primitive", "component"]
    group: str
    description: str | None = None
    fields: list[AgentField] = Field(default_factory=list)
    steps: list[str] | None = None
    path: str | None = None
    replaces_items: bool = False


class SavedPipeline(BaseModel):
    name: str
    path: str
    source: Literal["profiles", "tests", "local"]
    pipeline_id: str | None = None
    description: str | None = None
    agents: list[str] = Field(default_factory=list)
    modified: int


class PipelineYamlBody(BaseModel):
    yaml: str


class RunRequest(BaseModel):
    yaml: str
    input: str | list[str]
    output: str = "-"
    format: str | None = None
    tenant: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunStepState(BaseModel):
    agent: str
    stage: str | None
    status: str = "pending"
    start: int | None = None
    end: int | None = None
    completed: float | None = None
    total: float | None = None


class RunEvent(BaseModel):
    t: int
    kind: str
    step: str | None = None
    step_index: int | None = None
    stage: str | None = None
    completed: float | None = None
    total: float | None = None
    message: str | None = None
    status: str | None = None
    error: str | None = None


class Run(BaseModel):
    run_id: str
    pipeline_id: str | None = None
    source: str
    output: str
    tenant: str
    status: str = "running"
    started_at: int
    ended_at: int | None = None
    error: str | None = None
    output_result: str | None = None
    resumable: bool = False
    steps: list[RunStepState] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)


# --- run bookkeeping (in-memory, no database) ---

class RunState:
    def __init__(self, run: Run, started_dt: datetime, step_index: dict[int, int], stage_by_id: dict[int, str]):
        self.run = run
        self.started_dt = started_dt
        self.step_index = step_index
        self.stage_by_id = stage_by_id
        self.task: asyncio.Task | None = None
        self.subscribers: set[asyncio.Queue] = set()


_runs: dict[str, RunState] = {}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get_run_or_404(run_id: str) -> RunState:
    state = _runs.get(run_id)
    if state is None:
        raise HTTPException(404, detail=f"No run with id '{run_id}' exists. Check the run ID and try again.")
    return state


async def _with_resumable(run: Run) -> Run:
    if run.status == "failed":
        run.resumable = bool(await default_metadata_store().get_pipeline_checkpoint(UUID(run.run_id)))
    return run


MAX_RUNS = 50
MAX_EVENTS_PER_RUN = 2000


def _trim_events(run: Run) -> None:
    """Keep the head (pipeline_begin and the first steps) and the most recent tail."""
    if len(run.events) <= MAX_EVENTS_PER_RUN:
        return
    keep_head = 200
    run.events = run.events[:keep_head] + run.events[-(MAX_EVENTS_PER_RUN - keep_head):]


def _evict_old_runs() -> None:
    finished = [
        (rid, st) for rid, st in _runs.items()
        if st.run.status not in ("running", "queued") and not st.subscribers
    ]
    excess = len(_runs) - MAX_RUNS
    if excess <= 0:
        return
    finished.sort(key=lambda kv: kv[1].run.started_at)
    for rid, _ in finished[:excess]:
        _runs.pop(rid, None)


def _finish_run(state: RunState, status: str, error: str | None) -> None:
    if state.run.status != "running":
        return
    state.run.status = status
    state.run.error = error
    state.run.ended_at = _now_ms()
    t = int((datetime.now(UTC) - state.started_dt).total_seconds() * 1000)
    ev = RunEvent(t=t, kind="pipeline_end", status=status, error=error)
    state.run.events.append(ev)
    for q in list(state.subscribers):
        q.put_nowait(ev)


def _to_run_event(state: RunState, e: "event.Event") -> RunEvent:
    t = int((e.timestamp - state.started_dt).total_seconds() * 1000)
    if isinstance(e, event.PipelineBeginEvent):
        return RunEvent(t=t, kind="pipeline_begin")
    if isinstance(e, event.PipelineEndEvent):
        return RunEvent(t=t, kind="pipeline_end", status=e.status, error=e.error)

    step_name = lookup_step_id(e.step) or "unknown"
    step_index = state.step_index.get(id(e.step))
    stage = state.stage_by_id.get(id(e.step))
    if isinstance(e, event.PipelineStepBeginEvent):
        return RunEvent(t=t, kind="step_begin", step=step_name, step_index=step_index, stage=stage)
    if isinstance(e, event.PipelineStepProgressEvent):
        return RunEvent(t=t, kind="step_progress", step=step_name, step_index=step_index, stage=stage, completed=e.completed, total=e.total, message=e.message)
    if isinstance(e, event.PipelineStepEndEvent):
        return RunEvent(t=t, kind="step_end", step=step_name, step_index=step_index, stage=stage, status=e.step_result.status.value.lower(), error=e.step_result.error)
    return RunEvent(t=t, kind="unknown")


def _update_step(state: RunState, e: "event.Event", run_event: RunEvent) -> None:
    if not isinstance(e, event._StepMixin):
        return
    idx = state.step_index.get(id(e.step))
    if idx is None:
        return
    step = state.run.steps[idx]
    if isinstance(e, event.PipelineStepBeginEvent):
        step.status, step.start = "running", run_event.t
    elif isinstance(e, event.PipelineStepProgressEvent):
        step.completed, step.total = e.completed, e.total
    elif isinstance(e, event.PipelineStepEndEvent):
        step.status, step.end = run_event.status, run_event.t


async def _consume_events() -> None:
    async for e in event.subscribe(populate_with_buffered=False):
        state = _runs.get(str(e.run_id))
        if state is None:
            continue
        if isinstance(e, event.PipelineEndEvent) and state.run.status != "running":
            continue  # a cancel already finished this run; a second pipeline_end would double-close the stream
        run_event = _to_run_event(state, e)
        state.run.events.append(run_event)
        _trim_events(state.run)
        _update_step(state, e, run_event)
        if isinstance(e, event.PipelineEndEvent):
            state.run.status = "succeeded" if e.status == "succeeded" else "failed"
            state.run.error = e.error
            state.run.ended_at = _now_ms()
        for q in list(state.subscribers):
            q.put_nowait(run_event)


async def _run_pipeline(state: RunState, runner: "PipelineRunner", ctx: "IngestionContext", output: "Output | None") -> None:
    try:
        await runner.run(ctx, output=output)
        state.run.output_result = str(runner.output_result) if runner.output_result is not None else None
    except asyncio.CancelledError:
        _finish_run(state, "cancelled", None)
        raise
    except Exception as e:
        _finish_run(state, "failed", _explain_run_failure(state, e))


def _explain_run_failure(state: RunState, e: Exception) -> str:
    detail = f"{type(e).__name__}: {e}"
    if isinstance(e, (OSError, FileNotFoundError)) or "Could not resolve resource" in str(e):
        return f"Cannot read the input {state.run.source!r}. Check the path exists and is readable, then run again. ({detail})"
    if isinstance(e, ValueError):
        return f"The pipeline YAML was rejected. Fix the reported field, then run again. ({detail})"
    return f"The run stopped at an unexpected error. Check the server log for the full traceback. ({detail})"


# --- agent introspection ---

_JQ_VALIDATOR = typing.get_args(JqExpression)[1].func if _IMPORT_ERROR is None else None


def _is_jq_field(field: Any) -> bool:
    if any(getattr(m, "func", None) is _JQ_VALIDATOR for m in field.metadata):
        return True

    def scan(ann: Any) -> bool:
        if any(getattr(m, "func", None) is _JQ_VALIDATOR for m in getattr(ann, "__metadata__", ())):
            return True
        return any(scan(a) for a in typing.get_args(ann))

    return scan(field.annotation)


_ITEMS_WRITE_RE = re.compile(r"ctx\.items\s*(=[^=]|\.extend|\.append|\.clear)")


def _replaces_items(cls: type) -> bool:
    """Heuristic: two steps that both rebuild ctx.items must not share a stage."""
    try:
        return bool(_ITEMS_WRITE_RE.search(inspect.getsource(cls)))
    except (OSError, TypeError):
        return False


def _resolve_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            if option.get("type") != "null":
                return option
        return schema["anyOf"][0]
    return schema


_JSON_TYPE_MAP = {"integer": "integer", "number": "number", "boolean": "boolean", "array": "array", "object": "object"}


def _agent_field(name: str, raw_schema: dict[str, Any], required: bool, is_jq: bool = False) -> AgentField:
    resolved = _resolve_schema(raw_schema)
    if "enum" in resolved:
        ftype, choices = "enum", resolved["enum"]
    elif is_jq and resolved.get("type") in ("object", "array"):
        # `schema` is dict | JqExpression; almost every profile writes the literal JSON, so edit it as JSON.
        ftype, choices = _JSON_TYPE_MAP[resolved["type"]], None
    elif is_jq:
        ftype, choices = "jq", None
    elif name in ("prompt", "image_conversion_prompt"):
        ftype, choices = "text", None
    else:
        ftype, choices = _JSON_TYPE_MAP.get(resolved.get("type"), "string"), None
    return AgentField(name=name, type=ftype, required=required, default=raw_schema.get("default"), description=raw_schema.get("description"), choices=choices)


def _first_doc_line(doc: str | None) -> str | None:
    for line in (doc or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return None


def _primitive_agents() -> list[AgentSpec]:
    specs = []
    for reg_id, cls in _STEP_FACTORY.items():
        schema = cls.model_json_schema(by_alias=True)
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        field_by_alias = {(f.alias or fname): f for fname, f in cls.model_fields.items()}

        fields = []
        for name, raw_schema in properties.items():
            if name == "model_config":
                continue
            field = field_by_alias.get(name)
            is_jq = _is_jq_field(field) if field is not None else False
            fields.append(_agent_field(name, raw_schema, name in required, is_jq))

        group = cls.__module__.rsplit(".", 1)[-1]
        specs.append(AgentSpec(id=reg_id, cls=cls.__name__, kind="primitive", group=group, description=_first_doc_line(cls.__doc__), fields=fields, replaces_items=_replaces_items(cls)))
    return specs


def _component_agents() -> list[AgentSpec]:
    specs = []
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        parameters = data.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        fields = [_agent_field(name, raw_schema, name in required) for name, raw_schema in properties.items() if name != "model_config"]
        steps = [step.get("agent") for step in (data.get("steps") or [])]
        rel_path = f"profiles/{path.name}"
        specs.append(AgentSpec(id=path.stem, cls=rel_path, kind="component", group="components", description=data.get("description"), fields=fields, steps=steps, path=rel_path,
                               replaces_items=any(_STEP_FACTORY.get(a) is not None and _replaces_items(_STEP_FACTORY[a]) for a in steps if a)))
    return specs


# --- saved pipelines ---

def _list_pipelines() -> list[SavedPipeline]:
    out = []
    for directory, source in PIPELINE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            try:
                config = PipelineConfig.model_validate(yaml.safe_load(path.read_text()))
            except Exception:
                continue
            out.append(SavedPipeline(
                name=path.stem, path=str(path.resolve()), source=source,
                pipeline_id=config.pipeline_id, description=config.description,
                agents=[step.agent for step in config.steps],
                modified=int(path.stat().st_mtime * 1000),
            ))
    return out


# --- app setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOCAL_PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
    task = asyncio.create_task(_consume_events()) if _IMPORT_ERROR is None else None
    yield
    running = [st.task for st in _runs.values() if st.task is not None and not st.task.done()]
    for t in running:
        t.cancel()
    for t in running:
        with suppress(asyncio.CancelledError, Exception):
            await t
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


def _require_omni_ingest() -> None:
    if _IMPORT_ERROR is not None:
        raise HTTPException(status_code=503, detail=_UNSUPPORTED_ERROR)


router = APIRouter(prefix="/api", dependencies=[Depends(_require_omni_ingest)])


@router.get("/agents", response_model=list[AgentSpec])
async def list_agents() -> list[AgentSpec]:
    return _primitive_agents() + _component_agents()


@router.get("/pipelines", response_model=list[SavedPipeline])
async def list_pipelines() -> list[SavedPipeline]:
    return _list_pipelines()


@router.get("/pipelines/{name}")
async def get_pipeline(name: str) -> dict[str, Any]:
    _validate_pipeline_name(name)
    # Reversed: PUT always writes to the local dir, so a local copy must shadow the repo one.
    for directory, _source in reversed(PIPELINE_DIRS):
        candidate = directory / f"{name}.yaml"
        if candidate.is_file():
            text = candidate.read_text()
            try:
                parameters = PipelineConfig.model_validate(yaml.safe_load(text)).parameters
            except Exception:
                parameters = {}
            return {"yaml": text, "parameters": parameters}
    raise HTTPException(404, detail=f"No pipeline named '{name}' exists. Check the name and try again.")


@router.put("/pipelines/{name}", response_model=SavedPipeline)
async def put_pipeline(name: str, body: PipelineYamlBody) -> SavedPipeline:
    _validate_pipeline_name(name)
    try:
        parsed = yaml.safe_load(body.yaml)
    except yaml.YAMLError as e:
        raise HTTPException(422, detail=f"This pipeline is not valid YAML: {e}. Fix the YAML syntax and save again.")
    try:
        config = PipelineConfig.model_validate(parsed)
    except ValidationError as e:
        raise HTTPException(422, detail=f"This pipeline configuration is invalid: {e}. Fix the field(s) named above and save again.")

    path = LOCAL_PIPELINES_DIR / f"{name}.yaml"
    path.write_text(body.yaml)
    return SavedPipeline(
        name=name, path=str(path.resolve()), source="local",
        pipeline_id=config.pipeline_id, description=config.description,
        agents=[step.agent for step in config.steps],
        modified=int(path.stat().st_mtime * 1000),
    )


@router.delete("/pipelines/{name}")
async def delete_pipeline(name: str) -> dict[str, bool]:
    _validate_pipeline_name(name)
    local_path = LOCAL_PIPELINES_DIR / f"{name}.yaml"
    if local_path.is_file():
        local_path.unlink()
        return {"ok": True}
    for directory, _source in PIPELINE_DIRS[:-1]:
        if (directory / f"{name}.yaml").is_file():
            raise HTTPException(400, detail=f"'{name}' is a pipeline from the OmniIngest repository and cannot be deleted here. Save it as a local copy first, then delete the copy.")
    raise HTTPException(404, detail=f"No local pipeline named '{name}' exists. Check the name and try again.")


def _build_pipeline(yaml_text: str, params: dict[str, Any]) -> tuple[Path, "PipelineRunner"]:
    tmp = Path(tempfile.gettempdir()) / f"omniingest-run-{uuid4().hex}.yaml"
    tmp.write_text(yaml_text)
    try:
        return tmp, create_pipeline_from_config(tmp, config_args=params)
    except ValueError as e:
        tmp.unlink(missing_ok=True)
        if str(e).startswith("Invalid parameters for pipeline"):
            cause = e.__cause__
            field = ".".join(str(p) for p in cause.path) if cause is not None and getattr(cause, "path", None) else None
            hint = f" Correct the '{field}' parameter and try again." if field else " Correct the parameter named above and try again."
            raise HTTPException(422, detail=f"{e}{hint}")
        raise HTTPException(422, detail=f"This pipeline could not be built: {e}. Check the agent names and parameters in the YAML.")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, detail=f"This pipeline could not be built: {type(e).__name__}: {e}. Check the agent names and parameters in the YAML.")


def _new_run_state(runner: "PipelineRunner", ctx: "IngestionContext", pipeline_id: str | None, output_label: str, tenant: str) -> RunState:
    run_id = str(ctx.run_id)
    started_dt = datetime.now(UTC)
    stage_by_id = {id(step): _display_stage(stage) for stage, steps in runner.stages.items() for step in steps}
    step_index = {id(step): i for i, step in enumerate(runner.steps)}
    steps = [
        RunStepState(agent=runner.step_descriptions.get(id(step)) or lookup_step_id(step) or type(step).__name__, stage=stage_by_id[id(step)])
        for step in runner.steps
    ]
    run = Run(
        run_id=run_id, pipeline_id=pipeline_id, source=ctx.resource.uri, output=output_label,
        tenant=tenant, status="running", started_at=int(started_dt.timestamp() * 1000), steps=steps,
    )
    state = RunState(run=run, started_dt=started_dt, step_index=step_index, stage_by_id=stage_by_id)
    _runs[run_id] = state
    _evict_old_runs()
    return state


def _start_run(runner: "PipelineRunner", ctx: "IngestionContext", output: "Output | None", pipeline_id: str | None, output_label: str, tenant: str) -> Run:
    state = _new_run_state(runner, ctx, pipeline_id, output_label, tenant)
    state.task = asyncio.create_task(_run_pipeline(state, runner, ctx, output))
    return state.run


@router.post("/runs", response_model=list[Run])
async def create_run(body: RunRequest) -> list[Run]:
    try:
        config = PipelineConfig.model_validate(yaml.safe_load(body.yaml))
    except yaml.YAMLError as e:
        raise HTTPException(422, detail=f"This pipeline is not valid YAML: {e}. Fix the YAML syntax and try again.")
    except ValidationError as e:
        raise HTTPException(422, detail=f"This pipeline configuration is invalid: {e}. Fix the field(s) named above and try again.")

    try:
        output = None if body.output == "-" else parse_output(body.output, body.format)
    except Exception as e:
        raise HTTPException(422, detail=f"The output target '{body.output}' is invalid: {type(e).__name__}: {e}. Fix the output destination and try again.")

    tmp, resolver = _build_pipeline(body.yaml, body.params)
    tenant = body.tenant or "default"
    inputs = body.input if isinstance(body.input, list) else [body.input]
    try:
        runs = [
            _start_run(create_pipeline_from_config(tmp, config_args=body.params), ctx, output, config.pipeline_id, body.output, tenant)
            for input_value in inputs
            async for ctx in resolver.flatten(IngestionContext(resource=input_value))
        ]
    finally:
        tmp.unlink(missing_ok=True)
    return runs


@router.get("/runs", response_model=list[Run])
async def list_runs() -> list[Run]:
    runs = sorted((s.run for s in _runs.values()), key=lambda r: r.started_at, reverse=True)
    return [await _with_resumable(r) for r in runs]


@router.get("/runs/{run_id}", response_model=Run)
async def get_run(run_id: str) -> Run:
    return await _with_resumable(_get_run_or_404(run_id).run)


@router.post("/runs/{run_id}/resume", response_model=Run)
async def resume_run(run_id: str, body: PipelineYamlBody) -> Run:
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(400, detail=f"'{run_id}' is not a valid run ID. Check the ID and try again.")
    try:
        config = PipelineConfig.model_validate(yaml.safe_load(body.yaml))
    except yaml.YAMLError as e:
        raise HTTPException(422, detail=f"This pipeline is not valid YAML: {e}. Fix the YAML syntax and try again.")
    except ValidationError as e:
        raise HTTPException(422, detail=f"This pipeline configuration is invalid: {e}. Fix the field(s) named above and try again.")

    tmp, runner = _build_pipeline(body.yaml, {})
    tmp.unlink(missing_ok=True)
    try:
        ctx, awaitable = await runner.resume(run_uuid, default_metadata_store())
    except ValueError as e:
        if "checkpoint not found" in str(e):
            raise HTTPException(404, detail=f"Run '{run_id}' has no checkpoint and cannot be resumed. Start a new run instead. ({e})")
        raise HTTPException(422, detail=f"This YAML does not match the run being resumed: {e}. Use the pipeline YAML that originally created this run, then try again.")

    state = _new_run_state(runner, ctx, config.pipeline_id, "-", "default")

    async def _await_resume() -> None:
        try:
            await awaitable
            state.run.output_result = str(runner.output_result) if runner.output_result is not None else None
        except asyncio.CancelledError:
            _finish_run(state, "cancelled", None)
            raise
        except Exception as e:
            _finish_run(state, "failed", _explain_run_failure(state, e))

    state.task = asyncio.create_task(_await_resume())
    return state.run


@router.get("/resources")
async def list_resources(path: str) -> dict[str, Any]:
    try:
        fs, resolved_path = fsspec.core.url_to_fs(path, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        paths = fs.expand_path(resolved_path)
    except Exception as e:
        raise HTTPException(400, detail=f"Could not resolve '{path}': {type(e).__name__}: {e}. Check the path or glob and try again.")
    return {"resources": paths[:200], "count": len(paths)}


@router.get("/outputs")
async def list_outputs() -> list[dict[str, Any]]:
    return [
        {"value": "-", "label": "Discard", "kind": "none"},
        {"value": "out.json", "label": "JSON file", "kind": "file", "format": "json"},
        {"value": "out.pkl", "label": "Pickle file", "kind": "file", "format": "pkl"},
        {"value": "vector:default", "label": "Vector store", "kind": "store"},
        {"value": "graph:default", "label": "Graph store", "kind": "store"},
        {"value": "mongodb://", "label": "MongoDB", "kind": "uri"},
    ]


@router.get("/tool-vendor-schema")
async def tool_vendor_schema() -> dict[str, Any]:
    try:
        from fastmcp.mcp_config import MCPServerTypes
        from pydantic import TypeAdapter
        return TypeAdapter(MCPServerTypes).json_schema()
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


@router.post("/runs/{run_id}/cancel", response_model=Run)
async def cancel_run(run_id: str) -> Run:
    state = _get_run_or_404(run_id)
    if state.task is not None and not state.task.done():
        state.task.cancel()
    _finish_run(state, "cancelled", None)
    return state.run


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str) -> EventSourceResponse:
    state = _get_run_or_404(run_id)

    async def gen():
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        state.subscribers.add(queue)
        replay = list(state.run.events)  # no await between subscribing and snapshotting: atomic under asyncio's cooperative scheduling
        try:
            for ev in replay:
                yield {"data": ev.model_dump_json()}
                if ev.kind == "pipeline_end":
                    return
            while True:
                ev = await queue.get()
                yield {"data": ev.model_dump_json()}
                if ev.kind == "pipeline_end":
                    return
        finally:
            state.subscribers.discard(queue)

    return EventSourceResponse(gen())


app.include_router(router)
