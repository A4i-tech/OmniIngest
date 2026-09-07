# OmniIngest Pipeline Studio

An n8n-style UI for OmniIngest: build pipeline YAML on a node canvas, save it, run it, and
watch the steps execute live.

The agent palette is **generated from the installed `omni_ingest` package**, not hard-coded.
Every entry in `_STEP_FACTORY` becomes a node, and its config form is built from that Step's
Pydantic field schema. Register a new Step in Python and it appears in the palette with no
change here.

## Two kinds of node

| kind | source | in the UI |
|---|---|---|
| **primitive** | the 15 classes in `omni_ingest.core.pipeline._STEP_FACTORY` | cream node, config form generated from the Pydantic model |
| **component** | `OmniIngest/profiles/*.yaml` | amber node, a collapsed subgraph; config form built from the YAML's `parameters` schema |

`validation:` renders as a badge on the node.

## Parallel steps

OmniIngest runs a stage's steps concurrently — `PipelineRunner._run` collects each stage and
`asyncio.gather`s it. The UI exposes that directly:

- Select a step and press **RUN WITH PREVIOUS STEP**. It puts that step and the one above it
  on the same `stage:`, inventing a name (`fanout`, `fanout-2`, …) when neither has one.
  **SPLIT FROM PREVIOUS STEP** undoes it. The `STAGE` field stays editable for a real name.
- The canvas draws the fan-out: the trunk splits on a horizontal bus into one branch per
  step, and rejoins on a second bus into whatever follows. A stage that ends the pipeline
  draws no join — the branches simply terminate.
- The run timeline brackets a stage's bars with a dashed rule.

**It is a diamond, not a tree.** A stage fans out and then joins: every step in it shares one
`IngestionContext`, and the runner waits for all of them before the next stage. There is no
branch that runs to its own separate end with its own output — `PipelineConfig.steps` is a
flat list and a run writes exactly one output. Branches of unequal length are expressed as
consecutive stages, so a later stage still waits for the whole previous one.

**Any agent can go in a stage — the limit is state, not agent type.** Five agents rebuild
`ctx.items` wholesale (`chunking`, `page_chunking`, `page_stitching`, `timed_chunking`,
`image_extraction`). Two of those in one stage overwrite each other. The backend flags them
with `replaces_items` (detected by inspecting each Step class's source, so a newly registered
agent is classified without editing a list) and `validatePipeline` blocks the run. Everything
else writes to its own jq path and parallelises safely.

- A stage must be **contiguous**. `validatePipeline` blocks a run whose stage is split across
  non-adjacent steps, because `create_pipeline_from_config` raises on it.

Timing caveat: the runner emits every `step_begin` in a stage before the gather and every
`step_end` after it, so bars inside one stage show near-identical spans. That is the event
model, not the UI — per-step durations inside a stage are not separately reported.

## What the UI now covers

Every field of `PipelineConfig` and every CLI flag, so the UI is 1:1 with the runtime.

| OmniIngest capability | Where in the UI |
|---|---|
| `steps[].agent` / `config` | node canvas + generated config form |
| `steps[].stage` | RUN WITH PREVIOUS STEP, dashed concurrent group |
| `steps[].description` | inspector |
| `steps[].validation` (condition/before/plain/after + jq + message) | inspector, VALIDATION section |
| `pipeline_id`, `domain_profile`, `input_modality`, `description` | SETTINGS drawer, IDENTITY |
| `parameters` JSON schema + `${.config.x}` tokens | SETTINGS drawer, PARAMETERS |
| `tool_vendors` | SETTINGS drawer, read-only note (not editable yet) |
| `--input` with several values, globs, directories, URLs, stdin | RUN dialog, INPUTS |
| `--input-name` | RUN dialog, shown only when an input is `-` |
| `--output` (`-`, `.json`, `.pkl`, `vector:default`, `graph:default`, `mongodb://`) | RUN dialog, OUTPUT |
| `--format json\|pkl` | RUN dialog, FORMAT |
| per-run parameter overrides | RUN dialog, PARAMETERS |
| `--resume <run_id>` | `POST /api/runs/{id}/resume`; `Run.resumable` marks eligible runs |

A glob or directory expands into **one run per resolved resource**, matching `PipelineRunner.flatten`.

## Install

`omni-ingest` installs from the **local checkout at `../OmniIngest`**, in editable mode — the
PyPI release is behind. `server/pyproject.toml` pins it with `[tool.uv.sources]`.

```bash
npm install
uv sync --project server
```

## Run

Two processes.

```bash
uv run --project server uvicorn app:app --port 8077 --reload --app-dir server
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to port 8077.

The server reads `../OmniIngest/.env` for model and store credentials. Without it, `text`,
`chunking` and other non-model steps still run; `extract`, `prompt` and `embedding` fail at
the first model call.

## Test

```bash
npm test
```

Covers the YAML codec: round-trip against a real profile, stage grouping, default pruning,
and the validation rules that stop an unrunnable pipeline reaching the server.

## Where things are

```
design/          the design canvas artboards + SPEC.md (the visual contract)
server/app.py    FastAPI: registry introspection, profile CRUD, run control, SSE
src/yaml.ts      the YAML codec — graph model in, PipelineConfig out
src/api.ts       every server call, in one file
src/views/       Builder, Runs, Library
src/components/  Palette, GraphCanvas, NodeCard, Inspector, YamlPane, RunList, Timeline, EventStream
```

## Deliberate limits

- **Steps are an ordered list, not a free graph.** `PipelineConfig.steps` is a list with
  contiguous `stage` grouping, so the canvas reorders vertically instead of drawing arbitrary
  edges. A DAG editor would let you draw pipelines the YAML cannot express.
- **Runs live in server memory.** Restarting the server loses run history. The pipelines
  themselves are files on disk and survive.
- **Saving always writes to `server/pipelines/`.** Nothing in the OmniIngest checkout is
  overwritten from the UI.
