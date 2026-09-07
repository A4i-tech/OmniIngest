# omniingest-ui server

FastAPI backend for the pipeline UI. Wraps the `omni_ingest` pipeline engine: lists
available agents and saved pipelines, runs pipelines, and streams run progress over SSE.

## Install

```
uv sync --project server
```

`omni-ingest` is installed from the local checkout at `../OmniIngest` (relative to this
directory) in editable mode, not from PyPI — the published PyPI release is old. Keep the
two repos as sibling directories (`OmniIngest/` and `omniingest-ui/`) for the relative
path in `pyproject.toml` to resolve.

## Run

```
uv run --project server uvicorn app:app --port 8077 --reload --app-dir server
```

The server reads `C:\a4i\OmniIngest\.env` for model credentials (OpenAI keys, Azure
translation/speech keys, etc.) before importing `omni_ingest`, the same way
`omni_ingest`'s own CLI does.

If `omni_ingest` (or one of its dependencies) fails to import, the server still starts
and every `/api/*` route responds with HTTP 503 instead of crashing.

## API

- `GET /api/agents` — primitive step agents (from `omni_ingest`'s step registry) and
  component agents (from `OmniIngest/profiles/*.yaml`).
- `GET /api/pipelines`, `GET /api/pipelines/{name}`, `PUT /api/pipelines/{name}`,
  `DELETE /api/pipelines/{name}` — saved pipeline YAML, sourced from
  `OmniIngest/profiles`, `OmniIngest/tests/integration/profiles` (read-only), and
  `server/pipelines` (local, editable; created automatically on startup).
- `POST /api/runs`, `GET /api/runs`, `GET /api/runs/{id}`, `POST /api/runs/{id}/cancel`,
  `GET /api/runs/{id}/events` — start, list, inspect, cancel, and stream (SSE) pipeline
  runs. Runs are kept in memory only; nothing is persisted to a database.

CORS is enabled for `http://localhost:5173` and `http://127.0.0.1:5173`.
