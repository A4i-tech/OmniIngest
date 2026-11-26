# Command-line usage

`omni-ingest` runs one pipeline YAML against one or more resources. Source and destination can both be local or remote. Pipeline parameters become command-line options so one YAML can cover several jobs without edits.

A normal run has three parts:

```bash
omni-ingest pipeline.yaml --input source.pdf --output result.json
```

- `pipeline.yaml` describes processing.
- `--input` selects resources.
- `--output` selects final destination.

## Install as a uv tool

Install OmniIngest as a standalone command:

```bash
uv tool install omni-ingest
```

This gives you `omni-ingest` without adding it to current project's dependencies. Upgrade installed tool when a newer release is available:

```bash
uv tool upgrade omni-ingest
```

Use `uvx` for a one-off run without keeping tool installed:

```bash
uvx omni-ingest pipeline.yaml --input source.pdf --output result.json
```

When OmniIngest is already a project dependency use `uv run`:

```bash
uv run omni-ingest pipeline.yaml --input source.pdf --output result.json
```

## First run

Suppose `school_notes.yaml` extracts text then chunks it. Start with one local PDF and JSON output.

```bash
omni-ingest school_notes.yaml \
  --input lessons/soil-erosion.pdf \
  --output build/soil-erosion.json
```

Progress goes to stderr. Final JSON goes to requested file. Run id appears near start of progress output. Keep it when a long job matters because failed runs can be resumed later.

!!! tip
    Inspect JSON before writing to a vector store. It is much easier to spot broken text or poor chunks in a file than through search results.

Successful commands exit with status `0`. A failed pipeline exits with status `1` and prints step error. This makes normal shell checks work:

```bash
if omni-ingest school_notes.yaml --input lessons/soil-erosion.pdf --output build/result.json; then
  printf 'Ingestion complete\n'
else
  printf 'Ingestion failed\n' >&2
fi
```

## Pipeline options

Each pipeline can declare parameters. OmniIngest reads their JSON Schema before building argument parser. Property names become flags and descriptions become help text.

```bash
omni-ingest profiles/shiksha_chapter.yaml --help
```

A parameter named `start_page` becomes `--start-page`. Types also come from schema. Integers become integer arguments. Enum values become choices. Boolean parameters gain `--name` and `--no-name` forms.

```bash
omni-ingest profiles/shiksha_chapter.yaml \
  --input textbooks/science-9.pdf \
  --output build/science-9.json \
  --start-page 3 \
  --end-page 20 \
  --toc-start-page 7 \
  --toc-end-page 11
```

Defaults shown by `--help` come from pipeline YAML. Required schema properties become required CLI options.

!!! note
    Put pipeline-specific flags after standard arguments for readability. Argument parser accepts another order but tidy commands are easier to review in job logs.

## Several inputs

Pass several paths after `--input` when every file should use same pipeline and parameters.

```bash
omni-ingest school_notes.yaml \
  --input lessons/rainfall.pdf lessons/soil.pdf lessons/forests.pdf \
  --output build/geography.json
```

Each resource gets its own pipeline run and run id. One JSON destination contains a `runs` array:

```bash
jq '.runs[] | {source: .resource.uri, chunks: (.items | length)}' build/geography.json
```

Shell wildcards are handy for a local batch:

```bash
omni-ingest school_notes.yaml \
  --input lessons/*.pdf \
  --output build/all-lessons.json
```

Shell expands unquoted `*` before OmniIngest starts. Quote a wildcard when fsspec should expand it instead. This matters for remote paths and can also be used locally.

```bash
omni-ingest school_notes.yaml \
  --input 'az://textbooks/class-9/*.pdf' \
  --output build/class-9.json
```

Without quotes your shell may try to expand remote pattern as a local filename. Usually it leaves unmatched text alone but shell settings differ. Quoting removes that ambiguity.

## Standard input

Use `--input -` when another command produces source bytes. This avoids a temporary file:

```bash
curl -fsSL 'https://example.org/reports/energy-transition.pdf' \
  | omni-ingest report_index.yaml --input - --input-name energy-transition.pdf --output build/energy.json
```

`--input-name` gives piped resource a useful filename. Agents which inspect suffix can then recognise `.pdf`, `.docx`, or another format. Without this hint resource URI is `stdin://` and content detection must rely on bytes.

Standard input represents one resource. It cannot be mixed with file paths:

```bash
cat notes.md | omni-ingest note_index.yaml --input - handbook.md --output build/notes.json
```

This command is rejected because it is unclear how stdin should sit among named resources. Run separate commands or write stream to a file first.

Input and output can both use standard streams. This makes OmniIngest a normal Unix pipeline stage:

```bash
curl -fsSL 'https://example.org/notes/geography.md' \
  | omni-ingest note_index.yaml --input - --input-name geography.md --output - --quiet \
  | jq '.items[] | .content'
```

!!! note
    Persistent metadata store checkpoints stdin bytes with run state. This permits resume but can make checkpoint large. Use a named local or remote resource for very large recoverable jobs.

## Remote input

Input reading uses fsspec. A protocol prefix chooses filesystem implementation.

```bash
omni-ingest school_notes.yaml \
  --input 'az://textbooks/class-9/science.pdf' \
  --output build/science.json
```

Azure access is provided by `adlfs`. Credentials follow its normal fsspec configuration and environment conventions. Other protocols may need their matching fsspec package such as `s3fs` for S3 or `gcsfs` for Google Cloud Storage.

HTTP resources work without cloud filesystem setup:

```bash
omni-ingest article_index.yaml \
  --input 'https://example.org/reports/energy-transition.html' \
  --output build/energy-transition.json
```

Under hood OmniIngest calls `fsspec.core.url_to_fs()` then expands path. Every matched resource becomes a separate run. Resource bytes are read before first pipeline step.

!!! warning
    A broad wildcard can start many runs at once. Narrow remote patterns by directory or filename when memory is tight.

## Remote output

JSON output also uses fsspec. Local paths and remote URIs therefore share one command shape.

```bash
omni-ingest school_notes.yaml \
  --input lessons/rainfall.pdf \
  --output 'az://ingestion-results/geography/rainfall.json'
```

Output format normally comes from filename extension. Use `--format` when destination name does not advertise it:

```bash
omni-ingest school_notes.yaml \
  --input lessons/rainfall.pdf \
  --output 'az://ingestion-results/geography/latest.txt' \
  --format json
```

Explicit format wins over extension inference. Current CLI supports JSON file output. fsspec support still depends on filesystem driver being installed and its credentials being available.

## Standard output

Use `-` to write JSON to stdout.

```bash
omni-ingest school_notes.yaml --input lessons/rainfall.pdf --output -
```

This is useful when next tool can answer a question without an intermediate file:

```bash
omni-ingest school_notes.yaml \
  --input lessons/rainfall.pdf \
  --output - \
  | jq '.items[] | {text: .content, page: .metadata.page}'
```

For several inputs use `.runs[]`:

```bash
omni-ingest school_notes.yaml \
  --input lessons/*.pdf \
  --output - \
  | jq -r '.runs[] | [.resource.uri, (.items | length)] | @tsv'
```

A glob or directory can match many resources. `--max-concurrent-resources` limits how many run at the same time. Default is 8. Lower it to stay inside model provider rate limits.

Progress uses stderr so it does not corrupt JSON pipe. Add `--quiet` when progress is not useful in CI or cron:

```bash
omni-ingest school_notes.yaml \
  --input lessons/*.pdf \
  --output - \
  --quiet \
  | jq '.runs | length'
```

Stdout also works well for compression or atomic hand-off:

```bash
omni-ingest school_notes.yaml --input lessons/*.pdf --output - --quiet \
  | gzip > build/lessons.json.gz
```

## Store outputs

Some destinations are configured services rather than files. `vector:default` writes final knowledge items to vector store selected by environment settings.

```bash
omni-ingest retrieval.yaml \
  --input handbooks/*.pdf \
  --output vector:default
```

`graph:default` writes graph data produced by pipeline:

```bash
omni-ingest knowledge_graph.yaml \
  --input policies/*.pdf \
  --output graph:default
```

These names are destination selectors. They do not use fsspec and do not create files. Pipeline still needs suitable final items or graph metadata for chosen writer.

## Low-resource runs

Default metadata store writes run records, checkpoints, and knowledge item records to local SQLite database. Set `METADATA_STORE=null` for disposable runs where final output is enough.

```bash
METADATA_STORE=null omni-ingest school_notes.yaml \
  --input lessons/rainfall.pdf \
  --output build/rainfall.json
```

Null store uses in-memory SQLite for bookkeeping and stores empty content for knowledge item records. This avoids persistent database file and avoids retaining a second copy of item payload there. Current pipeline context still holds data needed by active steps and output writer.

For repeated use put setting in project `.env`:

```dotenv
METADATA_STORE=null
```

CLI loads `.env` from current working directory before settings are created.

!!! warning
    Null metadata store does not save checkpoints. Runs using it cannot be resumed. Use persistent metadata store for expensive jobs where recovery matters.

!!! warning
    Checkpoints are stored as Python pickles and `--resume` loads them. Treat the metadata database as trusted state, and do not resume from a database file or connection that someone you do not trust can write to.

## Resume failed work

Persistent metadata store saves checkpoint after each completed stage. If later stage fails then rerun with run id printed by original command.

```bash
omni-ingest school_notes.yaml \
  --resume 5fb58f3c-3a1c-47a1-9b12-7bb0d1a4c936 \
  --output build/recovered.json
```

Use same pipeline YAML. Runner restores resource, items, metadata, tenant, and completed stage count from checkpoint. It then starts at next stage. A changed pipeline can put stage number against different work and is rejected when checkpoint lies beyond new stage count.

Pipeline parameters are still parsed during resume. Supply same values when they affect steps which remain to run:

```bash
omni-ingest profiles/shiksha_chapter.yaml \
  --resume 5fb58f3c-3a1c-47a1-9b12-7bb0d1a4c936 \
  --output build/chapter.json \
  --start-page 3 \
  --end-page 20
```

Several failed runs can resume into one JSON result:

```bash
omni-ingest school_notes.yaml \
  --resume \
    5fb58f3c-3a1c-47a1-9b12-7bb0d1a4c936 \
    14635d18-df6f-47c7-a70d-cec68f747492 \
  --output build/recovered-batch.json
```

Successful runs remove their checkpoints. Resume is recovery rather than replay.

## Automation pattern

A practical batch job often uses remote input, quiet progress, explicit output format, and persistent metadata for recovery.

```bash
omni-ingest nightly_index.yaml \
  --input 'az://incoming-handbooks/2026-06/*.pdf' \
  --output 'az://processed-handbooks/2026-06/run-output' \
  --format json \
  --quiet
```

Keep stderr in job logs because it contains run ids and failures. Keep stdout free when `--output -` feeds another process. Choose null metadata only when rerunning from start is cheaper than preserving checkpoint.
