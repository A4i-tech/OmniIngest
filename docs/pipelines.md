# Pipeline playbook

Pipeline YAML is a recipe for turning source files into useful knowledge items. It names agents, gives them config, and says which bits can run together. It does not choose final storage by itself. CLI `--output` does that part.

Most good pipelines start dull. First make text. Then chunk it. Then inspect JSON. Only after that add model extraction, validation, embeddings, graph output, or page tricks. This saves a lot of faff because broken extraction is much easier to debug before vector storage gets involved.

OmniIngest loads YAML into `PipelineConfig`, expands any YAML components, builds step classes from registry names, then runs stages in order. Parameter interpolation happens before step objects are created. Validation runs around each step during execution.

## First pipeline

This is enough for a real document indexing job.

```yaml
pipeline_id: school_notes
domain_profile: education
input_modality: document

steps:
  - agent: text
  - agent: sentence_chunking
  - agent: embedding
```

Run it with JSON output first:

```bash
omni-ingest school_notes.yaml --input chapter-5-land-resources.pdf --output out.json
```

!!! tip
    Open `out.json` and check chunks. If text looks mangled here, embeddings will only make bad chunks searchable. Fix parsing or chunking before moving on.

When chunks look right, write to configured vector store:

```bash
omni-ingest school_notes.yaml --input chapter-5-land-resources.pdf --output vector:default
```

Step by step:

- `text` converts incoming resource into text when needed.
- `sentence_chunking` replaces current item set with text chunks. It splits at line breaks and hard-cuts overlong lines. It has no overlap.
- `embedding` adds vectors to current items.
- output writer saves final context.

Under hood each chunk is a `KnowledgeItem`. It carries bytes, source URI, tenant id where known, and metadata. Most agents either update `ctx.items` or root `ctx.metadata`.

## Pipeline shape

A pipeline can be as small as `steps`. Extra top-level fields make runs easier to understand later.

```yaml
pipeline_id: textbook_index
domain_profile: education
input_modality: document
description: "Prepare textbook pages for retrieval and lesson planning"

steps:
  - agent: text
  - agent: sentence_chunking
  - agent: embedding
```

Use `pipeline_id` as a stable name. Use `domain_profile` when same store holds several domains. Use `input_modality` as a human hint. Current runner does not need it to execute steps.

`steps` is ordered. Unless steps share a stage, next step waits for previous step.

## Step config

Each step has an `agent`. Config goes under `config`.

```yaml
steps:
  - agent: text

  - agent: sentence_chunking
    config:
      chunk_size: 900

  - agent: embedding
```

Only set knobs that matter. Defaults are part of agent code. For `sentence_chunking`, default size is useful for quick tests, but textbooks often benefit from bigger chunks because examples and explanation sit a few paragraphs apart.

Implementation note: config is passed into Pydantic model fields for that agent. Unknown or wrongly typed values fail while pipeline is created, not halfway through ingestion.

## Parameters

Parameters are JSON Schema for values supplied from CLI. They are useful when same YAML should run for different books, tenants, or quality bars.

```yaml
pipeline_id: configurable_textbook_index
domain_profile: education

parameters:
  properties:
    chunk_size:
      type: integer
      minimum: 200
      default: 900

steps:
  - agent: text
  - agent: sentence_chunking
    config:
      chunk_size: ${.config.chunk_size}
  - agent: embedding
```

Run with defaults:

```bash
omni-ingest configurable_textbook_index.yaml --input science.pdf --output vector:default
```

Override values:

```bash
omni-ingest configurable_textbook_index.yaml \
  --input science.pdf \
  --output vector:default \
  --chunk-size 700
```

Interpolation has two modes. If whole YAML value is `${...}`, result keeps its JSON type. That is why `chunk_size` stays an integer. If `${...}` appears inside a longer string, result becomes text.

!!! note
    Use parameters for choices a caller will actually change. Do not parameterise every small constant. That makes YAML look clever and read poorly.

## Structured output

`extract` is where model output becomes useful metadata. It selects input with jq, asks a model for structured output, validates that output against JSON Schema, then writes it back with a jq path.

Here is a normal classroom use case. Each chunk gets lesson planning metadata.

```yaml
pipeline_id: lesson_metadata
domain_profile: education

steps:
  - agent: text
  - agent: sentence_chunking
    config:
      chunk_size: 1200

  - agent: extract
    config:
      foreach: ".items[]"
      input: ".text"
      path: ".metadata.lesson"
      prompt: |
        Extract lesson planning metadata from this textbook passage.
        Keep wording short and useful for a teacher.
      schema:
        type: object
        properties:
          topic:
            type: string
          grade_level:
            type: string
          key_terms:
            type: array
            items:
              type: string
          classroom_activity:
            type: string
        required: [topic, grade_level, key_terms]

  - agent: embedding
```

Likely output on one item:

```json
{
  "lesson": {
    "topic": "soil erosion",
    "grade_level": "class 7",
    "key_terms": ["topsoil", "runoff", "vegetation"],
    "classroom_activity": "Compare soil loss from covered and bare trays after watering."
  }
}
```

`foreach` decides how many model calls happen. `.items[]` means one call per item. `.` means one call for whole context. `input` decides what each call sees. `path` decides where output lands.

!!! tip
    Use strict schemas. They are not decoration. `ExtractAgent` builds a Pydantic output model from schema and asks Pydantic AI for native structured output. Bad shapes fail close to source instead of leaking into downstream storage.

## Root metadata extraction

Sometimes result belongs to run as a whole. Quality checks, document titles, chapter lists, and run summaries are root metadata.

This pipeline samples first few chunks and writes one quality report.

```yaml
pipeline_id: quality_check

parameters:
  properties:
    sample_size:
      type: integer
      minimum: 1
      default: 4
    min_quality_score:
      type: number
      minimum: 0
      maximum: 1
      default: 0.75

steps:
  - agent: text
  - agent: sentence_chunking

  - agent: extract
    config:
      foreach: "."
      input: '.items[:${.config.sample_size}] | map(.text[:700]) | join("\n\n--- sample ---\n\n")'
      path: ".metadata.quality_report"
      prompt: |
        Review extracted text quality.
        Score usefulness for retrieval from 0 to 1.
        List concrete extraction issues only.
      schema:
        type: object
        properties:
          overall_score:
            type: number
            minimum: 0
            maximum: 1
          issues:
            type: array
            items:
              type: string
        required: [overall_score, issues]
    validation:
      - mode: after
        expr: ".metadata.quality_report.overall_score >= ${.config.min_quality_score}"
        message: "Quality check failed (score = ${.metadata.quality_report.overall_score})"
```

Notice where validation sits. It belongs to step that produces `quality_report`. That keeps cause and check next to each other.

## Step validation

Validation controls whether a step may run and whether its result is good enough to keep. Most false checks fail pipeline. A false `condition` check skips its step instead.

```yaml
validation:
  - mode: after
    expr: ".result.metadata.items_processed >= 3"
    message: "Only ${.result.metadata.items_processed} item(s) were enriched"
```

Every check has:

- `mode`: `condition`, `before`, `after`, or `plain`.
- `expr`: jq expression that must be truthy.
- `message`: error text. `${...}` runs jq against validation document.

Validation document has:

```yaml
metadata: root metadata
items: current items with id, source_uri, and metadata
result: step result for after checks only
```

`condition` runs first. Use it for optional work rather than making an agent understand pipeline parameters it does not need.

```yaml
parameters:
  properties:
    translate_to:
      type: [string, "null"]
      default: null

steps:
  - agent: translate
    validation:
      - mode: condition
        expr: '${if .config.translate_to == null then "false" else "true" end}'
        message: Translation not requested
    config:
      src: en
      dst: '${.config.translate_to // ""}'
      input: ".metadata.publication"
      path: ".metadata.publication_translated"
```

With no target language this step is recorded as skipped and pipeline carries on. A configured language runs translation as normal. Skip reason comes from `message`.

`before` runs before step. Use it when later work makes no sense without some existing state.

```yaml
validation:
  - mode: before
    expr: ".items | length > 0"
    message: "Cannot enrich empty item set"
```

`after` runs only when step succeeded. Use it for quality gates and count checks.

```yaml
validation:
  - mode: after
    expr: ".metadata.quality_report.overall_score >= 0.8"
    message: "Quality score too low: ${.metadata.quality_report.overall_score}"
```

`plain` runs before step. If it passes, remaining validation checks for that step are skipped. Step still runs. This mirrors Pydantic plain validators: one validator takes over validation chain.

```yaml
validation:
  - mode: plain
    expr: ".metadata.trusted_source == true"
    message: "Source is not trusted"
  - mode: after
    expr: ".metadata.quality_report.overall_score >= 0.8"
    message: "Quality score too low"
```

That example says trusted sources do not need quality validation. If `trusted_source` is true, after check is skipped. If false, step fails before it runs.

!!! warning
    Use `plain` sparingly. It is sharp. Most pipelines want ordinary `before` and `after`.

## Transforming metadata

`transform` is for deterministic jq work. No model call. No new content. Just reshape current context.

```yaml
steps:
  - agent: transform
    config:
      value: '.metadata.all_key_terms = ([.items[].metadata.lesson.key_terms[]?] | unique | sort)'
```

This is ideal after item-level extraction. Let model extract key terms per chunk. Let jq fold them into one clean list.

Implementation detail: transform writes back `metadata` and/or `items`. Default is `metadata`.

## Translating structured output

`translate` reads text, an object, or an ordered list from `input`. Translated data is written to `path`, so source metadata can remain available for later steps.

```yaml
- agent: translate
  config:
    src: null
    dst: cy
    input: ".metadata.publication"
    path: ".metadata.publication_welsh"
    skip_fields:
      - "*.record_id"
      - "*.sections.*.section_id"
```

`src: null` asks provider to detect source language. `skip_fields` uses shell-style patterns against paths such as `0.record_id` or `0.sections.2.section_id`. Matching values are not sent for translation. This is useful for identifiers, enum values, URLs, and other strings that must stay put.

## Stages

Stages let adjacent independent steps run together.

```yaml
steps:
  - agent: text
  - agent: sentence_chunking

  - agent: extract
    stage: analysis
    config:
      foreach: ".items[]"
      input: ".text"
      path: ".metadata.summary"
      prompt: "Summarise this passage for a teacher."
      schema:
        type: string

  - agent: extract
    stage: analysis
    config:
      foreach: ".items[]"
      input: ".text"
      path: ".metadata.key_terms"
      prompt: "Extract important terms from this passage."
      schema:
        type: array
        items:
          type: string

  - agent: embedding
```

Both `analysis` steps run after `sentence_chunking`. `embedding` waits for both.

!!! warning
    Only group steps that do not fight over same state. Two extracts writing different item metadata paths are fine. Two steps replacing `ctx.items` in same stage are asking for a dodgy day.

Stage names must be contiguous. Once another stage or unstaged step appears, older stage name cannot return. This keeps resume checkpoints simple: runner checkpoints after each completed stage index.

If one step fails, other already running steps in same stage finish and get recorded. Later stages do not run.

## Components

A component is just another YAML pipeline used as a step. This is how `quality`, `kg_builder`, and other profiles stay reusable.

```yaml
pipeline_id: education_search
domain_profile: education

steps:
  - agent: text
  - agent: sentence_chunking
  - agent: quality
    config:
      min_quality_score: 0.6
  - agent: faq_generation
  - agent: enrichment
  - agent: embedding
```

When runner sees `agent: quality`, it first looks for a registered Python step. If none exists, it tries `quality.yaml` near current file, then configured profile directory. `PROFILES_DIR` changes that directory from its packaged `src/omni_ingest/profiles` default. Found YAML gets flattened into leaf steps.

Config on component reference becomes parameter overrides for component YAML. That is why `min_quality_score` reaches `quality.yaml` in profile directory.

Component reference cannot have `stage` or `validation`. Put those inside component YAML. Otherwise it is unclear whether wrapper or inner steps should own execution behaviour.

## PDF pages

Text chunking is enough for many PDFs. Page workflows matter when page numbers, chapters, or original PDF slices matter.

Text page indexing:

```yaml
pipeline_id: page_text_index
domain_profile: education

steps:
  - agent: page_chunking
    config:
      mode: text
  - agent: extract
    config:
      foreach: ".items[]"
      input: ".text"
      path: ".metadata.page_summary"
      prompt: "Summarise this page in one sentence for a lesson plan."
      schema:
        type: string
  - agent: embedding
```

`mode: text` extracts text per page. This is cheap and makes `.text` available to `extract`.

PDF page stitching:

```yaml
pipeline_id: chapter_pdf_builder
domain_profile: education

steps:
  - agent: page_chunking
    config:
      mode: pdf

  - agent: extract
    config:
      foreach: "."
      input: ".items[] | {id, page: .metadata.page}"
      path: ".metadata.chapter_pages"
      prompt: |
        Group pages into textbook chapters.
        Return chapter number and page item ids for each chapter.
      schema:
        type: array
        items:
          type: object
          properties:
            chapter:
              type: integer
            title:
              type: string
            item_ids:
              type: array
              items:
                type: string
          required: [chapter, item_ids]

  - agent: page_stitching
    config:
      foreach: ".metadata.chapter_pages[]"
      input: ".item_ids"
      metadata: "{chapter: .chapter, title: .title}"
      replace: true
```

`mode: pdf` does not copy page bytes into every item. It emits virtual page content with a URI like `omni-ingest://resource/pages/7`. Actual one-page PDF bytes are materialised only when a later step calls `await item.content.read(ctx)`. That keeps memory sane for fat textbooks.

!!! note
    `page_stitching` can replace page items with stitched chapter PDFs. Leave `replace: true` when you only care about chapters. Set it false only if downstream work needs both pages and stitched outputs.

## Buffered extraction

Sometimes model should inspect many page items but not receive all page bytes at once. Use `mode: buffered`.

```yaml
steps:
  - agent: page_chunking
    config:
      mode: pdf

  - agent: extract
    config:
      mode: buffered
      foreach: "."
      input: "[.items[].id]"
      path: ".metadata.chapter_outline"
      prompt: |
        Read pages in order with available tool.
        Return chapter title and first page for each chapter.
      schema:
        type: array
        items:
          type: object
          properties:
            title:
              type: string
            first_page:
              type: integer
          required: [title, first_page]
```

In buffered mode, model gets a tool for reading selected items one at a time. This fits virtual PDF pages nicely. It avoids stuffing whole document into one prompt.

## Knowledge graphs

Use `kg_builder` when graph output is wanted.

```yaml
pipeline_id: science_graph
domain_profile: education

steps:
  - agent: text
  - agent: sentence_chunking
  - agent: kg_builder
```

Run:

```bash
omni-ingest science_graph.yaml --input ecosystem.pdf --output graph:default
```

`kg_builder` is a YAML component. It extracts nodes and edges per item, then folds them into root `.metadata.kg_data`. `graph:default` reads that metadata and writes graph store rows.

This split matters. Graph building is pipeline logic, not special Python agent logic. You can open `src/omni_ingest/profiles/kg_builder.yaml` and change prompt or schema like any other component.

## Tool vendors

Tool vendors configure MCP servers for agents that use tools.

```yaml
pipeline_id: cited_enrichment

tool_vendors:
  curriculum_api:
    command: uv
    args: ["run", "curriculum-mcp"]

steps:
  - agent: text
  - agent: sentence_chunking
  - agent: extract
    config:
      toolsets: [curriculum_api]
      foreach: ".items[]"
      input: ".text"
      path: ".metadata.curriculum_links"
      prompt: |
        Match passage to curriculum outcomes.
        Use available curriculum tool before answering.
      schema:
        type: array
        items:
          type: object
          properties:
            outcome_id:
              type: string
            reason:
              type: string
          required: [outcome_id, reason]
```

Runner creates MCP toolsets once and stores them on `PipelineRunner`. `AgentMixin` looks them up by name when an agent runs.

Keep tool use boring. If model can answer from item text, do not add a tool. Tools are for live data, private systems, or expensive context you do not want in every prompt.

## Outputs

YAML defines processing. CLI output defines sink.

```bash
# JSON file for inspection
omni-ingest pipeline.yaml --input textbook.pdf --output out.json

# JSON to stdout
omni-ingest pipeline.yaml --input textbook.pdf --output -

# configured vector store
omni-ingest pipeline.yaml --input textbook.pdf --output vector:default

# configured graph store
omni-ingest pipeline.yaml --input textbook.pdf --output graph:default
```

!!! info
    JSON output resolves item content and writes readable data where possible. Vector output deduplicates items, upserts kept items, then records child knowledge items in metadata store when parent item id exists. Graph output looks for `.metadata.kg_data`.

That is why output should come last in thinking. Build context first. Store it after it looks right.

## Resume

Pipeline run checkpoints after each successful stage. If a run fails, fix YAML or config and resume with run id.

```bash
omni-ingest pipeline.yaml --resume 5fb58f3c-3a1c-47a1-9b12-7bb0d1a4c936 --output out.json
```

Completed stages are skipped. Checkpoint is cleared after success.

This works best with sensible stages. Do not put half a pipeline into one giant stage unless those steps truly can run together.

## Build order

Good pipeline work usually goes like this:

1. Start with `text`, `sentence_chunking`, and JSON output.
2. Inspect actual chunks.
3. Tune chunk size.
4. Add one `extract` step with strict schema.
5. Add `transform` for deterministic cleanup.
6. Add validation when failure condition is obvious.
7. Add stages only for independent work.
8. Move repeated blocks into components.
9. Add `embedding`, `kg_builder`, vector output, or graph output last.

If something feels woolly, print JSON again. Pipeline bugs are usually obvious when current `metadata` and `items` are visible.
