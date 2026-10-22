# Advanced pipeline patterns

Advanced pipelines are rarely difficult because they have many steps. They become difficult when shape of work changes at run time. One document may contain several record types. One question may need a short answer while another needs four options and a marked answer. Some pages should be read only when model asks for them.

This page covers patterns for those jobs. It assumes ordinary steps, parameters, structured output, and jq transforms are already familiar. See [Pipeline playbook](pipelines.md) for those foundations.

## Dynamic structured output

Most `extract` steps use one literal JSON Schema:

```yaml
schema:
  type: object
  properties:
    title: {type: string}
    summary: {type: string}
  required: [title, summary]
```

That works when every selected item has same shape. Question banks are less tidy. A multiple-choice question needs options and a correct answer. A descriptive question needs prose. Making every field optional produces loose data and pushes validation downstream.

`schema` can instead be a jq expression. Expression runs against each value produced by `foreach`.

```yaml
- agent: extract
  description: Model each question
  config:
    foreach: |
      .items[]
      | {
          id,
          text,
          metadata,
          modelSchema: (
            if .metadata.answer_type == "multiple_choice" then
              {
                type: "object",
                additionalProperties: false,
                properties: {
                  question: {type: "string"},
                  options: {
                    type: "array",
                    minItems: 4,
                    maxItems: 4,
                    items: {type: "string"}
                  },
                  answer: {type: "string"}
                },
                required: [question, options, answer]
              }
            else
              {
                type: "object",
                additionalProperties: false,
                properties: {
                  question: {type: "string"},
                  answer: {type: "string"}
                },
                required: [question, answer]
              }
            end
          )
        }
    input: ".text"
    path: ".metadata.answer"
    prompt: "Clean this textbook question and preserve its answer exactly."
    schema: ".modelSchema"
```

`foreach` builds one work record per item. Each record carries item id, prompt text, metadata, and schema for that item. `ExtractAgent` evaluates `.modelSchema` then creates a Pydantic output model from resulting JSON Schema.

Keeping `id` matters. It tells extraction to write output back to matching knowledge item. Without `id`, output is treated as root metadata.

!!! tip
    Use `additionalProperties: false` in dynamic schemas. Runtime-generated models are most useful when they reject fields which do not belong to selected record type.

Dynamic schema is not a prompt hint. It is output contract. If model returns three options for a four-option question then output validation fails and model gets another attempt.

## Build work before calling a model

Complicated extraction becomes easier when jq first decides what work exists. Model should answer a prepared task rather than discover pipeline control flow from a long prompt.

Consider chapter questions spread across several PDF chunks. A sensible work record contains:

- target question chunk
- answer chunks for same chapter
- exact questions expected in output
- schema requiring every expected key

```yaml
- agent: extract
  description: Resolve model answers
  config:
    mode: buffered
    foreach: |
      . as $doc
      | .items[]
      | select(.metadata.kind == "question_chunk")
      | . as $question_chunk
      | ($question_chunk.metadata.questions // []) as $questions
      | {
          id: $question_chunk.id,
          inputs: (
            [$question_chunk.id]
            + [
                $doc.items[]
                | select(.metadata.kind == "answer_chunk")
                | select(.metadata.chapter == $question_chunk.metadata.chapter)
                | .id
              ]
          ),
          modelSchema: {
            type: "object",
            additionalProperties: false,
            properties: (
              reduce $questions[] as $question
                ({}; . + {
                  ($question.key): {
                    type: "object",
                    properties: {
                      question: {type: "string"},
                      answer: {type: "string"}
                    },
                    required: [question, answer]
                  }
                })
            ),
            required: [$questions[].key]
          }
        }
    input: ".inputs"
    path: ".metadata.model_answers"
    prompt: |
      Read question PDF first then inspect answer PDFs with see_next.
      Return every property required by output schema.
      Copy printed answers rather than answering from general knowledge.
    schema: ".modelSchema"
```

This combines three features:

- `foreach` plans tasks with jq.
- `mode: buffered` exposes selected PDFs one at a time.
- dynamic schema prevents model from omitting a question.

Implementation stays fairly plain. Runner does not have a separate workflow graph for every question type. jq emits records and extraction handles each record with its own contract.

!!! warning
    Check `inputs` is not empty while building work record. Buffered extraction cannot read missing items and a vague model answer may look successful despite having no source material.

After fan-out, fold item results into ordered root output with `transform`:

```yaml
- agent: transform
  description: Assemble chapter answers
  config:
    path: ".metadata.chapter_answers"
    value: |
      [
        .items[]
        | select(.metadata.model_answers != null)
        | {
            chapter: .metadata.chapter,
            answers: .metadata.model_answers
          }
      ]
      | sort_by(.chapter)
```

Always sort when output order is part of contract. Concurrent extraction finishes in completion order rather than source order.

## Retry semantic mistakes

JSON Schema checks shape. It cannot express every rule which makes output useful. `assertions` run jq against extracted value after schema validation. Failed assertions raise `ModelRetry` with supplied feedback.

```yaml
- agent: extract
  config:
    foreach: ".items[]"
    input: ".text"
    path: ".metadata.evidence"
    prompt: |
      Extract claims supported by this page.
      Keep quoted evidence short.
    schema:
      type: object
      properties:
        claims:
          type: array
          items:
            type: object
            properties:
              claim: {type: string}
              evidence: {type: string}
              page: {type: integer}
            required: [claim, evidence, page]
      required: [claims]
    assertions:
      - expr: "all(.claims[]; .page == $item.metadata.page)"
        feedback: "Use only page number supplied with current item."
      - expr: "all(.claims[]; (.evidence | length) <= 240)"
        feedback: "Shorten evidence to at most 240 characters without changing its words."
```

Assertion input is extracted output. `$item` contains current `foreach` record. This lets rule compare model result with source metadata without copying values into prompt.

Use schema for types, required fields, lengths, enums, and patterns. Use assertions for relationships across fields or source-aware rules. Keeping that split avoids a pile of brittle prompt instructions.

## Choose models per step

One default model is convenient but not compulsory. Every model-backed agent accepts `model`. Embedding step has its own model setting.

```yaml
steps:
  - agent: extract
    description: Classify document
    config:
      model: "openai:gpt-5-mini"
      foreach: "."
      input: ".items | map(.text[:500]) | join(\"\n\")"
      path: ".metadata.classification"
      prompt: "Classify document by subject and school grade."
      schema:
        type: object
        properties:
          subject: {type: string}
          grade: {type: integer}
        required: [subject, grade]

  - agent: extract
    description: Resolve difficult questions
    config:
      model: "anthropic:claude-sonnet-4-6"
      foreach: ".items[]"
      input: ".text"
      path: ".metadata.answer"
      prompt: "Extract question and worked answer without paraphrasing."
      schema:
        type: object
        properties:
          question: {type: string}
          answer: {type: string}
        required: [question, answer]

  - agent: embedding
    config:
      model: "openai:text-embedding-3-small"
```

Use quick model where task is narrow. Reserve stronger model for ambiguous pages or strict extraction. This usually saves more time than squeezing another sentence out of prompt.

Provider package and credentials must be installed for every model named by pipeline. Model choice changes execution only. It does not change storage destination.

## Virtual items

A knowledge item contains a `ByteContent`. The byte content can hold inline bytes or a virtual content URI. It keeps `content_type` in its internal metadata.

```python
KnowledgeItem(
    content=ByteContent.from_uri(
        "omni-ingest://resource/pages/7,8,9,10",
        content_type="application/pdf",
    ),
)
```

URI describes how content can be rebuilt. It is not final output path and does not need to point at a separate stored file. Route above means “take pages 7 to 10 from current source PDF”.

Consumers use same API for inline and virtual items:

```python
content = await item.content.read(ctx)
```

`ByteContent.read(ctx)` returns inline bytes or passes the URI to the context's `ContentResolver`. This keeps virtual routing out of agents that read content. Extractor, rasteriser, and output writer need not branch on item type.

PDF pipelines benefit because same source pages often appear in page items, chapter PDFs, metadata samples, and prompts. Copying each version wastes memory.

```text
omni-ingest://resource/pages/7
omni-ingest://resource/pages/7,8,9,10
```

`page_chunking` in PDF mode emits one-page virtual items. `page_stitching` flattens source pages into one short route. Stitching virtual stitches produces an encoded `pdf.concat` chain instead. Combined PDF is built only when a consumer asks for content.

```yaml
steps:
  - agent: page_chunking
    config:
      mode: pdf

  - agent: page_stitching
    config:
      foreach: |
        {
          chapter: 3,
          ids: [
            .items[]
            | select(.metadata.page >= 24 and .metadata.page <= 37)
            | .id
          ]
        }
      input: ".ids"
      metadata: '{chapter: .chapter}'
      replace: true
```

Rasterisation, model input, or JSON output materialises bytes because those consumers need real content. Items with inline bytes use eager stitching. Virtual inputs can remain virtual across repeated stitching.

!!! tip
    Select pages before rasterisation. Virtual stitch costs almost nothing. Rasterising whole textbook then selecting twelve pages does expensive work in wrong order.

Custom virtual routes implement `ContentResolver` protocol from `core.protocol`. Add resolver function to `CompositeContentResolver` when bytes come from another pipeline-scoped source. URI should describe stable recipe rather than hide large encoded payload.

## When Python is cleaner

YAML is good at selection, composition, validation, and calling existing agents. It is poor place for a new parsing algorithm or stateful integration.

Move work into Python when it needs:

- a new binary parser
- external connection lifecycle
- complex state shared across calls
- domain algorithm which deserves unit tests of its own
- jq expression so long that nobody can review it confidently

Keep orchestration in YAML after Python step exists. This leaves pipeline readable and implementation testable without pretending every problem is data transformation.

## Run pipelines inside your application

Use `omni_ingest.run` to run a pipeline from Python. It does not read `sys.argv`. Several calls can run at the same time in one process.

```python
import asyncio
import omni_ingest

a = omni_ingest.run("/srv/pipelines/school_notes.yaml", "/data/a.pdf", output="/data/out/a.json")
b = omni_ingest.run("/srv/pipelines/school_notes.yaml", "/data/b.pdf", output="/data/out/b.json")
results = await asyncio.gather(a, b)
```

Each call has its own run ID. `run` creates the ID, so subscribe without `run_id` and filter on `e.run_id` to follow one call. `run` calls `event.shutdown(run_id)` when the run ends. Other runs keep their events. To subscribe with `run_id`, build the `IngestionContext` yourself and call the runner.

The `params` argument sets pipeline parameters. The `output` argument accepts an `Output` object or an output string such as `result.json`.

!!! warning
    Relative default paths resolve against the current working directory of the process. This applies to the SQLite metadata store, the LanceDB and Chroma directories, and relative output paths. A host process must pass absolute paths.

!!! note
    OmniIngest reads `Settings` once per process. Set environment variables before the first import of `omni_ingest`.

