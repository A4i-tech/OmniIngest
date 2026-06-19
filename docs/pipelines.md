# Pipelines

Pipeline steps run sequentially unless adjacent steps share a `stage`.

```yaml
steps:
  - agent: text
  - agent: extract
    stage: analysis
    config:
      path: ".metadata.summary"
      prompt: Summarize the document.
      schema: {type: string}
  - agent: extract
    stage: analysis
    config:
      path: ".metadata.keywords"
      prompt: Extract search keywords.
      schema: {type: array, items: {type: string}}
  - agent: embedding
```

The two `extract` steps run concurrently after `text`. `embedding` starts when both finish.

!!! warning
    If a step fails, other steps already running in the same stage are allowed to finish and their results are recorded. Subsequent stages are skipped, and no pipeline output is produced.

!!! note
    - Only group independent steps. Steps within a stage share the same mutable ingestion context, so they must not replace or depend on each other's intermediate state.
    - Stage names must be contiguous: once another stage or an unstaged step begins, an earlier stage name cannot be reused. Each unstaged step remains sequential.
    - Pipeline components may define stages among their own leaf steps. A component reference cannot itself be assigned a stage.