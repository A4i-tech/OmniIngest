<div align="center">
  <picture>
    <img src="assets/logo.svg" alt="Omni Ingest">
  </picture>
</div>
<div align="center">
  <h3>Multimodal ingestion that does not dictate your stack</h3>
</div>
<div align="center">
 <a href="https://github.com/A4i-tech/OmniIngest/blob/a4i/main/LICENSE"><img src="https://img.shields.io/github/license/A4i-tech/OmniIngest" alt="License"/></a>
 <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version"/></a>
 <a href="https://github.com/A4i-tech/OmniIngest/actions/workflows/main.yaml"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/a4i-architect/9f6bf6c997ed2c171aed62e1c1583f2d/raw/omni_ingest.json" alt="Coverage"/></a>
 <a href="https://github.com/A4i-tech/OmniIngest/actions/workflows/main.yaml"><img src="https://github.com/A4i-tech/OmniIngest/actions/workflows/main.yaml/badge.svg?branch=a4i/main" alt="CI"/></a>
</div>

---

OmniIngest is a multimodal ingestion framework for building reusable knowledge pipelines. It turns documents, web pages, images, and audio into validated knowledge items without the faff of stitching together one-off preprocessing scripts.

**Documentation:** [omniingest.onrender.com](https://omniingest.onrender.com/)

## Why use OmniIngest

- **Multimodal processing.** PDF, DOCX, HTML, images, audio, and plain text use one pipeline model.
- **Multiple model providers.** Every model-backed step can select its own Pydantic AI model. Chat and embedding models are configured independently.
- **Storage-independent pipelines.** YAML describes transformation. CLI or Python caller chooses JSON, vector store, graph store, or another output.
- **Structured when needed.** JSON Schema output, jq transforms, and step validation turn model results into predictable data.
- **Built for sizeable jobs.** fsspec resources, concurrent stages, progress events, and stage checkpoints keep long ingestion runs manageable.

## Quick start

Install command-line tool with uv:

```bash
uv tool install omni-ingest
```

Create a pipeline which extracts text and makes sentence-aware chunks:

```yaml title="school_notes.yaml"
pipeline_id: school_notes
domain_profile: education

steps:
  - agent: text
  - agent: chunking
    config:
      chunk_size: 900
      overlap: 120
```

Run it against a PDF:

```bash
omni-ingest school_notes.yaml \
  --input lessons/soil-erosion.pdf \
  --output build/soil-erosion.json
```

YAML stops at transformation. Pick storage at run time:

```bash
omni-ingest school_notes.yaml --input lessons/soil-erosion.pdf --output - | jq '.items | length'
omni-ingest retrieval.yaml --input handbooks/*.pdf --output vector:default
```

## Learn more

- [Command-line usage](https://omniingest.onrender.com/cli/) covers files, remote resources, stdin, outputs, and resume.
- [Pipeline playbook](https://omniingest.onrender.com/pipelines/) builds from simple YAML to structured output, validation, components, and model tools.
- [Working with event stream](https://omniingest.onrender.com/events/) covers progress and live application integration.
- [API reference](https://omniingest.onrender.com/api/) documents Python interfaces.

## Development

```bash
git clone https://github.com/A4i-tech/OmniIngest.git
cd OmniIngest
uv sync --dev
uv run pytest tests/unit
```
