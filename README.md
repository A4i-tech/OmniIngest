# OmniIngest

[![License](https://img.shields.io/github/license/A4i-tech/OmniIngest)](https://github.com/A4i-tech/OmniIngest/blob/a4i/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/a4i-architect/9f6bf6c997ed2c171aed62e1c1583f2d/raw/omni_ingest.json)](https://github.com/A4i-tech/OmniIngest/actions/workflows/main.yaml)
[![CI](https://github.com/A4i-tech/OmniIngest/actions/workflows/main.yaml/badge.svg?branch=a4i/main)](https://github.com/A4i-tech/OmniIngest/actions/workflows/main.yaml)

Modern AI agents need high-quality data to work well. This framework helps you turn messy docs, audio, and web content into clean, search-ready data using modular pipelines. It is the "pre-processing" layer for your RAG or agentic memory.

Read the **[documentation](https://a4i-tech.github.io/OmniIngest)** for detailed information on agents, pipelines, and architecture.

## Key features

- **Multi-modal**: Process Audio, PDF, DOCX, and HTML in one place.
- **Agentic**: Chain specialized agents for transcription, parsing, and LLM enrichment.
- **Flexible storage**: Use the same schema for Vector, Graph, or Relational databases.

## Getting started

### Setup
```bash
uv sync
```

### Configuration
Export your environment variables or use a `.env` file:
```env
OPENAI_API_KEY=your_key_here
LANCEDB_PATH=./lancedb_data
METADATA_STORE=sqlite+aiosqlite:///./knowledge.db
```

### Usage
These example demonstrates a complete workflow: parsing a technical document, cleaning the markdown, extracting semantic tags and summaries, and writing the final context to a vector store.

```yaml
steps:
  - agent: text
  - agent: markdown_cleaning
  - agent: enrichment
  - agent: chunking
    config:
      chunk_size: 512
  - agent: embedding
```

To use Omni-Ingest via CLI, install Omni-Ingest as a `uv` tool. In particular, you pass 3 items - (a) the pipeline as a YAML file, (b) the input(s) to ingest, and (c) the output destination.
```bash
uv tool install git+https://github.com/A4i-tech/OmniIngest.git
```

```bash
# Run a pipeline and output all data in JSON format.
omni-ingest profiles/enrichment.yaml --input document.pdf --output out.json

# Process multiple inputs concurrently.
omni-ingest profiles/enrichment.yaml --input a.pdf b.pdf c.pdf --output out.json

# Extract just the metadata you need for a downstream script.
omni-ingest profiles/enrichment.yaml --input *.pdf --output - | jq '.runs[].metadata'

# Add embedded chunks to the configured vector store.
omni-ingest textbook_v1.yaml --input textbook.pdf --output vector:default

# Pipeline parameters become CLI flags.
omni-ingest profiles/quality.yaml --help
omni-ingest profiles/quality.yaml \
  --input document.pdf \
  --output - \
  --sample-size 5 \
  --min-quality-score 0.8
```

You may also run a pipeline on files from remote, e.g., an Azure Blob Storage account. Supported URIs include `ftp://`, `sftp://`, `s3://`, `dropbox://`, `gdrive://`, `zip://`, and many more. Visit [fsspec docs](https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations) for all supported URIs. Secured remote locations will often require specifying environment variables. E.g., Azure Blob Storage will require an `AZURE_STORAGE_CONNECTION_STRING` env. Read fsspec docs for envs.
```bash
omni-ingest profiles/enrichment.yaml --input "az://folder/class-*-maths.pdf" --output out.json
```

To invoke Omni-Ingest programatically, invoke `PipelineRunner.run()` with pipeline and input files.
You can build the pipeline programatically or use a helper function to build from YAML.
```python
import asyncio
from pathlib import Path
from omni_ingest.core.pipeline import create_pipeline_from_config
from omni_ingest.core.model import IngestionContext
from omni_ingest.core.output import VectorStoreOutput
from omni_ingest.port.vector_store import LanceDBVectorStore


async def main():
    pipeline = create_pipeline_from_config(Path("pipeline.yaml"))
    output = VectorStoreOutput(store=LanceDBVectorStore(table_name="knowledge_base"))
    ctx = IngestionContext(resource=Path("research_paper.pdf"))
    results = await pipeline.run(ctx, output=output)
    print(f"Processed {sum(len(e.items) for e in results)} items.")

if __name__ == "__main__":
    asyncio.run(main())
```