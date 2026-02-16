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
DATABASE_URL=sqlite+aiosqlite:///./knowledge.db
LANCEDB_PATH=./lancedb_data
```

### Usage: Semantic ingestion pipeline
This example demonstrates a complete workflow: parsing a technical document, cleaning the markdown, and extracting semantic tags and summaries before indexing into a vector store.

```yaml
store:
  type: lancedb
  table: knowledge_base
  path: ./lancedb_data
steps:
  - agent: document_parser
  - agent: markdown_cleaning
  - agent: enrichment
    config:
      model: "openai:gpt-5"
      prompt_template: "Summarize this content and extract key technical entities."
  - agent: chunking
    config:
      chunk_size: 512
  - agent: embedding
  - agent: indexing
```

```python
import asyncio
from pathlib import Path
from omni_ingest.core.pipeline import create_pipeline_from_config
from omni_ingest.core.model import IngestionContext
from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore


async def main():
    pipeline = create_pipeline_from_config(Path("pipeline.yaml"))
    mstore = SQLAlchemyMetadataStore("sqlite+aiosqlite:///./knowledge.db")
    ctx = IngestionContext(metadata={"file_path": "research_paper.pdf"}, store=mstore)
    results = await pipeline.run(ctx)
    print(f"Workflow complete. Processed {sum(len(e.items) for e in results)} items.")

if __name__ == "__main__":
    asyncio.run(main())
```
