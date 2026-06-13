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
This example demonstrates a complete workflow: parsing a technical document, cleaning the markdown, extracting semantic tags and summaries, and writing the final context to a vector store.

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

```python
import asyncio
from pathlib import Path
from omni_ingest.core.pipeline import create_pipeline_from_config
from omni_ingest.core.model import IngestionContext
from omni_ingest.core.output import VectorStoreOutput
from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore
from omni_ingest.port.vector_store import LanceDBVectorStore


async def main():
    pipeline = create_pipeline_from_config(Path("pipeline.yaml"))
    mstore = SQLAlchemyMetadataStore("sqlite+aiosqlite:///./knowledge.db")
    output = VectorStoreOutput(store=LanceDBVectorStore(table_name="knowledge_base", db_path="./lancedb_data"))
    ctx = IngestionContext(resource=Path("research_paper.pdf"), store=mstore)
    results = await pipeline.run(ctx, output=output)
    print(f"Processed {sum(len(e.items) for e in results)} items.")

if __name__ == "__main__":
    asyncio.run(main())
```
