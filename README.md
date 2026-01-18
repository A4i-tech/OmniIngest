# OmniIngest

[![License](https://img.shields.io/github/license/A4i-tech/OmniIngest)](https://github.com/A4i-tech/OmniIngest/blob/a4i/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
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
      model: gpt-4o
      prompt_template: "Summarize this content and extract key technical entities."
  - agent: chunking
    config:
      chunk_size: 512
  - agent: embedding
  - agent: indexing
```

```python
import asyncio
from omni_ingest.core.factory import create_pipeline_from_config
from omni_ingest.core.models import IngestionContext
from omni_ingest.core.pipeline import KnowledgeBaseAdapter
from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore
from omni_ingest.port.vector_store import LanceDBVectorStore


async def main():
    pipeline = create_pipeline_from_config("pipeline.yaml")
    ctx = IngestionContext(metadata={"file_path": "research_paper.pdf"})
    results = await pipeline.run(ctx)
    print(f"Workflow complete. Processed {len(results[-1].items)} items.")

    # Retrieval
    mstore = SQLAlchemyMetadataStore("sqlite+aiosqlite:///./knowledge.db")
    vstore = LanceDBVectorStore(table_name="knowledge_base", db_path="./lancedb_data")
    adapter = KnowledgeBaseAdapter(mstore, vstore)
    results = await adapter.query_text(query="What are the key findings?", k=5, tenant_id="default")
    for item in results:
        print(f"Found: {item.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```