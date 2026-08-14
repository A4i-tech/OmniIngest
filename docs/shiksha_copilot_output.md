# Output Architecture & Shiksha-Copilot Integration

## Overview & Architectural Principles

In `OmniIngest`, pipelines follow a strict separation of concerns:
* **Pipeline Responsibilities**: Ingestion, document parsing, chunking, domain enrichment, and vector embedding generation. Pipelines only generate and transform data.
* **Application Responsibilities (`shiksha-copilot`)**: Data persistence, vector store upserts (Qdrant, PgVector, Pinecone, etc.), database synchronization (MongoDB, Postgres), and custom downstream application logic.

Sink agents (e.g., `VectorStoreSinkAgent`) have been removed from `omni-ingest` core as output persistence belongs in the application layer (`shiksha-copilot`), not inside the pipeline transformation engine.

---

## How to Implement Custom Outputs in `shiksha-copilot`

### Pattern 1: Custom Output Class (Recommended for Python Applications)

`shiksha-copilot` can register custom output handlers using `omni-ingest`'s `Output` interface. This keeps pipeline execution clean while giving `shiksha-copilot` full control over how and where ingested items are stored.

```python
from pathlib import Path
from omni_ingest.core.factory import create_pipeline_from_config
from omni_ingest.core.model import StepStatus
from omni_ingest.core.output import Output, OutputResult
from omni_ingest.core.pipeline import IngestionContext


class ShikshaVectorStoreOutput(Output):
    """
    Custom output handler residing in shiksha-copilot.
    Handles vector store upserts and database updates.
    """

    def __init__(self, vector_client, collection_name: str):
        self.client = vector_client
        self.collection_name = collection_name

    async def write(self, ctx: IngestionContext) -> OutputResult:
        embedded_items = [
            item for item in ctx.items if item.metadata.get("embedding")
        ]
        if not embedded_items:
            return OutputResult(
                status=StepStatus.SKIPPED,
                metadata={"reason": "No items with embeddings found"},
            )

        # Custom vector store upsert logic in shiksha-copilot
        await self.client.upsert(
            collection=self.collection_name,
            points=[
                {
                    "id": str(item.id),
                    "vector": item.metadata["embedding"],
                    "payload": item.metadata,
                }
                for item in embedded_items
            ],
        )

        return OutputResult(
            status=StepStatus.SUCCESS,
            metadata={"written_count": len(embedded_items)},
        )


# Usage in shiksha-copilot
async def main():
    runner = create_pipeline_from_config(Path("school_notes.yaml"))
    ctx = IngestionContext(resource="lessons/soil-erosion.pdf")

    # Pass custom output handler to pipeline run
    await runner.run(ctx, output=ShikshaVectorStoreOutput(my_qdrant_client, "school_lessons"))
```

---

### Pattern 2: CLI Ingestion & Downstream Piping

For decoupled services, batch jobs, or multi-destination outputs, `omni-ingest` outputs structured JSON/artifacts which `shiksha-copilot` or external scripts can stream into target stores:

```bash
# 1. Run OmniIngest pipeline to extract, chunk, and embed data
omni-ingest mypipeline.yaml --input input.pdf --output out.json

# 2. Update Vector Store (e.g., Qdrant)
jq '.metadata.embeddings' out.json | curl -X PUT "http://localhost:6333/collections/my_collection/points?wait=true" \
  -H "Content-Type: application/json" -d @-

# 3. Update MongoDB
jq '.metadata.chapters' out.json | mongoimport --db proddb --collection chapters --jsonArray
```

---

## Migration Summary

* **Removed**: `VectorStoreSinkAgent` and `vector_store_sink` step registration in `omni-ingest`.
* **Action Required in `shiksha-copilot`**: Move vector store upsert logic into a custom `Output` implementation or a downstream runner step within `shiksha-copilot`.
