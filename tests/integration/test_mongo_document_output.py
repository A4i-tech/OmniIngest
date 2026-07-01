"""Integration tests for MongoDocumentOutput against a real MongoDB instance."""

import pytest
from pymongo import AsyncMongoClient

from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import MongoDocumentOutput
from omni_ingest.core.pipeline import IngestionContext

_OUTPUT_URI = "mongodb://localhost:27017/test_omni_ingest/test_mongo_document_output"


@pytest.fixture
async def mongo_collection(metadata_store):
    async with AsyncMongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000) as client:
        await client.admin.command("ping")
        col = client["test_omni_ingest"]["test_mongo_document_output"]
        await col.drop()
        yield col
        await col.drop()


async def test_write_inserts_documents(mongo_collection, metadata_store):
    out = MongoDocumentOutput(_OUTPUT_URI)
    items = [
        KnowledgeItem(raw_content=b"Question 1", metadata={"subject": "math"}),
        KnowledgeItem(raw_content=b"Question 2", metadata={"subject": "science"}),
    ]
    ctx = IngestionContext(
        resource=ResolvedResource(uri="test://doc.pdf", raw_content=b""),
        store=metadata_store,
        items=items,
    )

    result = await out.write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 2
    assert await mongo_collection.count_documents({}) == 2


async def test_write_upsert_idempotent(mongo_collection, metadata_store):
    out = MongoDocumentOutput(_OUTPUT_URI)
    item = KnowledgeItem(raw_content=b"Same content", metadata={})
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", raw_content=b""),
        store=metadata_store,
        items=[item],
    )

    await out.write(ctx)
    await out.write(ctx)

    assert await mongo_collection.count_documents({}) == 1


async def test_write_stores_expected_fields(mongo_collection, metadata_store):
    out = MongoDocumentOutput(_OUTPUT_URI)
    item = KnowledgeItem(raw_content=b"hello world", metadata={"grade": 6})
    ctx = IngestionContext(
        resource=ResolvedResource(uri="file://test.pdf", raw_content=b""),
        store=metadata_store,
        items=[item],
    )

    await out.write(ctx)

    doc = await mongo_collection.find_one({"_id": str(item.id)})
    assert doc == {"_id": str(item.id), "source_uri": "", "content": "hello world", "metadata": {"grade": 6}}


async def test_write_skips_when_no_items(mongo_collection, metadata_store):
    out = MongoDocumentOutput(_OUTPUT_URI)
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", raw_content=b""),
        store=metadata_store,
    )

    result = await out.write(ctx)

    assert result.status == StepStatus.SKIPPED
    assert await mongo_collection.count_documents({}) == 0
