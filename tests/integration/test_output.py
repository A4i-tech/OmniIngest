import pytest
from pymongo import AsyncMongoClient

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import BrfFileOutput, MongoDocumentOutput
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner

_MONGO_OUTPUT_URI = "mongodb://localhost:27017/test_omni_ingest/test_mongo_document_output"


@pytest.fixture
async def mongo_collection():
    async with AsyncMongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000) as client:
        await client.admin.command("ping")
        collection = client["test_omni_ingest"]["test_mongo_document_output"]
        await collection.drop()
        yield collection
        await collection.drop()


async def test_mongo_output_inserts_documents(mongo_collection, metadata_store):
    output = MongoDocumentOutput(_MONGO_OUTPUT_URI)
    items = [
        KnowledgeItem(content=ByteContent.from_bytes(b"Question 1"), metadata={"subject": "math"}),
        KnowledgeItem(content=ByteContent.from_bytes(b"Question 2"), metadata={"subject": "science"}),
    ]
    ctx = IngestionContext(resource=ResolvedResource(uri="test://doc.pdf", content=ByteContent.from_bytes(b"")), store=metadata_store, items=items)

    result = await output.write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 2
    assert await mongo_collection.count_documents({}) == 2


async def test_mongo_output_upsert_is_idempotent(mongo_collection, metadata_store):
    output = MongoDocumentOutput(_MONGO_OUTPUT_URI)
    item = KnowledgeItem(content=ByteContent.from_bytes(b"Same content"))
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), store=metadata_store, items=[item])

    await output.write(ctx)
    await output.write(ctx)

    assert await mongo_collection.count_documents({}) == 1


async def test_mongo_output_stores_expected_fields(mongo_collection, metadata_store):
    output = MongoDocumentOutput(_MONGO_OUTPUT_URI)
    item = KnowledgeItem(content=ByteContent.from_bytes(b"hello world"), metadata={"grade": 6})
    ctx = IngestionContext(resource=ResolvedResource(uri="file://test.pdf", content=ByteContent.from_bytes(b"")), store=metadata_store, items=[item])

    await output.write(ctx)

    document = await mongo_collection.find_one({"_id": str(item.id)})
    assert document == {"_id": str(item.id), "source_uri": None, "content": "hello world", "metadata": {"grade": 6}}


async def test_mongo_output_skips_when_no_items(mongo_collection, metadata_store):
    output = MongoDocumentOutput(_MONGO_OUTPUT_URI)
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), store=metadata_store)

    result = await output.write(ctx)

    assert result.status == StepStatus.SKIPPED
    assert await mongo_collection.count_documents({}) == 0


@pytest.mark.asyncio
async def test_brf_output_round_trips_through_pipeline(metadata_store, tmp_path, resources_dir):
    try:
        import louis
    except (ImportError, OSError) as e:
        pytest.skip(f"liblouis unavailable: {e}")

    ctx = IngestionContext(store=metadata_store, resource=resources_dir / "blueberries-are-now-green.brf")
    results = await PipelineRunner([TextAgent()]).run(ctx)
    assert results[-1].status == StepStatus.SUCCESS

    out_path = tmp_path / "output.brf"
    output = BrfFileOutput(str(out_path))
    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS

    # Raw file on disk should hold braille cells, not literal print text.
    raw = out_path.read_text()
    assert "blueberries" not in raw

    text = louis.backTranslateString(list(output.tables), raw)
    assert "unexpected transformation has challenged" in text
