from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omni_ingest.agent.governance import IngestionLoggerAgent
from omni_ingest.agent.indexing import VectorIndexingAgent
from omni_ingest.core.model import IngestionContext, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import PipelineRunner
from omni_ingest.port.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_tenant_crud(metadata_store):
    tenant = await metadata_store.create_tenant("test_tenant", "Test Tenant")

    assert tenant.slug == "test_tenant"
    assert tenant.display_name == "Test Tenant"

    retrieved = await metadata_store.get_tenant("test_tenant")
    assert retrieved is not None
    assert retrieved.id == tenant.id

    missing = await metadata_store.get_tenant("nonexistent")
    assert missing is None


@pytest.mark.asyncio
async def test_knowledge_item_crud(metadata_store):
    tenant = await metadata_store.create_tenant("ki_test_tenant", "KI Test Tenant")

    item_id = await metadata_store.create_knowledge_item(
        tenant_id=tenant.id,
        domain_profile="education_shiksha",
        modality="document",
        source_uri="blob://test/chapter1.pdf",
        title="Chapter 1: Introduction",
        description="First chapter of the textbook",
        metadata={"board": "CBSE", "grade": 6, "subject": "Science"},
    )

    assert item_id is not None

    item = await metadata_store.get_knowledge_item(item_id)
    assert item is not None
    assert item.content == b""
    assert item.source_uri == "blob://test/chapter1.pdf"
    assert item.metadata["grade"] == 6


@pytest.mark.asyncio
async def test_chunk_crud(metadata_store):
    tenant = await metadata_store.create_tenant("chunk_test_tenant", "Chunk Test Tenant")

    item_id = await metadata_store.create_knowledge_item(tenant_id=tenant.id, domain_profile="generic", modality="text", source_uri="file://test.txt")

    chunk1_id = await metadata_store.create_chunk(
        chunk_id="chunk_001",
        knowledge_item_id=item_id,
        tenant_id=tenant.id,
        text="This is the first chunk of text.",
        vector_store_kind="qdrant",
        vector_store_space="test_collection",
        vector_store_ref_id="qdrant_001",
        metadata={"page": 1, "char_start": 0},
    )

    chunk2_id = await metadata_store.create_chunk(
        chunk_id="chunk_002",
        knowledge_item_id=item_id,
        tenant_id=tenant.id,
        text="This is the second chunk of text.",
        vector_store_kind="qdrant",
        vector_store_space="test_collection",
        vector_store_ref_id="qdrant_002",
        metadata={"page": 1, "char_start": 100},
    )

    assert chunk1_id == "chunk_001"
    assert chunk2_id == "chunk_002"

    chunk = await metadata_store.get_chunk("chunk_001")
    assert chunk is not None
    assert chunk.text == "This is the first chunk of text."
    assert chunk.vector_store_kind == "qdrant"
    assert chunk.vector_store_space == "test_collection"

    chunks = await metadata_store.get_chunks_by_knowledge_item(item_id)
    assert len(chunks) == 2

    missing = await metadata_store.get_chunk("nonexistent")
    assert missing is None


@pytest.mark.asyncio
async def test_pipeline_run_tracking(metadata_store):
    tenant = await metadata_store.create_tenant("run_test_tenant", "Run Test Tenant")
    pipeline_version_id = uuid4()

    run_id = await metadata_store.create_pipeline_run(tenant_id=tenant.id, pipeline_version_id=pipeline_version_id, source_uri="s3://bucket/file.pdf")
    assert run_id is not None

    step1_id = await metadata_store.create_step_run(run_id, "chunking")
    step2_id = await metadata_store.create_step_run(run_id, "embedding")

    await metadata_store.update_step_run(step1_id, status="success", metadata={"count": 10, "chunk_size": 512})

    await metadata_store.update_step_run(step2_id, status="failure", error="OpenAI API rate limit exceeded")

    await metadata_store.update_pipeline_run(run_id, status="failed", completed_at=datetime.now(UTC), error="Pipeline failed at embedding step")


@pytest.mark.asyncio
async def test_db_registry_logging(metadata_store):
    ctx = IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", content=b""))
    runner = PipelineRunner(steps=[IngestionLoggerAgent()])
    results = await runner.run(ctx)
    assert results[0].status == StepStatus.SUCCESS
    async with metadata_store.session_factory() as s:
        from sqlalchemy import select

        from omni_ingest.port.metadata_store import PipelineStepRun

        res = await s.execute(select(PipelineStepRun).where(PipelineStepRun.pipeline_run_id == ctx.run_id))
        rows = res.scalars().all()
        assert len(rows) == 1
        assert rows[0].status == "success"


@pytest.mark.asyncio
async def test_pipeline_registration(metadata_store):
    pipeline_id = await metadata_store.upsert_pipeline("shiksha_v1", "education")
    version_id = await metadata_store.upsert_pipeline_version(pipeline_id, "1.0", "shiksha_v1.yaml", is_default=True)
    assert (await metadata_store.get_pipeline_version("shiksha_v1", "1.0")).id == version_id
    assert (await metadata_store.get_default_pipeline_version("shiksha_v1")).id == version_id
    await metadata_store.upsert_pipeline_version(pipeline_id, "1.1", "shiksha_v1_1.yaml", is_default=True)
    assert (await metadata_store.get_pipeline_version("shiksha_v1", "1.0")).is_default is False
    assert (await metadata_store.get_default_pipeline_version("shiksha_v1")).version == "1.1"


@pytest.mark.asyncio
async def test_indexing_agent_registers_chunks(metadata_store, tenant):
    ki_id = await metadata_store.create_knowledge_item(tenant_id=tenant.id, domain_profile="default", modality="text", source_uri="file://test.txt")
    vstore = InMemoryVectorStore()
    agent = VectorIndexingAgent(store=vstore)
    item = KnowledgeItem(content=b"Chunk 1", metadata={"embedding": [0.1] * 1536})
    ctx = IngestionContext(tenant_id=tenant.id, items=[item], store=metadata_store, metadata={"knowledge_item_id": ki_id}, resource=ResolvedResource(uri="", content=b""))
    res = await PipelineRunner([agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS, f"Step failed: {res[0].error}"
    chunks = await metadata_store.get_chunks_by_knowledge_item(ki_id)
    assert len(chunks) == 1
    assert chunks[0].text == "Chunk 1"
