import pickle
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import update

from omni_ingest.agent.governance import IngestionLoggerAgent
from omni_ingest.core.model import KnowledgeItem, PipelineCheckpoint, PipelineCheckpointRecord, ResolvedResource, StepStatus
from omni_ingest.core.output import VectorStoreOutput
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner
from omni_ingest.port.metadata_store import NullMetadataStore


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
    assert item.raw_content == b""
    assert item.source_uri == "blob://test/chapter1.pdf"
    assert item.metadata["grade"] == 6


@pytest.mark.asyncio
async def test_null_metadata_store_discards_knowledge_item_content():
    async with NullMetadataStore() as store:
        tenant = await store.create_tenant("null_ki_test_tenant", "Null KI Test Tenant")

        item_id = await store.create_knowledge_item(
            tenant_id=tenant.id,
            domain_profile="education_shiksha",
            modality="document",
            source_uri="blob://test/chapter1.pdf",
            content=b"content that should not be stored",
            metadata={"board": "CBSE", "grade": 6, "subject": "Science"},
        )

        item = await store.get_knowledge_item(item_id)
        items = await store.list_knowledge_items(tenant.id)

    assert item is not None
    assert item.raw_content == b""
    assert item.metadata["grade"] == 6
    assert len(items) == 1
    assert items[0].raw_content == b""
    assert items[0].metadata["subject"] == "Science"


@pytest.mark.asyncio
async def test_null_metadata_store_ignores_pipeline_checkpoints():
    run_id = uuid4()
    checkpoint = PipelineCheckpoint(
        stage=1,
        resource=ResolvedResource(uri="file://large.pdf", raw_content=b"%PDF-binary"),
        tenant_id=uuid4(),
        domain_profile="test",
        items=[KnowledgeItem(raw_content=b"generated content", metadata={"page": 1})],
    )

    async with NullMetadataStore() as store:
        await store.create_pipeline_run(checkpoint.tenant_id, uuid4(), checkpoint.resource.uri, id=run_id)
        await store.set_pipeline_checkpoint(run_id, checkpoint)
        restored = await store.get_pipeline_checkpoint(run_id)

    assert restored is None


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
async def test_pipeline_checkpoint_crud(metadata_store):
    run_id = await metadata_store.create_pipeline_run(uuid4(), uuid4(), "file://input.pdf")
    checkpoint = PipelineCheckpoint(stage=1, resource=ResolvedResource(uri="file://input.pdf", raw_content=b"%PDF-binary"), tenant_id=uuid4(), domain_profile="test", metadata={"stage": 1}, items=[KnowledgeItem(raw_content=b"\x89PNG-binary")])

    assert await metadata_store.get_pipeline_checkpoint(run_id) is None
    await metadata_store.set_pipeline_checkpoint(run_id, checkpoint)
    restored = await metadata_store.get_pipeline_checkpoint(run_id)
    assert restored == checkpoint

    checkpoint.metadata["stage"] = 2
    await metadata_store.set_pipeline_checkpoint(run_id, checkpoint)
    assert (await metadata_store.get_pipeline_checkpoint(run_id)).metadata["stage"] == 2

    await metadata_store.set_pipeline_checkpoint(run_id, None)
    assert await metadata_store.get_pipeline_checkpoint(run_id) is None


@pytest.mark.asyncio
async def test_invalid_pipeline_checkpoint_raises(metadata_store):
    run_id = await metadata_store.create_pipeline_run(uuid4(), uuid4(), "file://input.pdf")
    await metadata_store.set_pipeline_checkpoint(run_id, PipelineCheckpoint(stage=0, resource=ResolvedResource(uri="file://input.pdf", raw_content=b""), tenant_id=uuid4(), domain_profile="test"))
    async with metadata_store.session_factory() as s:
        await s.execute(update(PipelineCheckpointRecord).where(PipelineCheckpointRecord.pipeline_run_id == run_id).values(checkpoint=b"invalid pickle"))
        await s.commit()
    with pytest.raises(pickle.UnpicklingError):
        await metadata_store.get_pipeline_checkpoint(run_id)

    async with metadata_store.session_factory() as s:
        await s.execute(update(PipelineCheckpointRecord).where(PipelineCheckpointRecord.pipeline_run_id == run_id).values(checkpoint=pickle.dumps("not a checkpoint")))
        await s.commit()
    with pytest.raises(TypeError, match="Expected PipelineCheckpoint, got str"):
        await metadata_store.get_pipeline_checkpoint(run_id)


@pytest.mark.asyncio
async def test_db_registry_logging(metadata_store):
    ctx = IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", raw_content=b""))
    runner = PipelineRunner(steps=[IngestionLoggerAgent()])
    results = await runner.run(ctx)
    assert results[0].status == StepStatus.SUCCESS
    async with metadata_store.session_factory() as s:
        from sqlalchemy import select

        from omni_ingest.core.model import PipelineStepRunRecord

        res = await s.execute(select(PipelineStepRunRecord).where(PipelineStepRunRecord.pipeline_run_id == ctx.run_id))
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
async def test_vector_store_output_registers_items(metadata_store, vector_store, tenant):
    ki_id = await metadata_store.create_knowledge_item(tenant_id=tenant.id, domain_profile="default", modality="text", source_uri="file://test.txt")
    output = VectorStoreOutput(store=vector_store)
    item = KnowledgeItem(raw_content=b"Chunk 1", metadata={"embedding": [0.1] * 1536})
    ctx = IngestionContext(tenant_id=tenant.id, items=[item], store=metadata_store, metadata={"knowledge_item_id": ki_id}, resource=ResolvedResource(uri="", raw_content=b""))
    res = await output.write(ctx)
    assert res.status == StepStatus.SUCCESS, f"Output failed: {res.error}"
    indexed_items = [
        item
        for item in await metadata_store.list_knowledge_items(tenant.id, source_uri="")
        if item.metadata.get("kind") == "indexed" and item.metadata.get("parent_knowledge_item_id") == str(ki_id)
    ]
    assert len(indexed_items) == 1
    assert indexed_items[0].raw_content == b"Chunk 1"
    assert indexed_items[0].metadata["vector_store"]["kind"] == vector_store.kind
