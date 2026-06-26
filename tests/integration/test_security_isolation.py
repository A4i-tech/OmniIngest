from uuid import uuid4

import pytest

from omni_ingest.core.model import KnowledgeItem, ResolvedResource
from omni_ingest.core.pipeline import IngestionContext, KnowledgeBaseAdapter


@pytest.mark.asyncio
async def test_tenant_isolation_registry(metadata_store):
    t1 = await metadata_store.create_tenant("t1", "Tenant 1")
    t2 = await metadata_store.create_tenant("t2", "Tenant 2")

    await metadata_store.create_knowledge_item(tenant_id=t1.id, domain_profile="p1", modality="text", source_uri="s1", content="c1")

    items_t2 = await metadata_store.list_knowledge_items(tenant_id=t2.id)
    assert len(items_t2) == 0

    items_t1 = await metadata_store.list_knowledge_items(tenant_id=t1.id)
    assert len(items_t1) == 1


@pytest.mark.asyncio
async def test_tenant_isolation_vector_search(metadata_store, vector_store):
    adapter = KnowledgeBaseAdapter(metadata_store, vector_store)

    t1_id, t2_id = uuid4(), uuid4()

    ki1 = KnowledgeItem(id=uuid4(), tenant_id=t1_id, raw_content=b"T1 Content", metadata={"embedding": [0.1] * 1536})
    ki2 = KnowledgeItem(id=uuid4(), tenant_id=t2_id, raw_content=b"T2 Content", metadata={"embedding": [0.1] * 1536})

    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""))
    await vector_store.upsert([ki1, ki2], ctx)

    res1 = await adapter.query(tenant_id=t1_id, query=[0.1] * 1536, k=10)
    assert len(res1) == 1
    assert res1[0].tenant_id == t1_id

    res2 = await adapter.query(tenant_id=t2_id, query=[0.1] * 1536, k=10)
    assert len(res2) == 1
    assert res2[0].tenant_id == t2_id
