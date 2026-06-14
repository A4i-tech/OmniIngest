from uuid import uuid4

import pytest

from omni_ingest.core.model import KnowledgeItem
from omni_ingest.core.pipeline import KnowledgeBaseAdapter
from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore
from omni_ingest.port.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_tenant_isolation_registry():
    store = SQLAlchemyMetadataStore.in_memory()
    t1 = await store.create_tenant("t1", "Tenant 1")
    t2 = await store.create_tenant("t2", "Tenant 2")

    await store.create_knowledge_item(tenant_id=t1.id, domain_profile="p1", modality="text", source_uri="s1", content="c1")

    items_t2 = await store.list_knowledge_items(tenant_id=t2.id)
    assert len(items_t2) == 0

    items_t1 = await store.list_knowledge_items(tenant_id=t1.id)
    assert len(items_t1) == 1


@pytest.mark.asyncio
async def test_tenant_isolation_vector_search():
    vstore = InMemoryVectorStore()
    adapter = KnowledgeBaseAdapter(SQLAlchemyMetadataStore.in_memory(), vstore)

    t1_id, t2_id = uuid4(), uuid4()

    ki1 = KnowledgeItem(id=uuid4(), tenant_id=t1_id, content=b"T1 Content", metadata={"embedding": [0.1] * 1536})
    ki2 = KnowledgeItem(id=uuid4(), tenant_id=t2_id, content=b"T2 Content", metadata={"embedding": [0.1] * 1536})

    await vstore.upsert([ki1, ki2])

    res1 = await adapter.query(tenant_id=t1_id, query=[0.1] * 1536, k=10)
    assert len(res1) == 1
    assert res1[0].tenant_id == t1_id

    res2 = await adapter.query(tenant_id=t2_id, query=[0.1] * 1536, k=10)
    assert len(res2) == 1
    assert res2[0].tenant_id == t2_id
