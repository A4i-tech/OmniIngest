from uuid import uuid4

import pytest

from omni_ingest.core.model import KnowledgeItem
from omni_ingest.core.pipeline import KnowledgeBaseAdapter


@pytest.mark.asyncio
async def test_discovery_adapter_unified_search(metadata_store, vector_store):
    adapter = KnowledgeBaseAdapter(metadata_store, vector_store)

    tenant_id = uuid4()
    item_id = await metadata_store.create_knowledge_item(
        tenant_id=tenant_id,
        domain_profile="discovery_test",
        modality="text",
        source_uri="test://123",
        title="Discovery Piece",
        metadata={"tag": "vital"},
    )

    items = await adapter.list_items(tenant_id=tenant_id, domain_profile="discovery_test")
    assert len(items) == 1
    assert items[0].id == item_id

    ki = KnowledgeItem(id=item_id, tenant_id=tenant_id, content=b"Content", metadata={"embedding": [0.1] * 1536})
    await vector_store.upsert([ki])

    results = await adapter.query(tenant_id=tenant_id, query=[0.1] * 1536)
    assert len(results) >= 1
    assert results[0].id == item_id


@pytest.mark.asyncio
async def test_discovery_adapter_isolation_and_filters(metadata_store, vector_store):
    adapter = KnowledgeBaseAdapter(metadata_store, vector_store)

    t1, t2 = uuid4(), uuid4()
    await metadata_store.create_knowledge_item(tenant_id=t1, domain_profile="p1", modality="text", source_uri="s1", content="c1")
    await metadata_store.create_knowledge_item(tenant_id=t2, domain_profile="p1", modality="audio", source_uri="s2", content="c2")

    items_t1 = await adapter.list_items(tenant_id=t1)
    assert len(items_t1) == 1
    assert items_t1[0].source_uri == "s1"

    items_audio = await adapter.list_items(tenant_id=t2, modality="audio")
    assert len(items_audio) == 1

    items_text = await adapter.list_items(tenant_id=t2, modality="text")
    assert len(items_text) == 0
