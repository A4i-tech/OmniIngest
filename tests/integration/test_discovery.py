from uuid import uuid4

import pytest

from omni_ingest.core.model import KnowledgeItem
from omni_ingest.core.pipeline import KnowledgeBaseAdapter
from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore
from omni_ingest.port.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_discovery_adapter_unified_search():
    mstore = SQLAlchemyMetadataStore("sqlite+aiosqlite:///:memory:")
    await mstore.init_db()
    vstore = InMemoryVectorStore()
    adapter = KnowledgeBaseAdapter(mstore, vstore)

    tenant_id = uuid4()
    item_id = await mstore.create_knowledge_item(
        tenant_id=tenant_id,
        domain_profile="discovery_test",
        modality="text",
        source_uri="test://123",
        title="Discovery Piece",
        metadata={"tag": "vital"},
    )

    items = await adapter.list_items(tenant_id=tenant_id, domain_profile="discovery_test")
    assert len(items) == 1
    assert items[0].title == "Discovery Piece"
    assert items[0].id == item_id

    ki = KnowledgeItem(id=item_id, tenant_id=tenant_id, content="Content", metadata={"embedding": [0.1] * 1536})
    await vstore.upsert([ki])

    results = await adapter.query(tenant_id=str(tenant_id), query=[0.1] * 1536)
    assert len(results) >= 1
    assert results[0].id == item_id


@pytest.mark.asyncio
async def test_discovery_adapter_isolation_and_filters():
    mstore = SQLAlchemyMetadataStore("sqlite+aiosqlite:///:memory:")
    await mstore.init_db()
    adapter = KnowledgeBaseAdapter(mstore, InMemoryVectorStore())

    t1, t2 = uuid4(), uuid4()
    await mstore.create_knowledge_item(
        tenant_id=t1, domain_profile="p1", modality="text", source_uri="s1", content="c1"
    )
    await mstore.create_knowledge_item(
        tenant_id=t2, domain_profile="p1", modality="audio", source_uri="s2", content="c2"
    )

    items_t1 = await adapter.list_items(tenant_id=t1)
    assert len(items_t1) == 1
    assert items_t1[0].source_uri == "s1"

    items_audio = await adapter.list_items(tenant_id=t2, modality="audio")
    assert len(items_audio) == 1
    assert items_audio[0].modality == "audio"

    items_text = await adapter.list_items(tenant_id=t2, modality="text")
    assert len(items_text) == 0
