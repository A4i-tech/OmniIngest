import hashlib
from typing import Any, cast
from uuid import uuid4

import pytest

from omni_ingest.agent.deduplication import (
    DeterministicIdDeduplicationStrategy,
    NoDeduplicationStrategy,
    ShingleJaccardDeduplicationStrategy,
)
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import VectorStoreOutput
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.core.protocol import VectorStore


class RecordingVectorStore:
    def __init__(self):
        self.upsert_calls: list[list[KnowledgeItem]] = []

    async def upsert(self, items: list[KnowledgeItem], ctx: Any) -> None:
        del ctx
        self.upsert_calls.append(list(items))

    kind = "RecordingVectorStore"
    space = "default"


@pytest.mark.asyncio
async def test_vector_store_output_defaults_to_normalized_text_hash(metadata_store):
    tenant = await metadata_store.create_tenant("dedup-default", "Dedup Default")
    source_uri = "file://source-default.txt"

    existing_item_id = await metadata_store.create_knowledge_item(
        tenant_id=tenant.id,
        domain_profile="default",
        modality="text",
        source_uri=source_uri,
    )
    await metadata_store.create_chunk(
        chunk_id="existing-chunk",
        knowledge_item_id=existing_item_id,
        tenant_id=tenant.id,
        text="Hello world",
        vector_store_kind="memory",
        vector_store_space="test_space",
        vector_store_ref_id="existing-chunk",
    )

    new_item_id = await metadata_store.create_knowledge_item(
        tenant_id=tenant.id,
        domain_profile="default",
        modality="text",
        source_uri=source_uri,
    )

    vector_store = RecordingVectorStore()
    output = VectorStoreOutput(store=cast(VectorStore, vector_store))
    ctx = IngestionContext(
        tenant_id=tenant.id,
        items=[
            KnowledgeItem(raw_content=b"  HELLO   WORLD  ", metadata={"embedding": [0.1, 0.2, 0.3]}),
            KnowledgeItem(raw_content=b"Fresh content", metadata={"embedding": [0.4, 0.5, 0.6]}),
        ],
        store=metadata_store,
        metadata={"knowledge_item_id": new_item_id},
        resource=ResolvedResource(uri=source_uri, raw_content=b""),
    )

    result = await output.write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 1
    assert result.metadata["deduplicated_count"] == 1
    assert result.metadata["deduplication_strategy"] == "normalized_text_hash"
    assert len(vector_store.upsert_calls) == 1
    assert [await item.content(ctx) for item in vector_store.upsert_calls[0]] == [b"Fresh content"]

    chunks = await metadata_store.get_chunks_by_knowledge_item(new_item_id)
    assert len(chunks) == 1
    assert chunks[0].metadata_["dedup_normalized_text_hash"] == hashlib.sha256(b"fresh content").hexdigest()


@pytest.mark.asyncio
async def test_vector_store_output_no_deduplication_strategy_keeps_duplicates():
    vector_store = RecordingVectorStore()
    output = VectorStoreOutput(store=cast(VectorStore, vector_store), deduplication_strategy=NoDeduplicationStrategy())
    ctx = IngestionContext(
        tenant_id=uuid4(),
        items=[
            KnowledgeItem(raw_content=b"Same chunk", metadata={"embedding": [0.1]}),
            KnowledgeItem(raw_content=b"Same chunk", metadata={"embedding": [0.2]}),
        ],
        resource=ResolvedResource(uri="file://same.txt", raw_content=b""),
    )

    result = await output.write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 2
    assert result.metadata["deduplicated_count"] == 0
    assert result.metadata["deduplication_strategy"] == "none"
    assert len(vector_store.upsert_calls) == 1
    assert len(vector_store.upsert_calls[0]) == 2


@pytest.mark.asyncio
async def test_vector_store_output_deterministic_id_strategy_skips_existing_source_chunks(metadata_store):
    tenant = await metadata_store.create_tenant("dedup-deterministic", "Deterministic Dedup")
    source_uri = "file://stable-source.txt"
    strategy = DeterministicIdDeduplicationStrategy()

    initial_item_id = await metadata_store.create_knowledge_item(
        tenant_id=tenant.id,
        domain_profile="default",
        modality="text",
        source_uri=source_uri,
    )
    first_store = RecordingVectorStore()
    first_output = VectorStoreOutput(store=cast(VectorStore, first_store), deduplication_strategy=strategy)
    first_item = KnowledgeItem(raw_content=b"Stable chunk", metadata={"embedding": [0.1], "char_start": 0, "source_idx": 0})
    first_ctx = IngestionContext(
        tenant_id=tenant.id,
        items=[first_item],
        store=metadata_store,
        metadata={"knowledge_item_id": initial_item_id},
        resource=ResolvedResource(uri=source_uri, raw_content=b""),
    )

    first_result = await first_output.write(first_ctx)

    assert first_result.metadata["count"] == 1
    deterministic_id = str(first_item.id)

    replay_item_id = await metadata_store.create_knowledge_item(
        tenant_id=tenant.id,
        domain_profile="default",
        modality="text",
        source_uri=source_uri,
    )
    second_store = RecordingVectorStore()
    second_output = VectorStoreOutput(store=cast(VectorStore, second_store), deduplication_strategy=DeterministicIdDeduplicationStrategy())
    replay_item = KnowledgeItem(raw_content=b"Stable chunk", metadata={"embedding": [0.2], "char_start": 0, "source_idx": 0})
    replay_ctx = IngestionContext(
        tenant_id=tenant.id,
        items=[replay_item],
        store=metadata_store,
        metadata={"knowledge_item_id": replay_item_id},
        resource=ResolvedResource(uri=source_uri, raw_content=b""),
    )

    second_result = await second_output.write(replay_ctx)

    assert str(replay_item.id) == deterministic_id
    assert second_result.status == StepStatus.SUCCESS
    assert second_result.metadata["count"] == 0
    assert second_result.metadata["deduplicated_count"] == 1
    assert second_result.metadata["deduplication_strategy"] == "deterministic_id"
    assert second_store.upsert_calls == []
    assert await metadata_store.get_chunks_by_knowledge_item(replay_item_id) == []


@pytest.mark.asyncio
async def test_vector_store_output_shingle_jaccard_strategy_skips_near_duplicates():
    vector_store = RecordingVectorStore()
    output = VectorStoreOutput(
        store=cast(VectorStore, vector_store),
        deduplication_strategy=ShingleJaccardDeduplicationStrategy(threshold=0.8),
    )
    ctx = IngestionContext(
        tenant_id=uuid4(),
        items=[
            KnowledgeItem(raw_content=b"alpha beta gamma delta epsilon zeta", metadata={"embedding": [0.1]}),
            KnowledgeItem(raw_content=b"alpha beta gamma delta epsilon zeta eta", metadata={"embedding": [0.2]}),
            KnowledgeItem(raw_content=b"completely different content here", metadata={"embedding": [0.3]}),
        ],
        resource=ResolvedResource(uri="file://near-duplicates.txt", raw_content=b""),
    )

    result = await output.write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["count"] == 2
    assert result.metadata["deduplicated_count"] == 1
    assert result.metadata["deduplication_strategy"] == "shingle_jaccard"
    assert len(vector_store.upsert_calls) == 1
    assert [await item.content(ctx) for item in vector_store.upsert_calls[0]] == [
        b"alpha beta gamma delta epsilon zeta",
        b"completely different content here",
    ]
