import asyncio

import pytest

from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore
from omni_ingest.port.vector_store import InMemoryVectorStore


@pytest.fixture
def store():
    store = InMemoryVectorStore()
    yield store
    asyncio.run(store.close())


@pytest.fixture
def metadata_store():
    s = SQLAlchemyMetadataStore.in_memory()
    yield s
    asyncio.run(s.close())
