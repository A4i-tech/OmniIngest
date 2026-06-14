import pytest

from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore
from omni_ingest.port.vector_store import InMemoryVectorStore


@pytest.fixture
async def vector_store():
    async with InMemoryVectorStore() as store:
        yield store


@pytest.fixture
async def metadata_store():
    async with SQLAlchemyMetadataStore.in_memory() as store:
        yield store
