from pathlib import Path

import pytest

from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore
from omni_ingest.port.vector_store import InMemoryVectorStore

RESOURCES_DIR = Path(__file__).parent / "resources"

@pytest.fixture
def resources_dir():
    return RESOURCES_DIR


@pytest.fixture
async def vector_store():
    async with InMemoryVectorStore() as store:
        yield store


@pytest.fixture
async def metadata_store():
    async with SQLAlchemyMetadataStore() as store:
        yield store
