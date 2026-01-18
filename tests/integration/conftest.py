import pytest

from omni_ingest.port.metadata_store import SQLAlchemyMetadataStore


@pytest.fixture
async def store():
    s = SQLAlchemyMetadataStore(database_url="sqlite+aiosqlite:///:memory:")
    await s.init_db()
    yield s
    await s.close()


@pytest.fixture
async def tenant(store):
    return await store.create_tenant("test_tenant", "Test Tenant")
