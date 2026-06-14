import pytest


@pytest.fixture
async def tenant(metadata_store):
    return await metadata_store.create_tenant("test_tenant", "Test Tenant")
