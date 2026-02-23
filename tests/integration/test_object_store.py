import os
import uuid

import pytest

from omni_ingest.port.object_store import AzureBlobObjectStore

CONTAINER_NAME = "test-ingestion"


@pytest.fixture
async def azure_store():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        pytest.skip("AZURE_STORAGE_CONNECTION_STRING not set")

    store = AzureBlobObjectStore(connection_string, CONTAINER_NAME)

    try:
        await store._container.create_container()
    except Exception:
        pass

    yield store
    await store.close()


@pytest.mark.asyncio
async def test_azure_object_store_put_get_delete(azure_store):
    key = f"test-{uuid.uuid4().hex[:8]}.txt"
    data = b"integration test data"

    url = await azure_store.put(data, key)
    assert "azure://" in url
    assert key in url

    retrieved = await azure_store.get(key)
    assert retrieved == data

    assert await azure_store.exists(key) is True

    await azure_store.delete(key)
    assert await azure_store.exists(key) is False


@pytest.mark.asyncio
async def test_azure_object_store_metadata(azure_store):
    key = f"meta-{uuid.uuid4().hex[:8]}.json"
    data = b'{"test": true}'

    await azure_store.put(data, key, metadata={"content_type": "application/json"})
    assert await azure_store.exists(key) is True

    await azure_store.delete(key)
