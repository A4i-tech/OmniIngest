import uuid

import pytest

from omni_ingest.core.config import settings
from omni_ingest.port.connector import AzureBlobConnector, HttpUrlConnector


@pytest.mark.asyncio
async def test_http_connector_fetch_real():
    connector = HttpUrlConnector()
    data = await connector.fetch("https://httpbin.org/robots.txt")
    assert b"User-agent" in data


@pytest.mark.asyncio
async def test_azure_connector_list():
    assert settings.azure_storage_connection_string
    connector = AzureBlobConnector(settings.azure_storage_connection_string, "test-ingestion")

    resources = await connector.list_resources()
    assert isinstance(resources, list)

    await connector.close()


@pytest.mark.asyncio
async def test_azure_connector_fetch():
    assert settings.azure_storage_connection_string
    connector = AzureBlobConnector(settings.azure_storage_connection_string, "test-ingestion")

    from omni_ingest.port.object_store import AzureBlobObjectStore

    store = AzureBlobObjectStore(settings.azure_storage_connection_string, "test-ingestion")

    key = f"connector-test-{uuid.uuid4().hex[:8]}.txt"
    await store.put(b"connector test data", key)

    try:
        data = await connector.fetch(key)
        assert data == b"connector test data"
    finally:
        await store.delete(key)
        await store.close()
        await connector.close()
