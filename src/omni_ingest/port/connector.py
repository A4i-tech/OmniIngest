"""External data source connectors for fetching raw information into the pipeline."""

from typing import Any

import httpx
from azure.storage.blob.aio import BlobServiceClient

from ..core.protocol import SourceConnector, SourceResource


class AzureBlobConnector(SourceConnector):
    """
    Connector for listing and fetching files from Azure Blob Storage.
    """

    def __init__(self, connection_string: str, container_name: str):
        self.container = container_name
        self._client = BlobServiceClient.from_connection_string(connection_string)

    async def list_resources(self, filters: dict[str, Any] | None = None) -> list[SourceResource]:
        prefix = filters.get("prefix") if filters else None
        container = self._client.get_container_client(self.container)
        resources = []
        async for blob in container.list_blobs(name_starts_with=prefix):
            resources.append(SourceResource(id=blob.name, uri=f"azure://{self.container}/{blob.name}", metadata=dict(blob.metadata or {})))
        return resources

    async def fetch(self, blob_name: str) -> bytes:
        blob_client = self._client.get_blob_client(self.container, blob_name)
        stream = await blob_client.download_blob()
        return await stream.readall()

    async def close(self):
        await self._client.close()


class HttpUrlConnector(SourceConnector):
    """
    Connector for retrieving data from external HTTP(S) resources.
    """

    async def list_resources(self, filters: dict[str, Any] | None = None) -> list[SourceResource]:
        return []

    async def fetch(self, url: str) -> bytes:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.content

    async def close(self):
        pass
