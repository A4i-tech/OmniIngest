import os
from typing import Any

from azure.storage.blob.aio import BlobServiceClient

from ..core.protocol import ObjectStore


class AzureBlobObjectStore(ObjectStore):
    """
    Cloud-based object storage using Azure Blob Storage.
    """

    def __init__(self, connection_string: str, container_name: str):
        self.container_name = container_name
        self._client = BlobServiceClient.from_connection_string(connection_string)
        self._container = self._client.get_container_client(container_name)

    async def put(self, data: bytes, key: str, metadata: dict | None = None) -> str:
        await self._container.get_blob_client(key).upload_blob(data, overwrite=True, metadata=metadata)
        return f"azure://{self.container_name}/{key}"

    async def get(self, key: str) -> bytes:
        return await (await self._container.get_blob_client(key).download_blob()).readall()

    async def exists(self, key: str) -> bool:
        return await self._container.get_blob_client(key).exists()

    async def delete(self, key: str) -> None:
        await self._container.get_blob_client(key).delete_blob()

    async def close(self):
        await self._client.close()


class LocalObjectStore(ObjectStore):
    """
    File-system based object storage for local development.
    """

    def __init__(self, base_path: str = "./storage"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    async def put(self, data: bytes, key: str, metadata: dict[str, Any] | None = None) -> str:
        path = os.path.join(self.base_path, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return f"file://{os.path.abspath(path)}"

    async def get(self, key: str) -> bytes:
        path = os.path.join(self.base_path, key)
        with open(path, "rb") as f:
            return f.read()

    async def delete(self, key: str):
        path = os.path.join(self.base_path, key)
        if os.path.exists(path):
            os.remove(path)

    async def close(self):
        pass
