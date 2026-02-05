from pathlib import Path
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

    def __init__(self, base_path: Path = Path("./storage")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def put(self, data: bytes, key: str, metadata: dict[str, Any] | None = None) -> str:
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.resolve().as_uri()

    async def get(self, key: str) -> bytes:
        path = self.base_path / key
        return path.read_bytes()

    async def delete(self, key: str):
        path = self.base_path / key
        path.unlink(missing_ok=True)

    async def close(self):
        pass
