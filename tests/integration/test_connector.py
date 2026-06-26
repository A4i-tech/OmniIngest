import tempfile
from pathlib import Path

import pytest

from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.parser import read_resources


@pytest.mark.asyncio
async def test_https_fetch():
    resources = [r async for r in read_resources("https://httpbingo.org/robots.txt")]
    ctx = IngestionContext(resource=resources[0])
    assert len(resources) == 1
    assert "User-agent" in await resources[0].decode(ctx)


@pytest.mark.asyncio
async def test_azure_blob_storage_fetch():
    resources = [r async for r in read_resources("abfs://test-ingestion/when-tomorrow-was-yesterday.pdf")]
    ctx = IngestionContext(resource=resources[0])
    assert len(resources) == 1
    assert await resources[0].content_type(ctx) == "application/pdf"


@pytest.mark.asyncio
async def test_filesystem_fetch():
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        tmp.write(b"hello world")
        tmp.seek(0)
        resources = [r async for r in read_resources(Path(tmp.name).as_uri())]

    assert len(resources) == 1
    ctx = IngestionContext(resource=resources[0])
    assert await resources[0].content(ctx) == b"hello world"
