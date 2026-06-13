import tempfile
from pathlib import Path

import pytest

from omni_ingest.parser import read_resources


@pytest.mark.asyncio
async def test_https_fetch():
    resources = [r async for r in read_resources("https://httpbin.org/robots.txt")]
    assert len(resources) == 1
    assert "User-agent" in resources[0].decode()


@pytest.mark.asyncio
async def test_azure_blob_storage_fetch():
    resources = [r async for r in read_resources("abfs://test-ingestion/when-tomorrow-was-yesterday.pdf")]
    assert len(resources) == 1
    assert resources[0].content_type == "application/pdf"


@pytest.mark.asyncio
async def test_filesystem_fetch():
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        tmp.write(b"hello world")
        tmp.seek(0)
        resources = [r async for r in read_resources(Path(tmp.name).as_uri())]

    assert len(resources) == 1
    assert resources[0].content == b"hello world"
