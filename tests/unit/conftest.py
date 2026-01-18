import asyncio
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-key-for-unit-testing-only")
os.environ.setdefault("AZURE_SPEECH_KEY", "dummy-speech-key")
os.environ.setdefault("AZURE_SPEECH_REGION", "eastus")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "dummy-azure-key")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com/")

import pytest
from omni_ingest.port.vector_store import InMemoryVectorStore


@pytest.fixture
def store():
    store = InMemoryVectorStore()
    yield store
    asyncio.run(store.close())


@pytest.fixture
def tenant():
    class MockTenant:
        id = "test-tenant-id"
        slug = "test-tenant"

    return MockTenant()
