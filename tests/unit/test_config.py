from pathlib import Path

import pytest
from pydantic import ValidationError

from omni_ingest.core.config import (
    AzureVectorSearchStoreConfig,
    ChromaVectorStoreConfig,
    LanceDBVectorStoreConfig,
    MongoVectorStoreConfig,
    Settings,
)


def test_settings_load_profiles_dir(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("PROFILES_DIR=/srv/omni-ingest/profiles")

    assert Settings(_env_file=env_file).profiles_dir == Path("/srv/omni-ingest/profiles")


def test_settings_load_nested_vector_store_config(tmp_path):
    env_file = tmp_path / ".env"
    db_path = tmp_path / "lancedb"
    env_file.write_text(
        "\n".join(
            (
                "VECTOR_STORE__TYPE=lancedb",
                "VECTOR_STORE__TABLE_NAME=env_vectors",
                f"VECTOR_STORE__PATH={db_path}",
                "VECTOR_STORE__VECTOR_SIZE=256",
            )
        )
    )

    settings = Settings(_env_file=env_file)

    assert isinstance(settings.vector_store, LanceDBVectorStoreConfig)
    assert settings.vector_store.table_name == "env_vectors"
    assert str(settings.vector_store.path) == str(db_path)
    assert settings.vector_store.vector_size == 256


@pytest.mark.parametrize(
    ("vector_store_config", "expected_type"),
    [
        (
            {
                "type": "mongo",
                "uri": "mongodb://localhost:27017",
                "database_name": "knowledge",
                "collection_name": "chunks",
            },
            MongoVectorStoreConfig,
        ),
        (
            {
                "type": "azure_vector_search",
                "endpoint": "https://example.search.windows.net",
                "api_key": "secret",
                "index_name": "knowledge-index",
            },
            AzureVectorSearchStoreConfig,
        ),
        (
            {
                "type": "chroma",
                "collection_name": "knowledge",
            },
            ChromaVectorStoreConfig,
        ),
    ],
)
def test_settings_select_additional_vector_backends(vector_store_config, expected_type):
    settings = Settings(vector_store=vector_store_config)

    assert isinstance(settings.vector_store, expected_type)


def test_settings_require_complete_neo4j_config():
    with pytest.raises(ValidationError):
        Settings(graph_store={"type": "neo4j"})
