"""Configuration management and environment variable loading for the framework."""

from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import AnyHttpUrl, BaseModel, DirectoryPath, Field, NewPath, PositiveInt, SecretStr
from pydantic_ai.embeddings import KnownEmbeddingModelName
from pydantic_ai.models import KnownModelName
from pydantic_settings import BaseSettings, SettingsConfigDict

KnownTranslationModelName: TypeAlias = Literal["azure", "openai", "sarvam"]
KnownSTTModelName: TypeAlias = Literal["azure", "openai", "sarvam"]
KnownSpeechModelName: TypeAlias = Literal["azure"]


class InMemoryVectorStoreConfig(BaseModel):
    type: Literal["in_memory"] = "in_memory"
    vector_size: PositiveInt = 1536

    def build(self):
        from ..port.vector_store import InMemoryVectorStore
        return InMemoryVectorStore(vector_size=self.vector_size)


class LanceDBVectorStoreConfig(BaseModel):
    type: Literal["lancedb"] = "lancedb"
    table_name: str = "default_vectors"
    path: DirectoryPath | NewPath = Path("./lancedb_data")
    vector_size: PositiveInt = 1536

    def build(self):
        from ..port.vector_store import LanceDBVectorStore
        return LanceDBVectorStore(table_name=self.table_name, db_path=str(self.path), vector_size=self.vector_size)


class QdrantVectorStoreConfig(BaseModel):
    type: Literal["qdrant"] = "qdrant"
    url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    api_key: SecretStr | None = None
    collection_name: str = "default"
    vector_size: PositiveInt = 1536

    def build(self):
        from ..port.vector_store import QdrantVectorStore
        return QdrantVectorStore(
            col=self.collection_name,
            url=str(self.url),
            key=self.api_key.get_secret_value() if self.api_key else None,
            size=self.vector_size,
        )


class MongoVectorStoreConfig(BaseModel):
    type: Literal["mongo"] = "mongo"
    uri: str
    database_name: str
    collection_name: str
    index_name: str = "vector_index"
    num_candidates: PositiveInt = 100

    def build(self):
        from ..port.vector_store import MongoVectorStore

        return MongoVectorStore(
            uri=self.uri,
            database_name=self.database_name,
            collection_name=self.collection_name,
            index_name=self.index_name,
            num_candidates=self.num_candidates,
        )


class AzureVectorSearchStoreConfig(BaseModel):
    type: Literal["azure_vector_search"] = "azure_vector_search"
    endpoint: AnyHttpUrl
    api_key: SecretStr
    index_name: str

    def build(self):
        from ..port.vector_store import AzureVectorSearchStore

        return AzureVectorSearchStore(
            endpoint=str(self.endpoint),
            api_key=self.api_key.get_secret_value(),
            index_name=self.index_name,
        )


class ChromaVectorStoreConfig(BaseModel):
    type: Literal["chroma"] = "chroma"
    collection_name: str = "default"
    path: DirectoryPath | NewPath = Path("./chroma_data")

    def build(self):
        from ..port.vector_store import ChromaVectorStore

        return ChromaVectorStore(collection_name=self.collection_name, path=str(self.path))


class InMemoryGraphStoreConfig(BaseModel):
    type: Literal["in_memory"] = "in_memory"

    def build(self):
        from ..port.graph_store import InMemoryGraphStore

        return InMemoryGraphStore()


class Neo4jGraphStoreConfig(BaseModel):
    type: Literal["neo4j"] = "neo4j"
    uri: str
    user: str
    password: SecretStr

    def build(self):
        from ..port.graph_store import Neo4jGraphStore
        return Neo4jGraphStore(uri=self.uri, user=self.user, password=self.password.get_secret_value())


VectorStoreConfig: TypeAlias = Annotated[
    InMemoryVectorStoreConfig | LanceDBVectorStoreConfig | QdrantVectorStoreConfig | MongoVectorStoreConfig | AzureVectorSearchStoreConfig | ChromaVectorStoreConfig,
    Field(discriminator="type"),
]
GraphStoreConfig: TypeAlias = Annotated[
    InMemoryGraphStoreConfig | Neo4jGraphStoreConfig,
    Field(discriminator="type"),
]


class Settings(BaseSettings):
    """Global application settings, loaded from environment variables or .env file."""

    default_chat_completion_model: KnownModelName | str = "openai:gpt-5-mini"
    """Default chat completion model to use during pipeline execution"""

    default_embedding_model: KnownEmbeddingModelName | str = "openai:text-embedding-3-small"
    """Default embedding completion model to use during pipeline execution"""

    default_translation_model: KnownTranslationModelName = "azure"
    """Default translation model to use during pipeline execution"""

    metadata_store: str = "sqlite+aiosqlite:///./omni_ingest.db"
    """SQLAlchemy connection string"""

    profiles_dir: Path = Path(__file__).resolve().parent.parent / "profiles"
    """Directory containing reusable pipeline components"""

    vector_store: VectorStoreConfig = Field(default_factory=InMemoryVectorStoreConfig)
    """Default vector store configuration"""

    graph_store: GraphStoreConfig = Field(default_factory=InMemoryGraphStoreConfig)
    """Default graph store configuration"""

    default_ocr_engine: str = "llm"
    """Default OCR engine used by the ocr step"""

    azure_speech_key: SecretStr | None = None
    """Azure Speech SDK subscription key"""

    azure_speech_region: str | None = None
    """Azure Speech SDK service region"""

    azure_translation_key: SecretStr | None = None
    """Azure Translator subscription key"""

    azure_translation_region: str | None = None
    """Azure Translator service region"""

    documentintelligence_endpoint: AnyHttpUrl | None = None
    """Azure Document Intelligence endpoint"""

    documentintelligence_api_key: SecretStr | None = None
    """Azure Document Intelligence API key"""

    mistral_ocr_api_key: SecretStr | None = None
    """API key for the Mistral Document AI endpoint"""

    mistral_ocr_model: str = "mistral-ocr-4-0"
    """Mistral Document AI model or deployment name"""

    default_stt_model: KnownSTTModelName = "azure"
    """Default speech-to-text model to use during pipeline execution"""

    openai_translation_model: str = "gpt-4o-mini"
    """Chat completion model used for OpenAI-compatible translation"""

    openai_translation_base_url: AnyHttpUrl | None = None
    """OpenAI-compatible endpoint for translation"""

    default_speech_model: KnownSpeechModelName = "azure"
    """Default speech synthesis model to use during pipeline execution"""

    stt_base_url: AnyHttpUrl | None = None
    """Base URL override for an OpenAI-compatible speech-to-text endpoint"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore")


settings = Settings()
