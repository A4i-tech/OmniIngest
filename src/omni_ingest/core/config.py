"""Configuration management and environment variable loading for the framework."""

from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import AnyHttpUrl, BaseModel, DirectoryPath, Field, NewPath, PositiveInt, SecretStr
from pydantic_ai.embeddings import KnownEmbeddingModelName
from pydantic_ai.models import KnownModelName
from pydantic_settings import BaseSettings, SettingsConfigDict

KnownTranslationModelName: TypeAlias = Literal["azure"]


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


class SarvamOcrConfig(BaseModel):
    type: Literal["sarvam"] = "sarvam"
    api_key: SecretStr
    base_url: AnyHttpUrl = AnyHttpUrl("https://api.sarvam.ai")

    def build(self):
        from ..port.ocr import SarvamOcrEngine
        return SarvamOcrEngine(api_key=self.api_key.get_secret_value(), base_url=str(self.base_url))


class AzureCvOcrConfig(BaseModel):
    """Azure AI Vision (Computer Vision) Read OCR."""

    type: Literal["azure_cv"] = "azure_cv"
    endpoint: AnyHttpUrl
    api_key: SecretStr

    def build(self):
        from ..port.ocr import AzureCvOcrEngine
        return AzureCvOcrEngine(endpoint=str(self.endpoint), api_key=self.api_key.get_secret_value())


class AzureMlOcrConfig(BaseModel):
    """A custom OCR model deployed as an Azure ML online endpoint."""

    type: Literal["azure_ml"] = "azure_ml"
    endpoint: AnyHttpUrl
    api_key: SecretStr

    def build(self):
        from ..port.ocr import AzureMlOcrEngine
        return AzureMlOcrEngine(endpoint=str(self.endpoint), api_key=self.api_key.get_secret_value())


class LlamaCppOcrConfig(BaseModel):
    """A local vision model served by llama.cpp's OpenAI-compatible server."""

    type: Literal["llama_cpp"] = "llama_cpp"
    base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8080")
    model: str | None = None

    def build(self):
        from ..port.ocr import LlamaCppOcrEngine
        return LlamaCppOcrEngine(base_url=str(self.base_url), model=self.model)


OcrEngineConfig: TypeAlias = Annotated[
    SarvamOcrConfig | AzureCvOcrConfig | AzureMlOcrConfig | LlamaCppOcrConfig,
    Field(discriminator="type"),
]
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

    profiles_dir: Path = Path("profiles")
    """Directory containing reusable pipeline components"""

    vector_store: VectorStoreConfig = Field(default_factory=InMemoryVectorStoreConfig)
    """Vector store configuration selected entirely through environment configuration"""

    graph_store: GraphStoreConfig = Field(default_factory=InMemoryGraphStoreConfig)
    """Graph store configuration selected entirely through environment configuration"""

    ocr_engine: OcrEngineConfig | None = Field(default=None)
    """OCR engine used by the pdf_ocr step, selected entirely through environment configuration"""

    azure_speech_key: SecretStr | None = None
    """Azure Speech SDK subscription key"""

    azure_speech_region: str | None = None
    """Azure Speech SDK service region"""

    azure_translation_key: SecretStr | None = None
    """Azure Translator subscription key"""

    azure_translation_region: str | None = None
    """Azure Translator service region"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore")


settings = Settings()
