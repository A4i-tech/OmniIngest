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
    """Sarvam document-intelligence OCR, submitted as one job per page."""

    type: Literal["sarvam"] = "sarvam"
    api_key: SecretStr
    language: str = "hi-IN"
    timeout: PositiveInt = 300

    def build(self):
        from ..port.ocr import SarvamOcrEngine
        return SarvamOcrEngine(api_key=self.api_key.get_secret_value(), language=self.language, timeout=self.timeout)


class AzureDocIntelOcrConfig(BaseModel):
    """Azure Document Intelligence OCR (prebuilt-layout, Markdown output)."""

    type: Literal["azure_di"] = "azure_di"
    endpoint: AnyHttpUrl
    api_key: SecretStr
    model: str = "prebuilt-layout"
    api_version: str = "2024-11-30"
    timeout: PositiveInt = 300

    def build(self):
        from ..port.ocr import AzureDocIntelOcrEngine
        return AzureDocIntelOcrEngine(endpoint=str(self.endpoint), api_key=self.api_key.get_secret_value(), model=self.model, api_version=self.api_version, timeout=self.timeout)


class LLMOcrConfig(BaseModel):
    """OCR via a vision LLM, using the pipeline's configured chat model (AgentMixin).

    Any OpenAI-compatible vision model works here (Azure OpenAI, a llama.cpp
    server, etc.) — the model and its credentials come from OcrAgent's
    AgentMixin config / settings, so this config takes no connection args. The
    engine is built by OcrAgent rather than here, since agent vending needs the
    ingestion context.
    """

    type: Literal["llm"] = "llm"


class AzureMlOcrConfig(BaseModel):
    """A custom OCR container deployed to an Azure ML managed online endpoint (BYOC)."""

    type: Literal["azure_ml"] = "azure_ml"
    scoring_uri: AnyHttpUrl
    api_key: SecretStr
    deployment: str | None = None
    timeout: PositiveInt = 120

    def build(self):
        from ..port.ocr import AzureMlOcrEngine
        return AzureMlOcrEngine(scoring_uri=str(self.scoring_uri), api_key=self.api_key.get_secret_value(), deployment=self.deployment, timeout=self.timeout)


OcrEngineConfig: TypeAlias = Annotated[
    SarvamOcrConfig | AzureDocIntelOcrConfig | AzureMlOcrConfig | LLMOcrConfig,
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

    sarvam_api_key: SecretStr | None = None
    """Sarvam document-intelligence OCR subscription key"""

    azure_di_ocr_endpoint: AnyHttpUrl | None = None
    """Azure Document Intelligence OCR endpoint"""

    azure_di_ocr_key: SecretStr | None = None
    """Azure Document Intelligence OCR key"""

    azure_ml_ocr_scoring_uri: AnyHttpUrl | None = None
    """Azure ML custom-container OCR scoring URI"""

    azure_ml_ocr_key: SecretStr | None = None
    """Azure ML custom-container OCR endpoint key"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore")


settings = Settings()
