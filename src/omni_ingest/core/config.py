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

    default_ocr_engine: str = "llm"
    """Default OCR engine used by the ocr step, selected entirely through environment configuration"""

    azure_speech_key: SecretStr | None = None
    """Azure Speech SDK subscription key"""

    azure_speech_region: str | None = None
    """Azure Speech SDK service region"""

    azure_translation_key: SecretStr | None = None
    """Azure Translator subscription key"""

    azure_translation_region: str | None = None
    """Azure Translator service region"""

    ocr_api_key: SecretStr | None = None
    """API key for the generic OpenAI-compatible OCR vision endpoint (`openai` engine); point this
    at any OpenAI-compatible vision/chat-completions server (Sarvam, an OpenAI-compatible Azure ML
    BYOC deployment, a self-hosted vLLM/Ollama server, etc.)"""

    ocr_base_url: AnyHttpUrl | None = None
    """Base URL for the generic OpenAI-compatible OCR vision endpoint (`openai` engine)"""

    ocr_model: str = "gpt-4o-mini"
    """Model name to request from the generic OpenAI-compatible OCR vision endpoint (`openai` engine)"""

    documentintelligence_endpoint: AnyHttpUrl | None = None
    """Azure Document Intelligence endpoint"""

    documentintelligence_api_key: SecretStr | None = None
    """Azure Document Intelligence API key"""

    azure_vm_ocr_subscription_id: str | None = None
    """Azure subscription id used to provision the ephemeral OCR VM"""

    azure_vm_ocr_resource_group: str | None = None
    """Existing Azure resource group the ephemeral OCR VM (and its NIC/public IP) are created into and deleted from"""

    azure_vm_ocr_location: str = "eastus"
    """Azure region for the ephemeral OCR VM"""

    azure_vm_ocr_vm_size: str = "Standard_D2s_v3"
    """Azure VM size for the ephemeral OCR VM"""

    azure_vm_ocr_container_image: str | None = None
    """Container image reference (e.g. a registry/image:tag) the ephemeral OCR VM pulls and runs via cloud-init/Docker to serve OCR requests"""

    azure_vm_ocr_local_build_path: str | None = None
    """Optional local path to a Docker build context. If set, OmniIngest will attempt to auto-build and push the container_image to the registry if it does not already exist remotely."""

    azure_vm_ocr_container_port: PositiveInt = 8000
    """Port the OCR container listens on inside the ephemeral VM"""

    azure_vm_ocr_container_registry_username: str | None = None
    """Optional registry username for `docker login` before pulling `azure_vm_ocr_container_image` (omit login if unset)"""

    azure_vm_ocr_container_registry_password: SecretStr | None = None
    """Optional registry password/token for `docker login`"""

    azure_vm_ocr_admin_username: str = "ocrsvc"
    """Admin username for the ephemeral OCR VM"""

    azure_vm_ocr_ssh_public_key: str | None = None
    """SSH public key installed on the ephemeral OCR VM (password auth is disabled; this is required to provision)"""

    azure_vm_ocr_gpu_enabled: bool = False
    """Whether to pass `--gpus all` to the docker run command in the ephemeral OCR VM (container mode only); requires a GPU-capable vm_size and a GPU-ready image (e.g. microsoft-dsvm/ubuntu-hpc)"""

    azure_vm_ocr_bootstrap_script: str | None = None
    """Raw cloud-init/shell bootstrap script for the ephemeral OCR VM that installs and starts an OCR server directly (no Docker); mutually exclusive with azure_vm_ocr_container_image — if set, this is embedded verbatim into cloud-init instead of the Docker flow; the script is responsible for serving POST /score on azure_vm_ocr_container_port"""

    azure_vm_ocr_image_publisher: str = "Canonical"
    """VM image publisher for the ephemeral OCR VM (override to e.g. "microsoft-dsvm" for GPU-ready images)"""

    azure_vm_ocr_image_offer: str = "0001-com-ubuntu-server-jammy"
    """VM image offer for the ephemeral OCR VM (override to e.g. "ubuntu-hpc" for GPU-ready images with NVIDIA driver + nvidia-container-toolkit preinstalled)"""

    azure_vm_ocr_image_sku: str = "22_04-lts-gen2"
    """VM image SKU for the ephemeral OCR VM"""

    azure_vm_ocr_image_version: str = "latest"
    """VM image version for the ephemeral OCR VM"""

    azure_vm_ocr_ready_timeout_s: PositiveInt = 900
    """Seconds to poll the ephemeral OCR VM for readiness before giving up (cold ACR pull of model-weight-baked images + CPU model load can take 10+ min; increase further for GPU/larger models)"""

    azure_vm_ocr_score_timeout_s: PositiveInt = 900
    """Per-request timeout (seconds) for the /score call to the ephemeral OCR VM. Readiness only confirms the HTTP server is up; the OCR model itself may still lazy-load on the first real request (GGUF download/mmap, CUDA init), so this must be generous — same order as ready_timeout_s."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore")


settings = Settings()
