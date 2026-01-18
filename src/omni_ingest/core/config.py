"""Configuration management and environment variable loading for the framework."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings, loaded from environment variables or .env file."""

    database_url: str = Field("sqlite+aiosqlite:///./omni_ingest.db", description="SQLAlchemy connection string")
    """SQLAlchemy connection string"""

    azure_storage_connection_string: str | None = Field(None, description="Azure Blob Storage connection string")
    """Azure Blob Storage connection string"""

    openai_api_key: SecretStr | None = Field(None, description="API key for OpenAI services")
    """API key for OpenAI services"""

    openai_model: str = Field("text-embedding-3-small", description="Default model for embeddings")
    """Default model for embeddings"""

    azure_openai_api_key: SecretStr | None = Field(None, description="API key for Azure OpenAI")
    """API key for Azure OpenAI"""

    azure_openai_endpoint: str | None = Field(None, description="Endpoint for Azure OpenAI")
    """Endpoint for Azure OpenAI"""

    qdrant_url: str = Field("http://localhost:6333", description="URL for Qdrant vector database")
    """URL for Qdrant vector database"""

    qdrant_api_key: SecretStr | None = Field(None, description="API key for Qdrant")
    """API key for Qdrant"""

    lancedb_path: str = Field("./lancedb_data", description="Local path for LanceDB storage")
    """Local path for LanceDB storage"""

    azure_speech_key: SecretStr | None = Field(None, description="Azure Speech SDK subscription key")
    """Azure Speech SDK subscription key"""

    azure_speech_region: str | None = Field(None, description="Azure Speech SDK service region")
    """Azure Speech SDK service region"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()


class ProfileManager:
    """
    Manages loading and parsing of pipeline profile YAML files.

    Supports environment variable substitution in the format ${env.VAR_NAME}.
    """

    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)

    def _substitute_env_vars(self, content: str) -> str:
        pattern = re.compile(r"\$\{env\.(\w+)\}")

        def replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return pattern.sub(replace, content)

    def load_profile(self, profile_name: str) -> dict[str, Any]:
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile {profile_name} not found at {profile_path}")

        with open(profile_path) as f:
            raw_content = f.read()

        substituted_content = self._substitute_env_vars(raw_content)
        config = yaml.safe_load(substituted_content)
        if not isinstance(config, dict):
            raise ValueError(f"Profile {profile_name} invalid")
        return config
