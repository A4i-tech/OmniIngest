import pytest
from pydantic import SecretStr

import omni_ingest.port.translator as translator
from omni_ingest.port.translator import AzureTranslator, OpenAITranslator, SarvamTranslator, create_translator


def test_create_translator_azure_returns_azure_translator(monkeypatch):
    monkeypatch.setattr(translator.settings, "azure_translation_key", SecretStr("dummy-translation-key"))
    monkeypatch.setattr(translator.settings, "azure_translation_region", "eastus")
    assert isinstance(create_translator("azure"), AzureTranslator)


def test_create_translator_openai_returns_openai_translator(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-key-for-unit-testing-only")
    assert isinstance(create_translator("openai"), OpenAITranslator)


def test_create_translator_sarvam_returns_sarvam_translator(monkeypatch):
    # AsyncSarvamAI defaults its api key from os.getenv("SARVAM_API_KEY") as an
    # eagerly-bound default argument, evaluated at import time, so monkeypatch.setenv
    # here would be too late -- stub the client class itself instead.
    monkeypatch.setattr(translator, "AsyncSarvamAI", lambda: None)
    assert isinstance(create_translator("sarvam"), SarvamTranslator)


def test_create_translator_raises_for_unknown_model():
    with pytest.raises(ValueError, match="unsupported"):
        create_translator("unknown")  # type: ignore[arg-type]


def test_create_translator_raises_when_azure_config_missing(monkeypatch):
    monkeypatch.setattr("omni_ingest.port.translator.settings.azure_translation_key", None)
    with pytest.raises(ValueError, match="AZURE_TRANSLATION_KEY"):
        create_translator("azure")
