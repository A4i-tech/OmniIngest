import pytest

from omni_ingest.port.translator import create_translator


def test_create_translator_raises_for_unknown_model():
    with pytest.raises(ValueError, match="unsupported"):
        create_translator("unknown")  # type: ignore[arg-type]


def test_create_translator_raises_when_azure_config_missing(monkeypatch):
    monkeypatch.setattr("omni_ingest.port.translator.settings.azure_translation_key", None)
    with pytest.raises(ValueError, match="AZURE_TRANSLATION_KEY"):
        create_translator("azure")
