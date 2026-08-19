import pytest

from omni_ingest.core.config import settings
from omni_ingest.port.stt import (
    AzureSpeechToText,
    OpenAISpeechToText,
    SarvamSpeechToText,
    create_stt_client,
)


def test_create_stt_client_azure_returns_azure_client():
    client = create_stt_client("azure")
    assert isinstance(client, AzureSpeechToText)


def test_create_stt_client_openai_returns_openai_client():
    client = create_stt_client("openai")
    assert isinstance(client, OpenAISpeechToText)


def test_create_stt_client_sarvam_returns_sarvam_client():
    client = create_stt_client("sarvam")
    assert isinstance(client, SarvamSpeechToText)


def test_create_stt_client_azure_raises_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "azure_speech_key", None)
    with pytest.raises(ValueError, match="STT model is unsupported or not configured: azure"):
        create_stt_client("azure")


def test_create_stt_client_rejects_unknown_model():
    with pytest.raises(ValueError, match="STT model is unsupported or not configured: bogus"):
        create_stt_client("bogus")  # type: ignore[arg-type]
