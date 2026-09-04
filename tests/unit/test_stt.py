import pytest

from omni_ingest.core.config import settings
from omni_ingest.port.stt import default_factory


def test_default_factory_azure_returns_callable():
    stt = default_factory()("azure")
    assert callable(stt)


def test_default_factory_openai_returns_callable():
    stt = default_factory()("openai")
    assert callable(stt)


def test_default_factory_sarvam_returns_callable():
    stt = default_factory()("sarvam")
    assert callable(stt)


def test_default_factory_azure_raises_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "azure_speech_key", None)
    with pytest.raises(ValueError, match="AZURE_SPEECH_KEY"):
        default_factory()("azure")


def test_default_factory_rejects_unknown_engine():
    with pytest.raises(ValueError, match="Unknown STT builder: bogus"):
        default_factory()("bogus")
