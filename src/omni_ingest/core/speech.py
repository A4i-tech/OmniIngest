"""Speech-to-text and text-to-speech contracts and factory definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias

from .model import ByteContent, ResolvedResource

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext


Stt: TypeAlias = Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]
SttBuilder: TypeAlias = Callable[[], Stt]
SttFactory: TypeAlias = Callable[[str], Stt]


def simple_stt_factory(mapping: dict[str, SttBuilder]) -> SttFactory:
    """
    Build an STT factory that dispatches by some name to a fixed mapping.

    Args:
        mapping: Builders indexed by their name.

    Example:
        ```python
        from omni_ingest.core.speech import simple_stt_factory

        def llm_stt_builder() -> Stt:
            async def transcribe(wav_path):
                text = await whisper(wav_path)
                return text, {}
            return transcribe

        factory = simple_stt_factory({"llm": llm_stt_builder})
        ```
    """

    def factory(name: str) -> Stt:
        if name not in mapping:
            raise ValueError(f"Unknown STT builder: {name}")
        return mapping[name]()

    return factory


class SpeechFormat(StrEnum):
    MP3 = "mp3", "audio/mpeg"
    WAV = "wav", "audio/wav"

    mime_type: str

    def __new__(cls, value: str, mime_type: str):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.mime_type = mime_type
        return obj


@dataclass(kw_only=True)
class TTSBuilderParams:
    ctx: IngestionContext[ResolvedResource]
    voice: str | None = None


TTSModel: TypeAlias = Callable[[ByteContent, SpeechFormat], Awaitable[ByteContent]]
TTSBuilder: TypeAlias = Callable[[TTSBuilderParams], TTSModel]
TTSFactory: TypeAlias = Callable[[str, TTSBuilderParams], TTSModel]


def simple_tts_factory(mapping: dict[str, TTSBuilder]) -> TTSFactory:
    """
    Build a TTS factory that dispatches by model name to a fixed mapping.

    Args:
        mapping: Builders indexed by their name.
    """

    def factory(name: str, params: TTSBuilderParams) -> TTSModel:
        if name not in mapping:
            raise ValueError(f"Unknown TTS model: {name}")
        return mapping[name](params)

    return factory
