"""Speech synthesis format, model contract, and factory definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from .model import ByteContent


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
class SpeechBuilderParams:
    voice: str | None = None


SpeechModel: TypeAlias = Callable[[ByteContent, SpeechFormat], Awaitable[ByteContent]]
SpeechBuilder: TypeAlias = Callable[[SpeechBuilderParams], SpeechModel]
SpeechFactory: TypeAlias = Callable[[str, SpeechBuilderParams], SpeechModel]


def simple(mapping: dict[str, SpeechBuilder]) -> SpeechFactory:
    """
    Build a factory that dispatches by model name to a fixed mapping.

    Args:
        mapping: Builders indexed by their name.
    """

    def factory(name: str, params: SpeechBuilderParams) -> SpeechModel:
        if name not in mapping:
            raise ValueError(f"Unknown speech model: {name}")
        return mapping[name](params)

    return factory


def default_factory() -> SpeechFactory:
    from ..port.tts import azure_speech_builder
    return simple({"azure": azure_speech_builder})
