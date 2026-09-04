"""STT, builder, and factory definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

Stt: TypeAlias = Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]
SttBuilder: TypeAlias = Callable[[], Stt]
SttFactory: TypeAlias = Callable[[str], Stt]


def simple(mapping: dict[str, SttBuilder]) -> SttFactory:
    """
    Build a factory that dispatches by some name to a fixed mapping.

    Args:
        mapping: Builders indexed by their name.

    Example:
        ```python
        from omni_ingest.core.stt import simple

        def llm_stt_builder() -> Stt:
            async def transcribe(wav_path):
                text = await whisper(wav_path)
                return text, {}
            return transcribe

        factory = simple({"llm": llm_stt_builder})
        ```
    """

    def factory(name: str) -> Stt:
        if name not in mapping:
            raise ValueError(f"Unknown STT builder: {name}")
        return mapping[name]()

    return factory
