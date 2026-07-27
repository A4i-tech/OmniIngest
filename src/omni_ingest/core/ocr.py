"""OCR engine abstraction and factory wiring."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from .model import ResolvedResource

if TYPE_CHECKING:
    from .pipeline import IngestionContext


@runtime_checkable
class OcrEngine(Protocol):
    """Extract Markdown text from a single rendered page image."""

    async def __aenter__(self) -> OcrEngine: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def extract(self, image: bytes) -> str:
        """Return Markdown transcribed from a PNG-encoded page image."""
        ...


OcrEngineBuilder: TypeAlias = Callable[["IngestionContext[ResolvedResource]"], OcrEngine]
OcrFactory: TypeAlias = Callable[[str, "IngestionContext[ResolvedResource]"], OcrEngine]


def build_simple_factory(mapping: dict[str, OcrEngineBuilder]) -> OcrFactory:
    """Build an `OcrFactory` that dispatches by engine name to a fixed mapping."""

    def factory(engine: str, ctx: IngestionContext[ResolvedResource]) -> OcrEngine:
        if engine not in mapping:
            raise ValueError(f"Unknown OCR engine: {engine}")
        return mapping[engine](ctx)

    return factory
