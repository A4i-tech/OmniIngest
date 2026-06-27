"""Content resolver implementations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, urlparse

from ..core.content import resource_page
from ..core.model import ResolvedResource

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext
    from ..core.protocol import ContentResolver


class CompositeContentResolver:
    def __init__(self, resolvers: list[Callable[[ParseResult, IngestionContext[ResolvedResource]], Awaitable[bytes | None]]]):
        self.resolvers = resolvers

    async def resolve(self, uri: str, ctx: IngestionContext[ResolvedResource]) -> bytes:
        parsed = urlparse(uri)
        for resolver in self.resolvers:
            if result := await resolver(parsed, ctx):
                return result
        raise ValueError(f"Unsupported content URI: {uri}")


def default_content_resolver() -> ContentResolver:
    return CompositeContentResolver(resolvers=[resource_page])
