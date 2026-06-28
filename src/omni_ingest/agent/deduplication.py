"""Deduplication strategies for vector indexing."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from uuid import NAMESPACE_URL, uuid5

import jq
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import JqExpression, KnowledgeItem, ResolvedResource
from ..core.protocol import MetadataStore

type DeduplicationScope = Literal["source", "tenant"]

_NORMALIZED_TEXT_HASH_KEY = "dedup_normalized_text_hash"
_DETERMINISTIC_ID_KEY = "dedup_deterministic_id"
_SHINGLE_SIZE_KEY = "dedup_shingle_size"
_WHITESPACE_RE = re.compile(r"\s+")


class DeduplicationResult(BaseModel):
    """Outcome of applying deduplication to a batch of items."""

    kept_items: list[KnowledgeItem]
    input_count: int
    dropped_count: int
    strategy: str
    scope: DeduplicationScope | None = None


class DeduplicationStrategy(BaseModel, ABC):
    """Base class for item deduplication strategies."""

    model_config = ConfigDict(extra="forbid")
    kind: str

    @abstractmethod
    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        """Returns the subset of items that should be indexed."""


class ScopedDeduplicationStrategy(DeduplicationStrategy, ABC):
    """Deduplication strategy that can operate within a tenant or a source."""

    scope: DeduplicationScope = Field(default="source", description="Scope used to compare an item against previously indexed items")
    existing: JqExpression = Field(default='.items[] | select(.metadata.vector_store != null)', description="jq expression selecting stored items for comparison")

    async def _existing_items(self, ctx: IngestionContext[ResolvedResource]) -> list[KnowledgeItem]:
        store = ctx.store
        if not isinstance(store, MetadataStore):
            return []

        if self.scope == "tenant":
            items = await store.list_knowledge_items(ctx.tenant_id)
        else:
            items = await store.list_knowledge_items(tenant_id=ctx.tenant_id, source_uri=ctx.resource.uri)
        item_by_id = {str(item.id): item for item in items}
        selected = jq.compile(self.existing).input_value(await _item_document(items, ctx)).all()
        return [item_by_id[item["id"]] for item in selected]


class NoDeduplicationStrategy(DeduplicationStrategy):
    """Leaves the batch unchanged."""

    kind: Literal["none"] = "none"

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        del ctx
        return DeduplicationResult(kept_items=items, input_count=len(items), dropped_count=0, strategy=self.kind)


class DeterministicIdDeduplicationStrategy(ScopedDeduplicationStrategy):
    """Assigns stable IDs to items and skips IDs that already exist."""

    kind: Literal["deterministic_id"] = "deterministic_id"
    identity: JqExpression = Field(default="[.tenant_id, .source_uri, .text]", description="jq expression producing stable item identity")

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        seen_ids = {v for item in await self._existing_items(ctx) if isinstance(v := item.metadata.get(_DETERMINISTIC_ID_KEY), str)}
        kept_items: list[KnowledgeItem] = []

        for item in items:
            identity = json.dumps(jq.compile(self.identity).input_value((await _item_document([item], ctx))["items"][0]).first(), sort_keys=True, separators=(",", ":"))
            deterministic_id = str(uuid5(NAMESPACE_URL, identity))
            item.id = uuid5(NAMESPACE_URL, identity)
            item.metadata[_DETERMINISTIC_ID_KEY] = deterministic_id
            if deterministic_id in seen_ids:
                continue
            kept_items.append(item)
            seen_ids.add(deterministic_id)

        return DeduplicationResult(
            kept_items=kept_items,
            input_count=len(items),
            dropped_count=len(items) - len(kept_items),
            strategy=self.kind,
            scope=self.scope,
        )


class NormalizedTextHashDeduplicationStrategy(ScopedDeduplicationStrategy):
    """Deduplicates text items using a hash of normalized text."""

    kind: Literal["normalized_text_hash"] = "normalized_text_hash"

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        seen_hashes = {
            _normalized_text_hash(await item.decode(ctx), cast(str | None, item.metadata.get(_NORMALIZED_TEXT_HASH_KEY)))
            for item in await self._existing_items(ctx)
        }
        kept_items: list[KnowledgeItem] = []

        for item in items:
            normalized_hash = _normalized_text_hash(await item.decode(ctx))
            item.metadata[_NORMALIZED_TEXT_HASH_KEY] = normalized_hash
            if normalized_hash in seen_hashes:
                continue
            kept_items.append(item)
            seen_hashes.add(normalized_hash)

        return DeduplicationResult(
            kept_items=kept_items,
            input_count=len(items),
            dropped_count=len(items) - len(kept_items),
            strategy=self.kind,
            scope=self.scope,
        )


class ShingleJaccardDeduplicationStrategy(ScopedDeduplicationStrategy):
    """Deduplicates text items using Jaccard similarity over token shingles."""

    kind: Literal["shingle_jaccard"] = "shingle_jaccard"
    threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Minimum Jaccard similarity that counts as a duplicate")
    shingle_size: PositiveInt = Field(default=3, description="Token shingle size used for near-duplicate detection")

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        seen_shingles = [_shingles(await item.decode(ctx), self.shingle_size) for item in await self._existing_items(ctx)]
        kept_items: list[KnowledgeItem] = []

        for item in items:
            item_shingles = _shingles(await item.decode(ctx), self.shingle_size)
            item.metadata[_SHINGLE_SIZE_KEY] = self.shingle_size
            if any(_jaccard_similarity(item_shingles, existing) >= self.threshold for existing in seen_shingles):
                continue
            kept_items.append(item)
            seen_shingles.append(item_shingles)

        return DeduplicationResult(
            kept_items=kept_items,
            input_count=len(items),
            dropped_count=len(items) - len(kept_items),
            strategy=self.kind,
            scope=self.scope,
        )


type DeduplicationStrategyConfig = Annotated[
    NoDeduplicationStrategy | DeterministicIdDeduplicationStrategy | NormalizedTextHashDeduplicationStrategy | ShingleJaccardDeduplicationStrategy,
    Field(discriminator="kind"),
]


async def _item_document(items: list[KnowledgeItem], ctx: IngestionContext[ResolvedResource]) -> dict[str, Any]:
    return {"items": [{"id": str(item.id), "tenant_id": str(ctx.tenant_id), "source_uri": item.source_uri or ctx.resource.uri, "metadata": item.metadata, "text": await item.decode(ctx) if (await item.content_type(ctx)).startswith("text/") else None} for item in items]}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.casefold()


def _normalized_text_hash(text: str, existing_hash: str | None = None) -> str:
    if existing_hash:
        return existing_hash
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _shingles(text: str, shingle_size: int) -> set[str]:
    tokens = _normalize_text(text).split()
    if not tokens:
        return {""}
    if len(tokens) < shingle_size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + shingle_size]) for i in range(len(tokens) - shingle_size + 1)}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)
