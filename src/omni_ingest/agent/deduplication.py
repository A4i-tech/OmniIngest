"""Deduplication strategies for vector indexing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from ..core.model import IngestionContext, KnowledgeItem, ResolvedResource
from ..core.protocol import MetadataStore

type DeduplicationScope = Literal["source", "tenant"]

_NORMALIZED_TEXT_HASH_KEY = "dedup_normalized_text_hash"
_DETERMINISTIC_ID_KEY = "dedup_deterministic_id"
_SHINGLE_SIZE_KEY = "dedup_shingle_size"
_WHITESPACE_RE = re.compile(r"\s+")


class DeduplicationResult(BaseModel):
    """Outcome of applying deduplication to a batch of chunks."""

    kept_items: list[KnowledgeItem]
    input_count: int
    dropped_count: int
    strategy: str
    scope: DeduplicationScope | None = None


class DeduplicationStrategy(BaseModel, ABC):
    """Base class for chunk deduplication strategies."""

    model_config = ConfigDict(extra="forbid")
    kind: str

    @abstractmethod
    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        """Returns the subset of items that should be indexed."""


class ScopedDeduplicationStrategy(DeduplicationStrategy, ABC):
    """Deduplication strategy that can operate within a tenant or a source."""

    scope: DeduplicationScope = Field("source", description="Scope used to compare a chunk against previously indexed chunks")

    async def _existing_chunks(self, ctx: IngestionContext[ResolvedResource]) -> list[Any]:
        store = ctx.store
        if not isinstance(store, MetadataStore):
            return []

        if self.scope == "tenant":
            return await store.list_chunks(ctx.tenant_id)

        source_uri = ctx.resource.uri
        if source_uri:
            items = await store.list_knowledge_items(tenant_id=ctx.tenant_id, source_uri=source_uri)
            chunks: list[Any] = []
            for item in items:
                chunks.extend(await store.get_chunks_by_knowledge_item(item.id))
            return chunks

        if knowledge_item_id := ctx.metadata.get("knowledge_item_id"):
            return await store.get_chunks_by_knowledge_item(knowledge_item_id)

        return []


class NoDeduplicationStrategy(DeduplicationStrategy):
    """Leaves the batch unchanged."""

    kind: Literal["none"] = "none"

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        del ctx
        return DeduplicationResult(kept_items=items, input_count=len(items), dropped_count=0, strategy=self.kind)


class DeterministicIdDeduplicationStrategy(ScopedDeduplicationStrategy):
    """Assigns stable IDs to chunks and skips IDs that already exist."""

    kind: Literal["deterministic_id"] = "deterministic_id"

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        seen_ids = {chunk.id for chunk in await self._existing_chunks(ctx)}
        kept_items: list[KnowledgeItem] = []

        for item in items:
            deterministic_id = str(uuid5(NAMESPACE_URL, _deterministic_identity(item, ctx)))
            item.id = uuid5(NAMESPACE_URL, _deterministic_identity(item, ctx))
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
    """Deduplicates chunks using a hash of normalized text."""

    kind: Literal["normalized_text_hash"] = "normalized_text_hash"

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        seen_hashes = {
            _normalized_text_hash(chunk.text, (chunk.metadata_ or {}).get(_NORMALIZED_TEXT_HASH_KEY))
            for chunk in await self._existing_chunks(ctx)
        }
        kept_items: list[KnowledgeItem] = []

        for item in items:
            normalized_hash = _normalized_text_hash(item.decode())
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
    """Deduplicates chunks using Jaccard similarity over token shingles."""

    kind: Literal["shingle_jaccard"] = "shingle_jaccard"
    threshold: float = Field(0.9, ge=0.0, le=1.0, description="Minimum Jaccard similarity that counts as a duplicate")
    shingle_size: PositiveInt = Field(3, description="Token shingle size used for near-duplicate detection")

    async def deduplicate(self, items: list[KnowledgeItem], ctx: IngestionContext[Any]) -> DeduplicationResult:
        seen_shingles = [_shingles(chunk.text, self.shingle_size) for chunk in await self._existing_chunks(ctx)]
        kept_items: list[KnowledgeItem] = []

        for item in items:
            item_shingles = _shingles(item.decode(), self.shingle_size)
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


def _deterministic_identity(item: KnowledgeItem, ctx: IngestionContext[ResolvedResource]) -> str:
    source_uri = item.source_uri or ctx.resource.uri or ""
    position_keys = ("char_start", "source_idx", "page", "start_offset_ms", "end_offset_ms")
    position = "|".join(f"{k}={item.metadata[k]}" for k in position_keys if k in item.metadata)
    normalized = _normalize_text(item.decode())
    return "|".join((str(ctx.tenant_id), source_uri, ctx.domain_profile, position, normalized))


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
