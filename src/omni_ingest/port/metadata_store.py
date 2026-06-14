"""Persistent storage implementation for pipeline metadata and tenant information."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ..core.config import settings
from ..core.model import KnowledgeItem
from ..core.protocol import MetadataStore
from ..db import Base, Chunk, KnowledgeItemRecord, Pipeline, PipelineRun, PipelineStepRun, PipelineVersion, Tenant


class SQLAlchemyMetadataStore(MetadataStore):
    """
    SQLAlchemy-based implementation of the MetadataStore protocol.

    Persistent storage for tenants, pipeline runs, and knowledge base tracking
    using an asynchronous SQL backend.
    """

    @classmethod
    def in_memory(cls):
        return cls("sqlite+aiosqlite:///:memory:")

    def __init__(self, database_url: str | None = None):
        url = database_url or settings.database_url
        self.engine = create_async_engine(url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self._init_lock = asyncio.Lock()
        self._init = False

    async def _init_db(self):
        if self._init:
            return
        async with self._init_lock:
            if self._init:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._init = True

    async def close(self):
        """Disposes of the database engine and releases all connections."""
        await self._init_db()
        await self.engine.dispose()

    async def get_tenant(self, slug: str) -> Tenant | None:
        await self._init_db()
        async with self.session_factory() as s:
            return (await s.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()

    async def create_tenant(self, slug: str, display_name: str) -> Tenant:
        await self._init_db()
        async with self.session_factory() as s:
            t = Tenant(slug=slug, display_name=display_name)
            s.add(t)
            await s.commit()
            return t

    async def create_pipeline_run(self, tenant_id: UUID, pipeline_version_id: UUID, source_uri: str, id: UUID | None = None) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            r = PipelineRun(
                id=id or uuid4(),
                tenant_id=tenant_id,
                pipeline_version_id=pipeline_version_id,
                source_uri=source_uri,
                status="running",
                started_at=datetime.now(UTC),
            )
            s.add(r)
            await s.commit()
            return r.id

    async def update_pipeline_run(self, run_id: UUID, status: str, completed_at: Any | None = None, error: str | None = None) -> None:
        await self._init_db()
        async with self.session_factory() as s:
            r = await s.get(PipelineRun, run_id)
            if r:
                r.status = status
                if completed_at:
                    r.completed_at = completed_at
                if error:
                    r.error_message = error
                await s.commit()

    async def create_step_run(self, run_id: UUID, step_name: str) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            st = PipelineStepRun(pipeline_run_id=run_id, step_name=step_name, status="running", started_at=datetime.now(UTC))
            s.add(st)
            await s.commit()
            return st.id

    async def update_step_run(self, step_id: UUID, status: str, metadata: dict[str, Any] | None = None, error: str | None = None) -> None:
        await self._init_db()
        async with self.session_factory() as s:
            st = await s.get(PipelineStepRun, step_id)
            if st:
                st.status, st.metadata_ = status, metadata or st.metadata_
                if error:
                    st.error_message = error
                st.completed_at = datetime.now(UTC)
                await s.commit()

    async def create_knowledge_item(
        self,
        tenant_id: UUID,
        domain_profile: str,
        modality: str,
        source_uri: str,
        content: str | None = None,
        pipeline_version_id: Any | None = None,
        title: str | None = None,
        description: str | None = None,
        structured_views: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            i = KnowledgeItemRecord(
                tenant_id=tenant_id,
                domain_profile=domain_profile,
                modality=modality,
                source_uri=source_uri,
                content=content,
                pipeline_version_id=pipeline_version_id,
                title=title,
                description=description,
                structured_views=structured_views,
                lineage=lineage,
                metadata_=metadata,
            )
            s.add(i)
            await s.commit()
            return i.id

    async def get_knowledge_item(self, item_id: UUID) -> KnowledgeItem | None:
        await self._init_db()
        async with self.session_factory() as s:
            if r := await s.get(KnowledgeItemRecord, item_id):
                return KnowledgeItem(
                    id=r.id,
                    tenant_id=r.tenant_id,
                    source_uri=r.source_uri,
                    content=(r.content or "").encode(),
                    metadata=r.metadata_ or {},
                )
            return None

    async def create_chunk(
        self,
        chunk_id: UUID,
        knowledge_item_id: Any,
        tenant_id: UUID,
        text: str,
        vector_store_kind: str,
        vector_store_space: str,
        vector_store_ref_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        await self._init_db()
        async with self.session_factory() as s:
            c = Chunk(
                id=str(chunk_id),
                knowledge_item_id=knowledge_item_id,
                tenant_id=tenant_id,
                text=text,
                vector_store_kind=vector_store_kind,
                vector_store_space=vector_store_space,
                vector_store_ref_id=vector_store_ref_id,
                metadata_=metadata,
            )
            s.add(c)
            await s.commit()
            return c.id

    async def get_chunks_by_knowledge_item(self, knowledge_item_id: UUID) -> list[Chunk]:
        await self._init_db()
        async with self.session_factory() as s:
            return list((await s.execute(select(Chunk).where(Chunk.knowledge_item_id == knowledge_item_id))).scalars().all())

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        await self._init_db()
        async with self.session_factory() as s:
            return await s.get(Chunk, chunk_id)

    async def list_knowledge_items(self, tenant_id: UUID, domain_profile: str | None = None, **filters) -> list[KnowledgeItem]:
        await self._init_db()
        async with self.session_factory() as s:
            q = select(KnowledgeItemRecord).where(KnowledgeItemRecord.tenant_id == tenant_id)
            if domain_profile:
                q = q.where(KnowledgeItemRecord.domain_profile == domain_profile)
            for k, v in filters.items():
                q = q.where(getattr(KnowledgeItemRecord, k) == v)
            records = (await s.execute(q)).scalars().all()
            return [
                KnowledgeItem(
                    id=r.id,
                    tenant_id=r.tenant_id,
                    source_uri=r.source_uri,
                    content=(r.content or "").encode(),
                    metadata=r.metadata_ or {},
                )
                for r in records
            ]

    async def list_chunks(self, tenant_id: UUID, **filters) -> list[Chunk]:
        await self._init_db()
        async with self.session_factory() as s:
            q = select(Chunk).where(Chunk.tenant_id == tenant_id)
            for k, v in filters.items():
                q = q.where(getattr(Chunk, k) == v)
            return list((await s.execute(q)).scalars().all())

    async def get_pipeline_version(self, pipeline_name: str, version: str) -> PipelineVersion | None:
        await self._init_db()
        async with self.session_factory() as s:
            return (await s.execute(select(PipelineVersion).join(Pipeline).where(Pipeline.name == pipeline_name, PipelineVersion.version == version))).scalar_one_or_none()

    async def get_default_pipeline_version(self, pipeline_name: str) -> PipelineVersion | None:
        await self._init_db()
        async with self.session_factory() as s:
            return (await s.execute(select(PipelineVersion).join(Pipeline).where(Pipeline.name == pipeline_name, PipelineVersion.is_default))).scalar_one_or_none()

    async def upsert_pipeline(self, name: str, domain_profile: str) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            p = (await s.execute(select(Pipeline).where(Pipeline.name == name))).scalar_one_or_none()
            if not p:
                p = Pipeline(name=name, domain_profile=domain_profile)
                s.add(p)
            else:
                p.domain_profile = domain_profile
            await s.commit()
            return p.id

    async def upsert_pipeline_version(self, pipeline_id: str, version: str, yaml_profile_name: str, is_default: bool = False) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            pv = (await s.execute(select(PipelineVersion).where(PipelineVersion.pipeline_id == pipeline_id, PipelineVersion.version == version))).scalar_one_or_none()
            if not pv:
                pv = PipelineVersion(pipeline_id=pipeline_id, version=version, yaml_profile_name=yaml_profile_name, is_default=is_default)
                s.add(pv)
            else:
                pv.yaml_profile_name, pv.is_default = yaml_profile_name, is_default
            await s.flush()
            if is_default:
                await s.execute(update(PipelineVersion).where(PipelineVersion.pipeline_id == pipeline_id, PipelineVersion.id != pv.id).values(is_default=False))
            await s.commit()
            return pv.id
