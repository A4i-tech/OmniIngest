"""Persistent storage implementation for pipeline metadata and tenant information."""

import asyncio
import pickle
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, col

from ..core.config import settings
from ..core.model import KnowledgeItem, KnowledgeItemRecord, Pipeline, PipelineCheckpoint, PipelineCheckpointRecord, PipelineRunRecord, PipelineStepRunRecord, PipelineVersion, Tenant
from ..core.protocol import JsonObject, MetadataStore


class SQLAlchemyMetadataStore(MetadataStore):
    """
    SQLAlchemy-based implementation of the MetadataStore protocol.

    Persistent storage for tenants, pipeline runs, and knowledge base tracking
    using an asynchronous SQL backend.
    """

    def __init__(self, url: str | None = None):
        self.engine = create_async_engine(url or settings.metadata_store)
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
                await conn.run_sync(SQLModel.metadata.create_all)
            self._init = True

    async def close(self):
        """Disposes of the database engine and releases all connections."""
        await self._init_db()
        await self.engine.dispose()

    async def __aenter__(self):
        await self._init_db()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def get_tenant(self, slug: str) -> Tenant | None:
        await self._init_db()
        async with self.session_factory() as s:
            return (await s.execute(select(Tenant).where(col(Tenant.slug) == slug))).scalar_one_or_none()

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
            r = PipelineRunRecord(
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

    async def update_pipeline_run(self, run_id: UUID, status: str, completed_at: datetime | None = None, error: str | None = None) -> None:
        await self._init_db()
        async with self.session_factory() as s:
            r = await s.get(PipelineRunRecord, run_id)
            if r:
                r.status, r.completed_at, r.error_message = status, completed_at, error
                await s.commit()

    async def get_pipeline_checkpoint(self, id: UUID) -> PipelineCheckpoint | None:
        await self._init_db()
        async with self.session_factory() as s:
            row = await s.get(PipelineCheckpointRecord, id)
            if not row:
                return None
            checkpoint = pickle.loads(row.checkpoint)
            if not isinstance(checkpoint, PipelineCheckpoint):
                raise TypeError(f"Expected PipelineCheckpoint, got {type(checkpoint).__name__}")
            return checkpoint

    async def set_pipeline_checkpoint(self, id: UUID, checkpoint: PipelineCheckpoint | None) -> None:
        await self._init_db()
        async with self.session_factory() as s:
            if checkpoint is None:
                if row := await s.get(PipelineCheckpointRecord, id):
                    await s.delete(row)
            else:
                await s.merge(PipelineCheckpointRecord(pipeline_run_id=id, checkpoint=pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)))
            await s.commit()

    async def create_step_run(self, run_id: UUID, step_name: str) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            st = PipelineStepRunRecord(pipeline_run_id=run_id, step_name=step_name, status="running", started_at=datetime.now(UTC))
            s.add(st)
            await s.commit()
            return st.id

    async def update_step_run(self, step_id: UUID, status: str, metadata: JsonObject | None = None, error: str | None = None) -> None:
        await self._init_db()
        async with self.session_factory() as s:
            st = await s.get(PipelineStepRunRecord, step_id)
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
        content: bytes | None = None,
        pipeline_version_id: UUID | None = None,
        title: str | None = None,
        description: str | None = None,
        structured_views: JsonObject | None = None,
        lineage: JsonObject | None = None,
        metadata: JsonObject | None = None,
    ) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            i = KnowledgeItemRecord(
                tenant_id=tenant_id,
                domain_profile=domain_profile,
                modality=modality,
                source_uri=source_uri,
                content=content or b"",
                pipeline_version_id=pipeline_version_id,
                title=title,
                description=description,
                structured_views=structured_views,
                lineage=lineage,
                metadata_=metadata or {},
            )
            s.add(i)
            await s.commit()
            return i.id

    async def get_knowledge_item(self, item_id: UUID) -> KnowledgeItem | None:
        await self._init_db()
        async with self.session_factory() as s:
            if r := await s.get(KnowledgeItemRecord, item_id):
                return KnowledgeItem.model_validate(r)
            return None

    async def list_knowledge_items(self, tenant_id: UUID, domain_profile: str | None = None, **filters: object) -> list[KnowledgeItem]:
        await self._init_db()
        async with self.session_factory() as s:
            q = select(KnowledgeItemRecord).where(col(KnowledgeItemRecord.tenant_id) == tenant_id)
            if domain_profile:
                q = q.where(col(KnowledgeItemRecord.domain_profile) == domain_profile)
            for k, v in filters.items():
                q = q.where(getattr(KnowledgeItemRecord, k) == v)
            records = (await s.execute(q)).scalars().all()
            return [KnowledgeItem.model_validate(r) for r in records]

    async def get_pipeline_version(self, pipeline_name: str, version: str) -> PipelineVersion | None:
        await self._init_db()
        async with self.session_factory() as s:
            return (await s.execute(select(PipelineVersion).join(Pipeline).where(col(Pipeline.name) == pipeline_name, col(PipelineVersion.version) == version))).scalar_one_or_none()

    async def get_default_pipeline_version(self, pipeline_name: str) -> PipelineVersion | None:
        await self._init_db()
        async with self.session_factory() as s:
            return (await s.execute(select(PipelineVersion).join(Pipeline).where(col(Pipeline.name) == pipeline_name, col(PipelineVersion.is_default)))).scalar_one_or_none()

    async def upsert_pipeline(self, name: str, domain_profile: str) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            p = (await s.execute(select(Pipeline).where(col(Pipeline.name) == name))).scalar_one_or_none()
            if not p:
                p = Pipeline(name=name, domain_profile=domain_profile)
                s.add(p)
            else:
                p.domain_profile = domain_profile
            await s.commit()
            return p.id

    async def upsert_pipeline_version(self, pipeline_id: UUID, version: str, yaml_profile_name: str, is_default: bool = False) -> UUID:
        await self._init_db()
        async with self.session_factory() as s:
            pv = (await s.execute(select(PipelineVersion).where(col(PipelineVersion.pipeline_id) == pipeline_id, col(PipelineVersion.version) == version))).scalar_one_or_none()
            if not pv:
                pv = PipelineVersion(pipeline_id=pipeline_id, version=version, yaml_profile_name=yaml_profile_name, is_default=is_default)
                s.add(pv)
            else:
                pv.yaml_profile_name, pv.is_default = yaml_profile_name, is_default
            await s.flush()
            if is_default:
                await s.execute(update(PipelineVersion).where(col(PipelineVersion.pipeline_id) == pipeline_id, col(PipelineVersion.id) != pv.id).values(is_default=False))
            await s.commit()
            return pv.id


class NullMetadataStore(MetadataStore):
    """
    Metadata store that preserves metadata records without storing item content.
    """

    def __init__(self):
        self._inner = SQLAlchemyMetadataStore("sqlite+aiosqlite:///:memory:")

    async def close(self) -> None:
        await self._inner.close()

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._inner.__aexit__(exc_type, exc, tb)

    async def get_tenant(self, slug: str) -> Tenant | None:
        return await self._inner.get_tenant(slug)

    async def create_tenant(self, slug: str, display_name: str) -> Tenant:
        return await self._inner.create_tenant(slug, display_name)

    async def create_pipeline_run(self, tenant_id: UUID, pipeline_version_id: UUID, source_uri: str, id: UUID | None = None) -> UUID:
        return await self._inner.create_pipeline_run(tenant_id, pipeline_version_id, source_uri, id=id)

    async def update_pipeline_run(self, run_id: UUID, status: str, completed_at: datetime | None = None, error: str | None = None) -> None:
        await self._inner.update_pipeline_run(run_id, status, completed_at=completed_at, error=error)

    async def get_pipeline_checkpoint(self, id: UUID) -> PipelineCheckpoint | None:
        return None

    async def set_pipeline_checkpoint(self, id: UUID, checkpoint: PipelineCheckpoint | None) -> None:
        return None

    async def create_step_run(self, run_id: UUID, step_name: str) -> UUID:
        return await self._inner.create_step_run(run_id, step_name)

    async def update_step_run(self, step_id: UUID, status: str, metadata: JsonObject | None = None, error: str | None = None) -> None:
        await self._inner.update_step_run(step_id, status, metadata=metadata, error=error)

    async def create_knowledge_item(
        self,
        tenant_id: UUID,
        domain_profile: str,
        modality: str,
        source_uri: str,
        content: bytes | None = None,
        pipeline_version_id: UUID | None = None,
        title: str | None = None,
        description: str | None = None,
        structured_views: JsonObject | None = None,
        lineage: JsonObject | None = None,
        metadata: JsonObject | None = None,
    ) -> UUID:
        return await self._inner.create_knowledge_item(
            tenant_id=tenant_id,
            domain_profile=domain_profile,
            modality=modality,
            source_uri=source_uri,
            content=b"",
            pipeline_version_id=pipeline_version_id,
            title=title,
            description=description,
            structured_views=structured_views,
            lineage=lineage,
            metadata=metadata,
        )

    async def get_knowledge_item(self, item_id: UUID) -> KnowledgeItem | None:
        return await self._inner.get_knowledge_item(item_id)

    async def list_knowledge_items(self, tenant_id: UUID, domain_profile: str | None = None, **filters: object) -> list[KnowledgeItem]:
        return await self._inner.list_knowledge_items(tenant_id, domain_profile=domain_profile, **filters)

    async def get_pipeline_version(self, pipeline_name: str, version: str) -> PipelineVersion | None:
        return await self._inner.get_pipeline_version(pipeline_name, version)

    async def get_default_pipeline_version(self, pipeline_name: str) -> PipelineVersion | None:
        return await self._inner.get_default_pipeline_version(pipeline_name)

    async def upsert_pipeline(self, name: str, domain_profile: str) -> UUID:
        return await self._inner.upsert_pipeline(name, domain_profile)

    async def upsert_pipeline_version(self, pipeline_id: UUID, version: str, yaml_profile_name: str, is_default: bool = False) -> UUID:
        return await self._inner.upsert_pipeline_version(pipeline_id, version, yaml_profile_name, is_default=is_default)
