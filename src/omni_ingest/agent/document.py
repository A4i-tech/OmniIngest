"""Document agents."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

import jq
import pymupdf
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.config import settings
from ..core.model import ByteContent, JqExpression, KnowledgeItem, ResolvedResource, Step, StepResult, StepStatus
from ..core.ocr import Figure, OcrBuilderParams, OcrOutputFormat


class ImageExtractionAgent(BaseModel, Step):
    """Extract images and vector drawings from PDF items."""

    foreach: JqExpression = Field(default=".items[]", description="jq expression selecting PDF items")
    input: JqExpression = Field(default="[.id]", description="jq expression selecting PDF item ids")
    dpi: PositiveInt = Field(default=240, description="DPI used to render extracted images")
    padding: NonNegativeFloat = Field(default=6, description="Padding around extracted images in PDF points")
    include_vector: bool = Field(default=True, description="Extract vector drawings")
    include_tables: bool = Field(default=False, description="Extract vector tables")
    deduplicate: bool = Field(default=True, description="Remove duplicate images from the same page")
    replace: bool = Field(default=False, description="Replace existing items with extracted images")
    mode: Literal["preserve", "redact", "caption"] = Field(default="preserve", description=(
        "How extracted images are represented in the parent PDF: preserve leaves the PDF unchanged, "
        "redact replaces each image with an `<image id=...>` marker, and caption keeps each image while "
        "adding its marker below it."
    ))
    workers: PositiveInt = Field(default=4, description="Maximum number of PDFs to process concurrently")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        item_docs = [{"id": str(item.id), "metadata": item.metadata, "source_uri": item.source_uri} for item in ctx.items]
        item_by_id = {doc["id"]: item for doc, item in zip(item_docs, ctx.items, strict=True)}
        specs = jq.compile(self.foreach).input_value({"metadata": ctx.metadata, "items": item_docs}).all()
        input_expr = jq.compile(self.input)
        parents: list[KnowledgeItem] = []

        for spec in specs:
            item_ids = input_expr.input_value(spec).first()
            if not isinstance(item_ids, list) or not item_ids or any(item_id not in item_by_id for item_id in item_ids):
                raise ValueError("ImageExtractionAgent input must resolve to known item ids")
            parents.extend(item_by_id[item_id] for item_id in item_ids)

        sem, extracted = asyncio.Semaphore(self.workers), []
        ctx.progress(0, len(parents), "extracting images")
        pdf_parents = [(parent, await parent.content.read(ctx)) for parent in parents if await parent.content.content_type(ctx) == "application/pdf"]
        for i, task in enumerate(asyncio.as_completed([self._extract(parent, content, sem) for parent, content in pdf_parents]), 1):
            extracted.extend(await task)
            ctx.progress(i, len(parents), f"extracted {i}/{len(parents)}")

        ctx.items = extracted if self.replace else [*ctx.items, *extracted]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"count": len(extracted), "replace": self.replace})

    async def _extract(self, parent: KnowledgeItem, content: bytes, sem: asyncio.Semaphore) -> list[KnowledgeItem]:
        async with sem:
            return await asyncio.to_thread(self._extract_pdf, parent, content)

    def _extract_pdf(self, parent: KnowledgeItem, content: bytes) -> list[KnowledgeItem]:
        extracted = []
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf.pages(), 1):
                assert isinstance(page, pymupdf.Page)

                regions = [pymupdf.Rect(image["bbox"]) for image in page.get_image_info()]
                tables = [] if self.include_tables or not self.include_vector else [pymupdf.Rect(table.bbox) for table in page.find_tables().tables]
                if self.include_vector:
                    regions.extend(rect for rect in page.cluster_drawings() if not any((rect & table).get_area() >= rect.get_area() * 0.8 for table in tables))

                seen = set()
                for rect in regions:
                    rect &= page.rect
                    area, page_area = rect.get_area(), page.rect.get_area()
                    if rect.is_empty or area < page_area * 0.001 or area > page_area * 0.8:
                        continue

                    rect = (rect + (-self.padding, -self.padding, self.padding, self.padding)) & page.rect
                    content = page.get_pixmap(clip=rect, dpi=self.dpi, alpha=False).tobytes("png")
                    if self.deduplicate and content in seen:
                        continue
                    seen.add(content)

                    id = uuid.uuid4()
                    marker = f'<image id="{id}"/>'
                    if self.mode == "redact":
                        page.add_redact_annot(rect, text=marker, fontsize=6, cross_out=False)
                    elif self.mode == "caption":
                        page.insert_text((rect.x0, rect.y1 + 7), marker, fontsize=6)

                    metadata = {"kind": "image", "parent_id": str(parent.id), "page": page_number, "bbox": [round(value, 3) for value in rect]}
                    extracted.append(KnowledgeItem(id=id, content=ByteContent.from_bytes(content, content_type="image/png", content_encoding="binary"), tenant_id=parent.tenant_id, source_uri=parent.source_uri, metadata=metadata))

                if self.mode == "redact":
                    page.apply_redactions()

            if self.mode != "preserve":
                parent.content = ByteContent.from_bytes(pdf.tobytes(deflate=True), content_type="application/pdf", content_encoding=parent.content.encoding)

        return extracted


class RasterizeAgent(BaseModel, Step):
    """Rasterizes PDF items into image-only PDFs.

    Useful when a PDF has a corrupted or unreliable text layer and downstream
    vision models must read the visible page content instead.
    """

    foreach: JqExpression = Field(default=".items[]", description="jq expression selecting PDF items")
    input: JqExpression = Field(default="[.id]", description="jq expression selecting PDF item ids")
    dpi: PositiveInt = Field(default=180, description="DPI used to render PDF pages")
    workers: PositiveInt = Field(default=4, description="Maximum number of PDFs to process concurrently")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        item_docs = [{"id": str(item.id), "metadata": item.metadata, "source_uri": item.source_uri} for item in ctx.items]
        item_by_id = {doc["id"]: item for doc, item in zip(item_docs, ctx.items, strict=True)}
        specs = jq.compile(self.foreach).input_value({"metadata": ctx.metadata, "items": item_docs}).all()
        input_expr = jq.compile(self.input)
        items: list[KnowledgeItem] = []

        for spec in specs:
            item_ids = input_expr.input_value(spec).first()
            if not isinstance(item_ids, list) or not item_ids or any(item_id not in item_by_id for item_id in item_ids):
                raise ValueError("RasterizeAgent input must resolve to known item ids")
            items.extend(item_by_id[item_id] for item_id in item_ids)

        if any([await item.content.content_type(ctx) != "application/pdf" for item in items]):
            raise ValueError("RasterizeAgent only supports application/pdf items")

        sem = asyncio.Semaphore(self.workers)
        ctx.progress(0, len(items), "rasterizing PDFs")
        tasks = [self._rasterize(item, await item.content.read(ctx), sem) for item in items]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            ctx.progress(i, len(items), f"rasterized {i}/{len(items)}")

        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"count": len(items), "dpi": self.dpi})

    async def _rasterize(self, item: KnowledgeItem, content: bytes, sem: asyncio.Semaphore) -> None:
        async with sem:
            await asyncio.to_thread(self._rasterize_pdf, item, content)

    def _rasterize_pdf(self, item: KnowledgeItem, content: bytes) -> None:
        with pymupdf.open(stream=content, filetype="pdf") as source, pymupdf.open() as output:
            for page in source:
                target = output.new_page(width=page.rect.width, height=page.rect.height)
                target.insert_image(target.rect, stream=page.get_pixmap(dpi=self.dpi, alpha=False).tobytes("png"))
            item.content = ByteContent.from_bytes(output.tobytes(deflate=True), content_type="application/pdf", content_encoding=item.content.encoding)


class OcrAgent(BaseModel, Step):
    """Extracts items using a configured OCR engine."""

    foreach: JqExpression = Field(default=".items[]", description="jq expression selecting items")
    input: JqExpression = Field(default="[.id]", description="jq expression selecting item ids")
    engine: str = Field(default=settings.default_ocr_engine, description="OCR engine name")
    format: OcrOutputFormat = Field(default=OcrOutputFormat.MARKDOWN, description="OCR output format")
    src_lang: str | None = Field(default=None, description="Source language hint for OCR engines that support it")
    dst_lang: str | None = Field(default=None, description="Destination language hint for OCR engines that support it")
    concurrency: PositiveInt = Field(default=4, description="Maximum number of items OCR'd concurrently")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        item_docs = [{"id": str(item.id), "metadata": item.metadata, "source_uri": item.source_uri} for item in ctx.items]
        item_by_id = {doc["id"]: item for doc, item in zip(item_docs, ctx.items, strict=True)}
        specs = jq.compile(self.foreach).input_value({"metadata": ctx.metadata, "items": item_docs}).all()
        input_expr = jq.compile(self.input)
        items: list[KnowledgeItem] = []

        for spec in specs:
            item_ids = input_expr.input_value(spec).first()
            if not isinstance(item_ids, list) or not item_ids or any(item_id not in item_by_id for item_id in item_ids):
                raise ValueError("OcrAgent input must resolve to known item ids")
            items.extend(item_by_id[item_id] for item_id in item_ids)

        ctx.progress(0, len(items), "extracting text")
        ocr = ctx.ocr_factory(self.engine, OcrBuilderParams(ctx=ctx, format=self.format, concurrency=self.concurrency, src_lang=self.src_lang, dst_lang=self.dst_lang))

        item_by_content_id = {id(item.content): item for item in items}

        async def contents() -> AsyncIterator[ByteContent]:
            for item in items:
                yield item.content

        n = 0
        extracted: list[KnowledgeItem] = []
        figures_failed = 0
        async for source, changed in ocr(contents()):
            if changed is source:  # 'is' is the right way to test for identical
                continue

            original = item_by_content_id[id(source)]
            original.content = changed

            # Figures become items in their own right so later steps can address them by id.
            figures: list[Figure] = changed.metadata.get("figures") or []
            for figure in figures:
                page = original.metadata.get("page")
                if page is None:
                    page = figure.content.metadata.get("page")
                extracted.append(KnowledgeItem(
                    content=figure.content,
                    tenant_id=original.tenant_id,
                    source_uri=original.source_uri,
                    metadata={
                        "kind": "image",
                        "parent_id": str(original.id),
                        "page": page,
                        "caption": figure.caption,
                        "figure_id": figure.content.metadata.get("id"),
                        "polygon": figure.content.metadata.get("polygon"),
                        "content_type": figure.content.metadata.get("content_type"),
                        "source": "ocr_engine",
                    },
                ))

            figures_failed += changed.metadata.get("figures_failed", 0)
            n += 1
            ctx.progress(n, len(items), f"extracted {n}/{len(items)}")

        ctx.items = [*ctx.items, *extracted]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items,
                           metadata={"count": len(items), "figures": len(extracted), "figures_failed": figures_failed})
