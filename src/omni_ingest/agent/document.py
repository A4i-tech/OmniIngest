"""Document agents."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

import jq
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.config import settings
from ..core.model import ByteContent, JqExpression, KnowledgeItem, ResolvedResource, Step, StepResult, StepStatus
from ..core.ocr import Figure, OcrBuilderParams, OcrOutputFormat
from ..pdf import Pdf, Rect, rasterize


def _intersect(a: Rect, b: Rect) -> Rect:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _area(r: Rect) -> float:
    return max(r[2] - r[0], 0) * max(r[3] - r[1], 0)


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
        "redact removes each image and the text under it and writes an `<image id=...>` marker in its place, "
        "and caption keeps each image while adding its marker below it."
    ))
    workers: PositiveInt = Field(default=4, description="Maximum number of PDFs to process concurrently")

    @property
    def replaces_items(self) -> bool:
        return self.replace

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
        with Pdf.open(content) as pdf:
            for page_number in range(1, len(pdf) + 1):
                page = pdf.page(page_number - 1)

                regions = page.image_boxes()
                tables = [] if self.include_tables or not self.include_vector else page.table_boxes()
                if self.include_vector:
                    regions.extend(rect for rect in page.drawing_clusters() if not any(_area(_intersect(rect, table)) >= _area(rect) * 0.8 for table in tables))

                seen: dict[bytes, uuid.UUID] = {}
                for rect in regions:
                    rect = _intersect(rect, page.rect)
                    area, page_area = _area(rect), _area(page.rect)
                    if area < page_area * 0.001 or area > page_area * 0.8:
                        continue

                    rect = _intersect((rect[0] - self.padding, rect[1] - self.padding, rect[2] + self.padding, rect[3] + self.padding), page.rect)
                    content = page.render_png(rect, self.dpi)
                    duplicate = self.deduplicate and content in seen
                    id = seen[content] if duplicate else uuid.uuid4()
                    seen.setdefault(content, id)
                    marker = f'<image id="{id}"/>'
                    if self.mode == "redact":
                        page.redact(rect, marker)
                    elif self.mode == "caption":
                        page.caption(rect, marker)
                    if duplicate:
                        continue

                    metadata = {"kind": "image", "parent_id": str(parent.id), "page": page_number, "bbox": [round(value, 3) for value in rect]}
                    extracted.append(KnowledgeItem(id=id, content=ByteContent.from_bytes(content, content_type="image/png", content_encoding="binary"), tenant_id=parent.tenant_id, source_uri=parent.source_uri, metadata=metadata))

            if self.mode != "preserve":
                parent.content = ByteContent.from_bytes(pdf.to_bytes(), content_type="application/pdf", content_encoding=parent.content.encoding)

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
        item.content = ByteContent.from_bytes(rasterize(content, self.dpi), content_type="application/pdf", content_encoding=item.content.encoding)


class OcrAgent(BaseModel, Step):
    """Extracts items using a configured OCR engine."""

    foreach: JqExpression = Field(default=".items[]", description="jq expression selecting items")
    input: JqExpression = Field(default="[.id]", description="jq expression selecting item ids")
    engine: str = Field(default=settings.default_ocr_engine, description="OCR engine name")
    format: OcrOutputFormat = Field(default=OcrOutputFormat.MARKDOWN, description="OCR output format")
    src_lang: str | None = Field(default=None, description="Source language hint for OCR engines that support it")
    dst_lang: str | None = Field(default=None, description="Destination language hint for OCR engines that support it")
    concurrency: PositiveInt = Field(default=4, description="Maximum number of items OCR'd concurrently")
    extract_images: bool = Field(default=False, description="Return detected figures as image items (Mistral engines only)")
    image_min_size: PositiveInt | None = Field(default=None, description="Smallest figure edge in pixels to extract")
    extract_header_footer: bool = Field(default=False, description="Keep page headers and footers out of the text and store them in item metadata")
    deduplicate: bool = Field(default=False, description="Create one image item for figures with identical bytes. Needs extract_images")

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
        ocr = ctx.ocr_factory(self.engine, OcrBuilderParams(
            ctx=ctx, format=self.format, concurrency=self.concurrency, src_lang=self.src_lang, dst_lang=self.dst_lang,
            extract_images=self.extract_images, image_min_size=self.image_min_size,
            extract_header_footer=self.extract_header_footer, deduplicate=self.deduplicate,
        ))

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
            for key in ("ocr_header", "ocr_footer"):
                if key in changed.metadata:
                    original.metadata[key] = changed.metadata[key]

            # Figures become items in their own right so later steps can address them by id.
            figures: list[Figure] = changed.metadata.get("figures") or []
            for figure in figures:
                page = original.metadata.get("page")
                if page is None:
                    page = figure.content.metadata.get("page")
                extracted.append(KnowledgeItem(
                    id=figure.id,
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
                        "bbox": figure.content.metadata.get("bbox"),
                        "page_dimensions": figure.content.metadata.get("page_dimensions"),
                        "content_type": figure.content.metadata.get("content_type"),
                        "source": self.engine,
                    },
                ))

            figures_failed += changed.metadata.get("figures_failed", 0)
            n += 1
            ctx.progress(n, len(items), f"extracted {n}/{len(items)}")

        ctx.items = [*ctx.items, *extracted]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items,
                           metadata={"count": len(items), "figures": len(extracted), "figures_failed": figures_failed})
