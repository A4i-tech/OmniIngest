"""Document agents."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import jq
import pymupdf
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import JqExpression, KnowledgeItem, ResolvedResource, Step, StepResult, StepStatus


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
    workers: PositiveInt = Field(default=4, description="Maximum number of PDFs to process concurrently")

    async def run(self, ctx: IngestionContext[ResolvedResource]) -> StepResult:
        item_docs = [{"id": str(item.id), "metadata": item.metadata, "source_uri": item.source_uri} for item in ctx.items]
        item_by_id = {doc["id"]: item for doc, item in zip(item_docs, ctx.items, strict=True)}
        specs = jq.compile(self.foreach).input_value({"metadata": ctx.metadata, "items": item_docs}).all()
        input_expr, parents = jq.compile(self.input), []

        for spec in specs:
            item_ids = input_expr.input_value(spec).first()
            if not isinstance(item_ids, list) or not item_ids or any(item_id not in item_by_id for item_id in item_ids):
                raise ValueError("ImageExtractionAgent input must resolve to known item ids")
            parents.extend(item_by_id[item_id] for item_id in item_ids)

        sem, extracted = asyncio.Semaphore(self.workers), []
        ctx.progress(0, len(parents), "extracting images")
        for i, task in enumerate(asyncio.as_completed([self._extract(parent, sem) for parent in parents if parent.content_type == "application/pdf"]), 1):
            extracted.extend(await task)
            ctx.progress(i, len(parents), f"extracted {i}/{len(parents)}")

        ctx.items = extracted if self.replace else [*ctx.items, *extracted]
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"count": len(extracted), "replace": self.replace})

    async def _extract(self, parent: KnowledgeItem, sem: asyncio.Semaphore) -> list[KnowledgeItem]:
        async with sem:
            return await asyncio.to_thread(self._extract_pdf, parent)

    def _extract_pdf(self, parent: KnowledgeItem) -> list[KnowledgeItem]:
        extracted = []
        with pymupdf.open(stream=parent.content, filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf.pages(), 1):
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
                    metadata = {"kind": "image", "parent_id": str(parent.id), "page": page_number, "bbox": [round(value, 3) for value in rect], "content_type": "image/png"}
                    extracted.append(KnowledgeItem(content=content, content_encoding="binary", tenant_id=parent.tenant_id, source_uri=parent.source_uri, metadata=metadata))

        return extracted