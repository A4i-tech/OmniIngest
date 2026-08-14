"""OCR engine builders for document/page-image to Markdown extraction."""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING

import pymupdf
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import BinaryContent

from ..core.config import settings
from ..core.model import AgentMixin, ByteContent, ResolvedResource
from ..core.ocr import Ocr, OcrFactory, build_simple_factory

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext


class _DefaultAgentMixin(BaseModel, AgentMixin):
    """Combines with `BaseModel` so `AgentMixin`'s fields resolve to real defaults."""


DEFAULT_VISION_OCR_PROMPT = "Transcribe this document to Markdown, preserving headings, lists, and tables. Return only the Markdown."


def by_page(extract_page: Ocr, dpi: int = 200, ocr_scanned: bool = True, text_layer_threshold: int = 20) -> Ocr:
    """Wrap a single-image `Ocr` into a full-document `Ocr`"""

    def render_pdf_pages(content: bytes) -> list[tuple[str, bool, bytes | None]]:
        pages = []
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            for page in pdf.pages():
                assert isinstance(page, pymupdf.Page)
                text = page.get_text().strip()
                needs_ocr = ocr_scanned and len(text) < text_layer_threshold
                image = page.get_pixmap(dpi=dpi, alpha=False).tobytes("png") if needs_ocr else None
                pages.append((text, needs_ocr, image))
        return pages

    async def ocr(item: ByteContent) -> ByteContent:
        if item.metadata.get("content_type") != "application/pdf":
            return await extract_page(item)

        pages = await asyncio.to_thread(render_pdf_pages, item.raw_content)

        async def resolve(text: str, needs_ocr: bool, image: bytes | None) -> str:
            if not needs_ocr or image is None:
                return text
            result = await extract_page(ByteContent(raw_content=image, metadata={"content_type": "image/png"}))
            return result.raw_content.decode(result.content_encoding)

        markdown = "\n\n".join(await asyncio.gather(*(resolve(*page) for page in pages)))
        return ByteContent(raw_content=markdown.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return ocr


def llm_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    """OCR via a vision LLM. Sends the document as-is (PDF or image) in one call — any"""
    agent = _DefaultAgentMixin()._agent(ctx)

    async def ocr(item: ByteContent) -> ByteContent:
        content_type = await item.content_type(ctx)
        result = await agent.run([DEFAULT_VISION_OCR_PROMPT, BinaryContent(data=await item.content(ctx), media_type=content_type)])
        return ByteContent(raw_content=result.output.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return ocr


def openai_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    """OCR via any OpenAI-compatible vision/chat-completions endpoint."""
    if settings.ocr_api_key is None or settings.ocr_base_url is None:
        raise ValueError("OCR_API_KEY/OCR_BASE_URL are not configured")
    client = AsyncOpenAI(api_key=settings.ocr_api_key.get_secret_value(), base_url=str(settings.ocr_base_url))

    async def extract_page(image: ByteContent) -> ByteContent:
        content_type = await image.content_type(ctx)
        data = await image.content(ctx)
        response = await client.chat.completions.create(
            model=settings.ocr_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEFAULT_VISION_OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64.b64encode(data).decode()}"}},
                    ],
                }
            ],
        )
        markdown = response.choices[0].message.content or ""
        return ByteContent(raw_content=markdown.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return by_page(extract_page)


def azure_di_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    if settings.documentintelligence_endpoint is None or settings.documentintelligence_api_key is None:
        raise ValueError("DOCUMENTINTELLIGENCE_ENDPOINT/DOCUMENTINTELLIGENCE_API_KEY are not configured")
    endpoint = str(settings.documentintelligence_endpoint)
    credential = AzureKeyCredential(settings.documentintelligence_api_key.get_secret_value())

    async def extract_page(image: ByteContent) -> ByteContent:
        async with DocumentIntelligenceClient(endpoint=endpoint, credential=credential) as client:
            poller = await client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=await image.content(ctx)),
                output_content_format=DocumentContentFormat.MARKDOWN,
            )
            result = await poller.result()
        markdown = result.content or ""
        return ByteContent(raw_content=markdown.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return by_page(extract_page)


def default_factory() -> OcrFactory:
    return build_simple_factory({
        "llm": llm_ocr_builder,
        "openai": openai_ocr_builder,
        "azure_di": azure_di_ocr_builder,
    })
