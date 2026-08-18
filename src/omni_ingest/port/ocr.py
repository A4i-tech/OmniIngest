"""OCR engine and factory implementations."""

from __future__ import annotations

import asyncio
import io
import zipfile
from string import Template
from typing import TYPE_CHECKING

import pymupdf
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
from pydantic_ai import BinaryContent
from sarvamai import SarvamAI

from omni_ingest.parser import read_resources

from ..core.config import settings
from ..core.model import AgentMixin, ByteContent, ResolvedResource
from ..core.ocr import Ocr, OcrFactory, OcrOutputFormat, simple

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext


class _DefaultAgentMixin(BaseModel, AgentMixin):
    """Combines with `BaseModel` so `AgentMixin`'s fields resolve to real defaults."""


BASE_LLM_PROMPT = Template("Transcribe this document to $format, preserving headings, lists, and tables. Return only the $format. Do not wrap the response in a code fence.")
SRC_LANG_LLM_PROMPT = Template("The document is in $lang language.")
DST_LANG_LLM_PROMPT = Template("Translate it to $lang language.")

def build_llm_prompt(out_fmt: OcrOutputFormat, src_lang: str | None, dst_lang: str | None):
    return " ".join(seg for seg in (
        BASE_LLM_PROMPT.substitute(format=out_fmt.label),
        SRC_LANG_LLM_PROMPT.substitute(lang=src_lang) if src_lang is not None else None,
        DST_LANG_LLM_PROMPT.substitute(lang=dst_lang) if dst_lang is not None else None,
    ) if seg is not None)


def by_page(ctx: IngestionContext[ResolvedResource], out_fmt: OcrOutputFormat, parent: Ocr, dpi: int = 200, text_layer_threshold: int = 20) -> Ocr:
    def render_pdf_pages(content: bytes) -> list[tuple[str, bool, bytes | None]]:
        pages = []
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            for page in pdf.pages():
                assert isinstance(page, pymupdf.Page)
                text = page.get_text().strip()
                needs_ocr = len(text) < text_layer_threshold
                image = page.get_pixmap(dpi=dpi, alpha=False).tobytes("png") if needs_ocr else None
                pages.append((text, needs_ocr, image))
        return pages

    async def ocr(item: ByteContent) -> ByteContent:
        if (await item.content_type(ctx)) != "application/pdf":
            return await parent(item)

        pages = await asyncio.to_thread(render_pdf_pages, await item.content(ctx))

        async def resolve(text: str, needs_ocr: bool, image: bytes | None) -> str:
            if not needs_ocr or image is None:
                return text
            result = await parent(ByteContent.from_bytes(image, "image/png"))
            return (await result.content(ctx)).decode(result.content_encoding)

        content = "\n\n".join(await asyncio.gather(*(resolve(*page) for page in pages)))
        return ByteContent.from_bytes(content.encode("utf-8"), out_fmt.mime_type)

    return ocr


def llm_ocr_builder(ctx: IngestionContext[ResolvedResource], out_fmt: OcrOutputFormat, src_lang: str | None, dst_lang: str | None) -> Ocr:
    """OCR via a vision LLM. Sends the document as-is in one call."""
    agent = _DefaultAgentMixin()._agent(ctx)

    async def ocr(item: ByteContent) -> ByteContent:
        content_type = await item.content_type(ctx)
        result = await agent.run([build_llm_prompt(out_fmt, src_lang, dst_lang), BinaryContent(data=await item.content(ctx), media_type=content_type)])
        return ByteContent.from_bytes(result.output.encode("utf-8"), out_fmt.mime_type)

    return ocr


def sarvam_ocr_builder(ctx: IngestionContext[ResolvedResource], out_fmt: OcrOutputFormat, src_lang: str | None, dst_lang: str | None) -> Ocr:
    if out_fmt == OcrOutputFormat.HTML:
        output_format = "html"
    elif out_fmt == OcrOutputFormat.JSON:
        output_format = "json"
    elif out_fmt == OcrOutputFormat.MARKDOWN:
        output_format = "md"
    else:
        raise ValueError(f"SarvamAI does not support output format {out_fmt}")

    if dst_lang is not None:
        raise ValueError("SarvamAI does not support a destination language option")

    client = SarvamAI()
    async def ocr(image: ByteContent) -> ByteContent:
        data = await image.content(ctx)
        job = client.doc_ai.digitise(file=[data], output_format=output_format, **({"language": src_lang} if src_lang is not None else {}))
        while True:
            st = client.doc_ai.get_status(job_id=job.job_id)
            print(f"status={st.status} pages={st.usage.pages_processed}/{st.usage.pages_total}")
            if st.status.lower() in {"completed", "partially_completed", "failed", "rejected"}:
                break
            await asyncio.sleep(5)

        if st.status not in {"completed", "partially_completed"}:
            raise RuntimeError(f"SarvamAI job terminated unexpectedly ({st.status})")

        dl = client.doc_ai.get_download_url(job_id=job.job_id)
        resources = [r async for r in read_resources(dl.url, headers=dl.headers)]
        if len(resources) != 1:
            raise RuntimeError(f"SarvamAI returned unexpected number of resources ({len(resources)})")

        with zipfile.ZipFile(io.BytesIO(await resources[0].content(ctx))) as output:
            return ByteContent.from_bytes(output.read(f"upload/upload.{output_format}"), out_fmt.mime_type)
    return ocr


def azure_di_ocr_builder(ctx: IngestionContext[ResolvedResource], out_fmt: OcrOutputFormat, src_lang: str | None, dst_lang: str | None) -> Ocr:
    if out_fmt == OcrOutputFormat.MARKDOWN:
        output_content_format = DocumentContentFormat.MARKDOWN
    elif out_fmt == OcrOutputFormat.TEXT:
        output_content_format = DocumentContentFormat.TEXT
    else:
        raise ValueError(f"Invalid output format '{out_fmt.value}', expected one of: {', '.join(DocumentContentFormat)}")

    if src_lang is not None:
        raise ValueError("Document Intelligence does not support a source language option")
    if dst_lang is not None:
        raise ValueError("Document Intelligence does not support a destination language option")
    if settings.documentintelligence_endpoint is None or settings.documentintelligence_api_key is None:
        raise ValueError("DOCUMENTINTELLIGENCE_ENDPOINT/DOCUMENTINTELLIGENCE_API_KEY are not configured")

    endpoint = str(settings.documentintelligence_endpoint)
    credential = AzureKeyCredential(settings.documentintelligence_api_key.get_secret_value())

    async def extract_page(image: ByteContent) -> ByteContent:
        async with DocumentIntelligenceClient(endpoint=endpoint, credential=credential) as client:
            poller = await client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(bytes_source=await image.content(ctx)), output_content_format=output_content_format)
            result = await poller.result()
        return ByteContent.from_bytes(result.content.encode("utf-8"), out_fmt.mime_type)

    return by_page(ctx, out_fmt, extract_page)


def default_factory() -> OcrFactory:
    return simple({
        "azure_di": azure_di_ocr_builder,
        "llm": llm_ocr_builder,
        "sarvam": sarvam_ocr_builder,
    })
