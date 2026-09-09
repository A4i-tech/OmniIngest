"""OCR engine and factory implementations."""

from __future__ import annotations

import asyncio
import io
import zipfile
from base64 import b64encode
from collections.abc import AsyncIterator
from string import Template
from typing import Any

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeOutputOption, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from sarvamai import SarvamAI

from omni_ingest.parser import read_resources

from ..core.config import settings
from ..core.model import AgentMixin, ByteContent
from ..core.ocr import Ocr, OcrBuilderParams, OcrFactory, OcrOutputFormat, simple


class _DefaultAgentMixin(BaseModel, AgentMixin):
    """Combines with `BaseModel` so `AgentMixin`'s fields resolve to real defaults."""


BASE_LLM_PROMPT = Template("The user will provide a documnt. Transcribe the document into $format format, preserving headings, lists, and tables. Return only the $format. Do not wrap the response in a code fence.")
SRC_LANG_LLM_PROMPT = Template("The document is in $lang language.")
DST_LANG_LLM_PROMPT = Template("Translate it to $lang language.")


def llm_ocr_builder(params: OcrBuilderParams) -> Ocr:
    """OCR via a vision LLM. Sends the document as-is in one call."""

    instructions = " ".join(seg for seg in (
        BASE_LLM_PROMPT.substitute(format=params.format.label),
        SRC_LANG_LLM_PROMPT.substitute(lang=params.src_lang) if params.src_lang is not None else None,
        DST_LANG_LLM_PROMPT.substitute(lang=params.dst_lang) if params.dst_lang is not None else None,
    ) if seg is not None)

    async def one(agent: Agent[object, str], item: ByteContent, sem: asyncio.Semaphore):
        content_type = await item.content_type(params.ctx)
        result = await agent.run(instructions=instructions, user_prompt=(BinaryContent(data=await item.content(params.ctx), media_type=content_type),))
        return item, ByteContent.from_bytes(result.output.encode("utf-8"), params.format.mime_type)

    agent = _DefaultAgentMixin()._agent(params.ctx)
    async def ocr(items: AsyncIterator[ByteContent]):
        async with agent:
            sem = asyncio.Semaphore(params.concurrency)
            for task in asyncio.as_completed([one(agent, item, sem) async for item in items]):
                yield await task
    return ocr


def sarvam_ocr_builder(params: OcrBuilderParams) -> Ocr:
    if params.format == OcrOutputFormat.HTML:
        output_format = "html"
    elif params.format == OcrOutputFormat.JSON:
        output_format = "json"
    elif params.format == OcrOutputFormat.MARKDOWN:
        output_format = "md"
    else:
        raise ValueError(f"SarvamAI does not support output format {params.format}")

    if params.dst_lang is not None:
        raise ValueError("SarvamAI does not support a destination language option")

    async def one(client: SarvamAI, item: ByteContent, sem: asyncio.Semaphore):
        data = await item.content(params.ctx)
        job = client.doc_ai.digitise(file=[data], output_format=output_format, **({"language": params.src_lang} if params.src_lang is not None else {}))
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

        with zipfile.ZipFile(io.BytesIO(await resources[0].content(params.ctx))) as output:
            return item, ByteContent.from_bytes(output.read(f"upload/upload.{output_format}"), params.format.mime_type)

    client = SarvamAI()
    async def ocr(items: AsyncIterator[ByteContent]):
        sem = asyncio.Semaphore(params.concurrency)
        for task in asyncio.as_completed([one(client, item, sem) async for item in items]):
            yield await task
    return ocr


def azure_di_ocr_builder(params: OcrBuilderParams) -> Ocr:
    if params.format == OcrOutputFormat.MARKDOWN:
        output_content_format = DocumentContentFormat.MARKDOWN
    elif params.format == OcrOutputFormat.TEXT:
        output_content_format = DocumentContentFormat.TEXT
    else:
        raise ValueError(f"Invalid output format '{params.format.value}', expected one of: {', '.join(DocumentContentFormat)}")

    if params.dst_lang is not None:
        raise ValueError("Document Intelligence does not support a destination language option")
    if settings.documentintelligence_endpoint is None or settings.documentintelligence_api_key is None:
        raise ValueError("DOCUMENTINTELLIGENCE_ENDPOINT/DOCUMENTINTELLIGENCE_API_KEY are not configured")

    async def figures(client: DocumentIntelligenceClient, result, result_id: str | None) -> list[dict[str, Any]]:
        """Collect the figures the layout model found, with their cropped images."""
        collected: list[dict[str, Any]] = []
        for figure in result.figures or []:
            region = (figure.get("boundingRegions") or [{}])[0]
            entry: dict[str, Any] = {
                "id": figure.get("id"),
                "page": region.get("pageNumber"),
                "caption": (figure.get("caption") or {}).get("content"),
                "bbox": region.get("polygon"),
            }
            if result_id and figure.get("id"):
                try:
                    stream = await client.get_analyze_result_figure(
                        model_id=result.model_id, result_id=result_id, figure_id=figure["id"]
                    )
                    entry["image"] = b64encode(b"".join([chunk async for chunk in stream])).decode("ascii")
                    entry["content_type"] = "image/png"
                except Exception:
                    # A crop is a bonus; losing one must not lose the page text.
                    entry["crop_failed"] = True
            collected.append(entry)
        return collected

    async def one(client: DocumentIntelligenceClient, item: ByteContent, sem: asyncio.Semaphore):
        async with sem:
            poller = await client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=await item.content(params.ctx)),
                locale=params.src_lang, output_content_format=output_content_format,
                output=[AnalyzeOutputOption.FIGURES],
            )
            result = await poller.result()
            found = await figures(client, result, poller.details.get("operation_id"))
        changed = ByteContent.from_bytes(result.content.encode("utf-8"), params.format.mime_type)
        if found:
            changed.metadata["figures"] = found
        if failed := sum(1 for f in found if f.get("crop_failed")):
            changed.metadata["figures_failed"] = failed
        return item, changed

    endpoint = str(settings.documentintelligence_endpoint)
    credential = AzureKeyCredential(settings.documentintelligence_api_key.get_secret_value())
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
    async def ocr(items: AsyncIterator[ByteContent]):
        async with client:
            sem = asyncio.Semaphore(params.concurrency)
            for task in asyncio.as_completed([one(client, item, sem) async for item in items]):
                yield await task
    return ocr


def default_factory() -> OcrFactory:
    return simple({
        "azure_di": azure_di_ocr_builder,
        "llm": llm_ocr_builder,
        "sarvam": sarvam_ocr_builder,
    })
