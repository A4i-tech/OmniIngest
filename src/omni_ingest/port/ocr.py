"""OCR engine and factory implementations."""

from __future__ import annotations

import asyncio
import base64
import io
import zipfile
from collections.abc import AsyncIterator
from string import Template

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from mistralai.client import Mistral
from mistralai.client import models as mistral_models
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
        result = await agent.run(instructions=instructions, user_prompt=(BinaryContent(data=await item.read(params.ctx), media_type=content_type),))
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
        data = await item.read(params.ctx)
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

        with zipfile.ZipFile(io.BytesIO(await resources[0].content.read(params.ctx))) as output:
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

    async def one(client: DocumentIntelligenceClient, item: ByteContent, sem: asyncio.Semaphore):
        async with sem:
            poller = await client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=await item.read(params.ctx)),
                locale=params.src_lang, output_content_format=output_content_format
            )
            result = await poller.result()
        return item, ByteContent.from_bytes(result.content.encode("utf-8"), params.format.mime_type)

    endpoint = str(settings.documentintelligence_endpoint)
    credential = AzureKeyCredential(settings.documentintelligence_api_key.get_secret_value())
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
    async def ocr(items: AsyncIterator[ByteContent]):
        async with client:
            sem = asyncio.Semaphore(params.concurrency)
            for task in asyncio.as_completed([one(client, item, sem) async for item in items]):
                yield await task
    return ocr


def mistral_ocr_builder(params: OcrBuilderParams) -> Ocr:
    if params.format not in {OcrOutputFormat.MARKDOWN}:
        raise ValueError(f"Mistral Document AI only returns Markdown, got '{params.format.value}'")
    if params.dst_lang is not None:
        raise ValueError("Mistral Document AI does not support a destination language option")
    if settings.mistral_ocr_api_key is None:
        raise ValueError("MISTRAL_OCR_API_KEY is not configured")

    key = settings.mistral_ocr_api_key.get_secret_value()

    async def one(client: Mistral, item: ByteContent, sem: asyncio.Semaphore):
        async with sem:
            content_type = await item.content_type(params.ctx)
            uri = f"data:{content_type};base64,{base64.b64encode(await item.read(params.ctx)).decode()}"
            document = mistral_models.ImageURLChunk(image_url=uri) if content_type.startswith("image/") else mistral_models.DocumentURLChunk(document_url=uri)
            result = await client.ocr.process_async(model=settings.mistral_ocr_model, document=document)
        if not result.pages:
            raise RuntimeError("Mistral Document AI returned no pages, cannot silently produce empty OCR output")
        text = "\n\n".join(page.markdown for page in result.pages)
        changed = ByteContent.from_bytes(text.encode("utf-8"), params.format.mime_type)
        return item, changed

    async def ocr(items: AsyncIterator[ByteContent]):
        client = Mistral(api_key=key)
        async with client:
            sem = asyncio.Semaphore(params.concurrency)
            for task in asyncio.as_completed([one(client, item, sem) async for item in items]):
                yield await task
    return ocr


def default_factory() -> OcrFactory:
    return simple({
        "azure_di": azure_di_ocr_builder,
        "llm": llm_ocr_builder,
        "mistral": mistral_ocr_builder,
        "sarvam": sarvam_ocr_builder,
    })
