"""OCR engine and factory implementations."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import zipfile
from collections.abc import AsyncIterator
from string import Template
from uuid import UUID

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeOutputOption, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from mistralai.azure.client import MistralAzure
from mistralai.azure.client import models as azure_models
from mistralai.azure.client import utils as azure_utils
from mistralai.client import Mistral
from mistralai.client import models as mistral_models
from mistralai.client import utils as mistral_utils
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from sarvamai import SarvamAI

from omni_ingest.parser import read_resources

from ..core.config import settings
from ..core.model import AgentMixin, ByteContent
from ..core.ocr import Figure, Ocr, OcrBuilderParams, OcrFactory, OcrOutputFormat, simple

logger = logging.getLogger(__name__)


class _DefaultAgentMixin(BaseModel, AgentMixin):
    """Combines with `BaseModel` so `AgentMixin`'s fields resolve to real defaults."""


BASE_LLM_PROMPT = Template("The user will provide a documnt. Transcribe the document into $format format, preserving headings, lists, and tables. Return only the $format. Do not wrap the response in a code fence.")
SRC_LANG_LLM_PROMPT = Template("The document is in $lang language.")
DST_LANG_LLM_PROMPT = Template("Translate it to $lang language.")


def llm_ocr_builder(params: OcrBuilderParams) -> Ocr:
    """OCR via a vision LLM. Sends the document as-is in one call."""
    _reject_image_options(params, "llm")

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
    _reject_image_options(params, "sarvam")
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


async def _figures(client: DocumentIntelligenceClient, result, result_id: str | None) -> tuple[list[Figure], int]:
    """Collect the figures the layout model found, with their cropped images."""
    collected: list[Figure] = []
    all_figures = result.figures or []
    if not result_id:
        logger.warning("No result_id from analyze poller; dropping %d figure(s)", len(all_figures))
        return collected, len(all_figures)

    failed = 0
    for figure in all_figures:
        if not figure.id:
            failed += 1
            continue
        try:
            stream = await client.get_analyze_result_figure(model_id=result.model_id, result_id=result_id, figure_id=figure.id)
            image = b"".join([chunk async for chunk in stream])
        except Exception:
            # A crop is a bonus; losing one must not lose the page text.
            logger.warning("Failed to fetch figure %s", figure.id, exc_info=True)
            failed += 1
            continue
        region = (figure.bounding_regions or [None])[0]
        content = ByteContent.from_bytes(image, "image/png")
        content.metadata["id"] = figure.id
        content.metadata["page"] = region.page_number if region else None
        content.metadata["polygon"] = region.polygon if region else None
        collected.append(Figure(content=content, caption=figure.caption.content if figure.caption else None))
    return collected, failed


def azure_di_ocr_builder(params: OcrBuilderParams) -> Ocr:
    _reject_image_options(params, "azure_di")
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
                locale=params.src_lang, output_content_format=output_content_format,
                output=[AnalyzeOutputOption.FIGURES],
            )
            result = await poller.result()
            found, failed = await _figures(client, result, poller.details.get("operation_id"))
        changed = ByteContent.from_bytes(result.content.encode("utf-8"), params.format.mime_type)
        if found:
            changed.metadata["figures"] = found
        if failed:
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


def _reject_image_options(params: OcrBuilderParams, engine: str) -> None:
    for option in ("extract_images", "extract_header_footer"):
        if getattr(params, option):
            raise ValueError(f"OCR engine '{engine}' does not support the '{option}' option")


def _mistral_page_figures(page, params: OcrBuilderParams, seen: dict[str, UUID]) -> tuple[str, list[Figure]]:
    """Turn the images of one page into figures and swap their markdown references for `<image id/>` markers."""
    text = page.markdown
    figures: list[Figure] = []
    for image in page.images or []:
        if not image.image_base64:
            continue
        prefix, _, encoded = image.image_base64.rpartition(",")
        content_type = prefix.removeprefix("data:").removesuffix(";base64") or "image/jpeg"
        data = base64.b64decode(encoded)
        digest = hashlib.sha256(data).hexdigest() if params.deduplicate else None
        if digest is not None and digest in seen:
            figure_id = seen[digest]
        else:
            content = ByteContent.from_bytes(data, content_type)
            content.metadata["id"] = image.id
            content.metadata["page"] = page.index + 1
            content.metadata["bbox"] = [image.top_left_x, image.top_left_y, image.bottom_right_x, image.bottom_right_y]
            content.metadata["page_dimensions"] = page.dimensions.model_dump() if page.dimensions else None
            figure = Figure(content=content)
            figures.append(figure)
            figure_id = figure.id
            if digest is not None:
                seen[digest] = figure_id
        text = text.replace(f"![{image.id}]({image.id})", f'<image id="{figure_id}"/>')
    return text, figures


def _mistral_builder(params: OcrBuilderParams, *, models, utils, make_client) -> Ocr:
    """Shared Mistral Document AI logic. Only the client, its models, and its retry types differ per engine."""
    if params.format not in {OcrOutputFormat.MARKDOWN}:
        raise ValueError(f"Mistral Document AI only returns Markdown, got '{params.format.value}'")
    if params.dst_lang is not None:
        raise ValueError("Mistral Document AI does not support a destination language option")

    options: dict[str, bool | int] = {}
    if params.extract_images:
        options["include_image_base64"] = True
        if params.image_min_size is not None:
            options["image_min_size"] = params.image_min_size
    if params.extract_header_footer:
        options["extract_header"] = True
        options["extract_footer"] = True
    seen: dict[str, UUID] = {}  # figure hash -> item id, one table per step run

    async def one(client, item: ByteContent, sem: asyncio.Semaphore):
        async with sem:
            content_type = await item.content_type(params.ctx)
            uri = f"data:{content_type};base64,{base64.b64encode(await item.read(params.ctx)).decode()}"
            document = models.ImageURLChunk(image_url=uri) if content_type.startswith("image/") else models.DocumentURLChunk(document_url=uri)
            result = await client.ocr.process_async(model=settings.mistral_ocr_model, document=document, **options)
        if not result.pages:
            raise RuntimeError("Mistral Document AI returned no pages, cannot silently produce empty OCR output")
        texts: list[str] = []
        figures: list[Figure] = []
        for page in result.pages:
            text, page_figures = _mistral_page_figures(page, params, seen)
            texts.append(text)
            figures.extend(page_figures)
        changed = ByteContent.from_bytes("\n\n".join(texts).encode("utf-8"), params.format.mime_type)
        if figures:
            changed.metadata["figures"] = figures
        if params.extract_header_footer:
            changed.metadata["ocr_header"] = "\n\n".join(p.header for p in result.pages if p.header)
            changed.metadata["ocr_footer"] = "\n\n".join(p.footer for p in result.pages if p.footer)
        return item, changed

    retry_config = utils.RetryConfig(
        strategy="backoff",
        backoff=utils.BackoffStrategy(initial_interval=500, max_interval=60_000, exponent=1.5, max_elapsed_time=600_000),
        retry_connection_errors=True,
    )

    async def ocr(items: AsyncIterator[ByteContent]):
        client = make_client(retry_config)
        async with client:
            sem = asyncio.Semaphore(params.concurrency)
            for task in asyncio.as_completed([one(client, item, sem) async for item in items]):
                yield await task
    return ocr


def mistral_ocr_builder(params: OcrBuilderParams) -> Ocr:
    if settings.mistral_ocr_api_key is None:
        raise ValueError("MISTRAL_OCR_API_KEY is not configured")
    key = settings.mistral_ocr_api_key.get_secret_value()
    return _mistral_builder(params, models=mistral_models, utils=mistral_utils,
                            make_client=lambda retry: Mistral(api_key=key, retry_config=retry))


def azure_mistral_ocr_builder(params: OcrBuilderParams) -> Ocr:
    if settings.mistral_ocr_api_key is None or settings.mistral_ocr_endpoint is None:
        raise ValueError("MISTRAL_OCR_API_KEY/MISTRAL_OCR_ENDPOINT are not configured")
    key = settings.mistral_ocr_api_key.get_secret_value()
    endpoint = str(settings.mistral_ocr_endpoint)
    return _mistral_builder(params, models=azure_models, utils=azure_utils,
                            make_client=lambda retry: MistralAzure(api_key=key, server_url=endpoint, retry_config=retry))


def default_factory() -> OcrFactory:
    return simple({
        "azure_di": azure_di_ocr_builder,
        "azure_mistral": azure_mistral_ocr_builder,
        "llm": llm_ocr_builder,
        "mistral": mistral_ocr_builder,
        "sarvam": sarvam_ocr_builder,
    })
