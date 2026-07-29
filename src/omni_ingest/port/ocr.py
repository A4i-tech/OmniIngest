"""OCR engine builders for document/page-image to Markdown extraction."""

from __future__ import annotations

import asyncio
import base64
import glob
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pymupdf
from pydantic import BaseModel
from pydantic_ai import BinaryContent
from sarvamai import AsyncSarvamAI

from ..core.config import settings
from ..core.model import AgentMixin, ByteContent, ResolvedResource
from ..core.ocr import Ocr, OcrFactory, build_simple_factory

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext


class _DefaultAgentMixin(BaseModel, AgentMixin):
    """Combines with `BaseModel` so `AgentMixin`'s fields resolve to real defaults, for factory
    contexts that have no bound `Step` instance to vend model/toolset config from."""


DEFAULT_VISION_OCR_PROMPT = "Transcribe this document to Markdown, preserving headings, lists, and tables. Return only the Markdown."


def by_page(extract_page: Ocr, dpi: int = 200, ocr_scanned: bool = True, text_layer_threshold: int = 20) -> Ocr:
    """Wrap a single-image `Ocr` into a full-document `Ocr`.

    PDFs are rendered page-by-page, skipping pages whose extractable text is
    already above `text_layer_threshold`; images are passed straight through.
    Used by engines (Sarvam, Azure DI, Azure ML) that only OCR single images.
    """

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
    """OCR via a vision LLM. Sends the document as-is (PDF or image) in one call — any
    OpenAI-compatible vision model (Azure OpenAI, a llama.cpp server, gpt-4o, etc.)
    natively reads PDF input, so no page-rendering preprocessing is needed here.
    """
    agent = _DefaultAgentMixin()._agent(ctx)

    async def ocr(item: ByteContent) -> ByteContent:
        content_type = await item.content_type(ctx)
        result = await agent.run([DEFAULT_VISION_OCR_PROMPT, BinaryContent(data=await item.content(ctx), media_type=content_type)])
        return ByteContent(raw_content=result.output.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return ocr


def sarvam_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    """Sarvam document-intelligence OCR, via the async job API (upload -> poll -> download).

    Each page is submitted as its own single-image job; Sarvam has no
    synchronous single-image OCR endpoint.
    """
    if settings.sarvam_api_key is None:
        raise ValueError("SARVAM_API_KEY is not configured")
    client = AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key.get_secret_value())
    language = src_lang or "hi-IN"

    async def extract_page(image: ByteContent) -> ByteContent:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image_path.write_bytes(await image.content(ctx))

            job = await client.document_intelligence.create_job(language=language, output_format="md")
            await job.upload_file(str(image_path))
            await job.start()
            status = await job.wait_until_complete(poll_interval=2.0, timeout=300.0)
            if status.job_state not in ("Completed", "PartiallyCompleted"):
                raise RuntimeError(f"Sarvam OCR job {job.job_id} failed: {status.job_state}")

            zip_path = Path(tmp) / "output.zip"
            await job.download_output(str(zip_path))
            extract_dir = Path(tmp) / "output"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            md_files = sorted(glob.glob(str(extract_dir / "**" / "*.md"), recursive=True))
            markdown = "\n\n".join(Path(f).read_text(encoding="utf-8") for f in md_files)

        return ByteContent(raw_content=markdown.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return by_page(extract_page)


def azure_di_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    """Azure Document Intelligence OCR via the prebuilt-layout model.

    Uses the async analyze API (submit -> 202 -> poll Operation-Location) and
    requests `outputContentFormat=markdown`, so the service itself returns
    structure-preserving Markdown with tables.
    """
    if settings.azure_di_ocr_endpoint is None or settings.azure_di_ocr_key is None:
        raise ValueError("AZURE_DI_OCR_ENDPOINT/AZURE_DI_OCR_KEY are not configured")
    client = httpx.AsyncClient(base_url=str(settings.azure_di_ocr_endpoint), headers={"Ocp-Apim-Subscription-Key": settings.azure_di_ocr_key.get_secret_value()}, timeout=60)

    async def extract_page(image: ByteContent) -> ByteContent:
        submit = await client.post(
            "/documentintelligence/documentModels/prebuilt-layout:analyze",
            params={"api-version": "2024-11-30", "outputContentFormat": "markdown"},
            content=await image.content(ctx),
            headers={"Content-Type": "application/octet-stream"},
        )
        submit.raise_for_status()
        operation_url = submit.headers["Operation-Location"]

        deadline, poll_interval, elapsed = 300.0, 2.0, 0.0
        while True:
            poll = await client.get(operation_url)
            poll.raise_for_status()
            body = poll.json()
            status = body.get("status")
            if status == "succeeded":
                markdown = str(body["analyzeResult"]["content"])
                return ByteContent(raw_content=markdown.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})
            if status in ("failed", "canceled"):
                raise RuntimeError(f"Azure Document Intelligence job {status}: {body.get('error')}")
            if elapsed >= deadline:
                raise TimeoutError(f"Azure Document Intelligence job timed out after {deadline}s")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    return by_page(extract_page)


def azure_ml_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    """A custom OCR container deployed to an Azure ML managed online endpoint (BYOC).

    Azure ML does not fix the request/response shape for a custom container —
    the deployment's own server defines it. This engine posts to the endpoint's
    scoring URI (from `az ml online-endpoint show -n <name> --query scoring_uri`)
    with key-based auth, and assumes the container's scoring route implements:

        POST <scoring_uri>
        Authorization: Bearer <key>
        request  body: {"image": "<base64-encoded PNG>"}
        response body: {"markdown": "<extracted markdown>"}
    """
    if settings.azure_ml_ocr_scoring_uri is None or settings.azure_ml_ocr_key is None:
        raise ValueError("AZURE_ML_OCR_SCORING_URI/AZURE_ML_OCR_KEY are not configured")
    client = httpx.AsyncClient(headers={"Authorization": f"Bearer {settings.azure_ml_ocr_key.get_secret_value()}"}, timeout=120)
    scoring_uri = str(settings.azure_ml_ocr_scoring_uri)

    async def extract_page(image: ByteContent) -> ByteContent:
        response = await client.post(scoring_uri, json={"image": base64.b64encode(await image.content(ctx)).decode()})
        response.raise_for_status()
        markdown = str(response.json()["markdown"])
        return ByteContent(raw_content=markdown.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return by_page(extract_page)


def default_factory() -> OcrFactory:
    return build_simple_factory({
        "llm": llm_ocr_builder,
        "sarvam": sarvam_ocr_builder,
        "azure_di": azure_di_ocr_builder,
        "azure_ml": azure_ml_ocr_builder,
    })
