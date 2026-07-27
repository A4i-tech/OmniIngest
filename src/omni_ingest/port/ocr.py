"""OCR engine implementations for document/page-image to Markdown extraction."""

from __future__ import annotations

import asyncio
import base64
import glob
import tempfile
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import httpx
import pymupdf
from pydantic_ai import Agent, BinaryContent
from sarvamai import AsyncSarvamAI

from ..core.config import settings
from ..core.model import ByteContent, ResolvedResource
from ..core.ocr import Ocr, OcrFactory, build_simple_factory

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext


class _HttpOcrEngine:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def __aenter__(self) -> Self:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.client.__aexit__(exc_type, exc, tb)


class SarvamOcrEngine:
    """Sarvam document-intelligence OCR, via the async job API (upload -> poll -> download).

    Each page is submitted as its own single-image job; Sarvam has no
    synchronous single-image OCR endpoint.
    """

    def __init__(self, api_key: str, language: str = "hi-IN", poll_interval: float = 2.0, timeout: float = 300.0):
        self.client = AsyncSarvamAI(api_subscription_key=api_key)
        self.language = language
        self.poll_interval = poll_interval
        self.timeout = timeout

    async def extract(self, image: bytes) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image_path.write_bytes(image)

            job = await self.client.document_intelligence.create_job(language=self.language, output_format="md")
            await job.upload_file(str(image_path))
            await job.start()
            status = await job.wait_until_complete(poll_interval=self.poll_interval, timeout=self.timeout)
            if status.job_state not in ("Completed", "PartiallyCompleted"):
                raise RuntimeError(f"Sarvam OCR job {job.job_id} failed: {status.job_state}")

            zip_path = Path(tmp) / "output.zip"
            await job.download_output(str(zip_path))
            extract_dir = Path(tmp) / "output"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            md_files = sorted(glob.glob(str(extract_dir / "**" / "*.md"), recursive=True))
            return "\n\n".join(Path(f).read_text(encoding="utf-8") for f in md_files)


class AzureDocIntelOcrEngine(_HttpOcrEngine):
    """Azure Document Intelligence OCR via the prebuilt-layout model.

    Uses the async analyze API (submit -> 202 -> poll Operation-Location) and
    requests `outputContentFormat=markdown`, so the service itself returns
    structure-preserving Markdown with tables.
    """

    def __init__(self, endpoint: str, api_key: str, model: str = "prebuilt-layout", api_version: str = "2024-11-30", poll_interval: float = 2.0, timeout: float = 300.0):
        super().__init__(httpx.AsyncClient(base_url=endpoint, headers={"Ocp-Apim-Subscription-Key": api_key}, timeout=60))
        self.model = model
        self.api_version = api_version
        self.poll_interval = poll_interval
        self.timeout = timeout

    async def extract(self, image: bytes) -> str:
        submit = await self.client.post(
            f"/documentintelligence/documentModels/{self.model}:analyze",
            params={"api-version": self.api_version, "outputContentFormat": "markdown"},
            content=image,
            headers={"Content-Type": "application/octet-stream"},
        )
        submit.raise_for_status()
        operation_url = submit.headers["Operation-Location"]

        deadline = self.timeout
        elapsed = 0.0
        while True:
            poll = await self.client.get(operation_url)
            poll.raise_for_status()
            body = poll.json()
            status = body.get("status")
            if status == "succeeded":
                return str(body["analyzeResult"]["content"])
            if status in ("failed", "canceled"):
                raise RuntimeError(f"Azure Document Intelligence job {status}: {body.get('error')}")
            if elapsed >= deadline:
                raise TimeoutError(f"Azure Document Intelligence job timed out after {deadline}s")
            await asyncio.sleep(self.poll_interval)
            elapsed += self.poll_interval


DEFAULT_VISION_OCR_PROMPT = "Transcribe this document to Markdown, preserving headings, lists, and tables. Return only the Markdown."


class AzureMlOcrEngine(_HttpOcrEngine):
    """A custom OCR container deployed to an Azure ML managed online endpoint (BYOC).

    Azure ML does not fix the request/response shape for a custom container —
    the deployment's own server defines it. This engine posts to the endpoint's
    scoring URI (from `az ml online-endpoint show -n <name> --query scoring_uri`)
    with key-based auth, and assumes the container's scoring route implements:

        POST <scoring_uri>
        Authorization: Bearer <key>
        request  body: {"image": "<base64-encoded PNG>"}
        response body: {"markdown": "<extracted markdown>"}

    `deployment` optionally pins a specific deployment behind the endpoint via
    the azureml-model-deployment header (useful for blue/green rollouts).
    """

    def __init__(self, scoring_uri: str, api_key: str, deployment: str | None = None, timeout: float = 120):
        headers = {"Authorization": f"Bearer {api_key}"}
        if deployment:
            headers["azureml-model-deployment"] = deployment
        super().__init__(httpx.AsyncClient(headers=headers, timeout=timeout))
        self.scoring_uri = scoring_uri

    async def extract(self, image: bytes) -> str:
        response = await self.client.post(self.scoring_uri, json={"image": base64.b64encode(image).decode()})
        response.raise_for_status()
        return str(response.json()["markdown"])


def by_page(extract_image: Callable[[bytes], Awaitable[str]], dpi: int = 200, ocr_scanned: bool = True, text_layer_threshold: int = 20) -> Ocr:
    """Wrap a page-image extractor (`extract(png_bytes) -> markdown`) into a full `Ocr`.

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
        content_type = item.metadata.get("content_type", "")
        if content_type == "application/pdf":
            pages = await asyncio.to_thread(render_pdf_pages, item.raw_content)

            async def resolve(text: str, needs_ocr: bool, image: bytes | None) -> str:
                return await extract_image(image) if needs_ocr and image is not None else text

            markdown = "\n\n".join(await asyncio.gather(*(resolve(*page) for page in pages)))
        elif content_type.startswith("image/"):
            markdown = await extract_image(item.raw_content)
        else:
            raise ValueError(f"Unsupported OCR input content type: {content_type!r}")

        return ByteContent(raw_content=markdown.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return ocr


def llm_ocr(agent: Agent[Any, str], prompt: str = DEFAULT_VISION_OCR_PROMPT) -> Ocr:
    """OCR via a vision LLM, driven by a pydantic-ai agent vended by OcrAgent.

    Sends the document as-is (PDF or image) in one call — any OpenAI-compatible
    vision model (Azure OpenAI, a llama.cpp server, gpt-4o, etc.) natively reads
    PDF input, so no page-rendering preprocessing is needed here.
    """

    async def ocr(item: ByteContent) -> ByteContent:
        content_type = item.metadata.get("content_type", "application/octet-stream")
        result = await agent.run([prompt, BinaryContent(data=item.raw_content, media_type=content_type)])
        return ByteContent(raw_content=result.output.encode("utf-8"), content_encoding="utf-8", metadata={"content_type": "text/markdown"})

    return ocr


def sarvam_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    if settings.sarvam_api_key is None:
        raise ValueError("SARVAM_API_KEY is not configured")
    engine = SarvamOcrEngine(api_key=settings.sarvam_api_key.get_secret_value(), language=src_lang or "hi-IN")
    return by_page(engine.extract)


def azure_di_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    if settings.azure_di_ocr_endpoint is None or settings.azure_di_ocr_key is None:
        raise ValueError("AZURE_DI_OCR_ENDPOINT/AZURE_DI_OCR_KEY are not configured")
    engine = AzureDocIntelOcrEngine(endpoint=str(settings.azure_di_ocr_endpoint), api_key=settings.azure_di_ocr_key.get_secret_value())
    return by_page(engine.extract)


def azure_ml_ocr_builder(ctx: IngestionContext[ResolvedResource], src_lang: str | None, dst_lang: str | None) -> Ocr:
    if settings.azure_ml_ocr_scoring_uri is None or settings.azure_ml_ocr_key is None:
        raise ValueError("AZURE_ML_OCR_SCORING_URI/AZURE_ML_OCR_KEY are not configured")
    engine = AzureMlOcrEngine(scoring_uri=str(settings.azure_ml_ocr_scoring_uri), api_key=settings.azure_ml_ocr_key.get_secret_value())
    return by_page(engine.extract)


def default_factory() -> OcrFactory:
    """Default OCR factory. `llm` is vended by `OcrAgent` itself since it needs agent construction."""
    return build_simple_factory({
        "sarvam": sarvam_ocr_builder,
        "azure_di": azure_di_ocr_builder,
        "azure_ml": azure_ml_ocr_builder,
    })
