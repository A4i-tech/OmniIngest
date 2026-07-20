"""OCR engine implementations for PDF page image to Markdown extraction."""

from __future__ import annotations

import asyncio
import base64
import glob
import tempfile
import zipfile
from pathlib import Path

import httpx
from sarvamai import AsyncSarvamAI


class _HttpOcrEngine:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def __aenter__(self) -> _HttpOcrEngine:
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

    async def __aenter__(self) -> SarvamOcrEngine:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

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


DEFAULT_VISION_OCR_PROMPT = "Transcribe this page to Markdown, preserving headings, lists, and tables. Return only the Markdown."


class AzureOpenAIOcrEngine(_HttpOcrEngine):
    """A vision model hosted on Azure OpenAI, used for OCR via chat completions.

    Calls the deployment's chat-completions endpoint with the page image inline,
    the same request shape as any Azure OpenAI vision model (gpt-4o, gpt-5.x).
    """

    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str = "2025-03-01-preview", prompt: str = DEFAULT_VISION_OCR_PROMPT, timeout: float = 120):
        base_url = endpoint.rstrip("/") + f"/openai/deployments/{deployment}"
        super().__init__(httpx.AsyncClient(base_url=base_url, headers={"api-key": api_key}, timeout=timeout))
        self.api_version = api_version
        self.prompt = prompt

    async def extract(self, image: bytes) -> str:
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image).decode()}"}},
                ],
            }],
        }
        response = await self.client.post("/chat/completions", params={"api-version": self.api_version}, json=payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


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


# Retained for backwards compatibility with existing configs/tests.
DEFAULT_LLAMA_CPP_PROMPT = DEFAULT_VISION_OCR_PROMPT


class LlamaCppOcrEngine(_HttpOcrEngine):
    """A vision model served by llama.cpp's OpenAI-compatible chat endpoint.

    `prompt` is model-specific: general-purpose VLMs follow the default
    freeform instruction, but task-tuned checkpoints (e.g. surya-ocr-2) only
    respond correctly to their own training-time prompt strings.
    """

    def __init__(self, base_url: str, model: str | None = None, prompt: str = DEFAULT_LLAMA_CPP_PROMPT):
        super().__init__(httpx.AsyncClient(base_url=base_url, timeout=120))
        self.model = model
        self.prompt = prompt

    async def extract(self, image: bytes) -> str:
        payload = {
            "model": self.model or "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image).decode()}"}},
                    {"type": "text", "text": self.prompt},
                ],
            }],
        }
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
