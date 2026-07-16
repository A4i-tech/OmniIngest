"""OCR engine implementations for PDF page image to Markdown extraction."""

from __future__ import annotations

import base64

import httpx


class _HttpOcrEngine:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def __aenter__(self) -> _HttpOcrEngine:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.client.__aexit__(exc_type, exc, tb)


class SarvamOcrEngine(_HttpOcrEngine):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(httpx.AsyncClient(base_url=base_url, headers={"api-subscription-key": api_key}, timeout=60))

    async def extract(self, image: bytes) -> str:
        response = await self.client.post("/v1/ocr", files={"file": ("page.png", image, "image/png")}, data={"output_format": "markdown"})
        response.raise_for_status()
        return str(response.json()["markdown"])


class AzureCvOcrEngine(_HttpOcrEngine):
    """Azure AI Vision Read OCR (imageanalysis:analyze, features=read)."""

    def __init__(self, endpoint: str, api_key: str):
        super().__init__(httpx.AsyncClient(base_url=endpoint, headers={"Ocp-Apim-Subscription-Key": api_key}, timeout=60))

    async def extract(self, image: bytes) -> str:
        response = await self.client.post(
            "/computervision/imageanalysis:analyze",
            params={"api-version": "2024-02-01", "features": "read"},
            content=image,
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        lines = [str(line["text"]) for block in response.json()["readResult"]["blocks"] for line in block["lines"]]
        return "\n".join(lines)


class AzureMlOcrEngine(_HttpOcrEngine):
    """A custom OCR model deployed as an Azure ML online endpoint."""

    def __init__(self, endpoint: str, api_key: str):
        super().__init__(httpx.AsyncClient(base_url=endpoint, headers={"Authorization": f"Bearer {api_key}"}, timeout=120))

    async def extract(self, image: bytes) -> str:
        response = await self.client.post("/score", json={"image": base64.b64encode(image).decode()})
        response.raise_for_status()
        return str(response.json()["markdown"])


class LlamaCppOcrEngine(_HttpOcrEngine):
    """A local vision model served by llama.cpp's OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, model: str | None = None):
        super().__init__(httpx.AsyncClient(base_url=base_url, timeout=120))
        self.model = model

    async def extract(self, image: bytes) -> str:
        payload = {
            "model": self.model or "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this page to Markdown, preserving headings, lists, and tables. Return only the Markdown."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image).decode()}"}},
                ],
            }],
        }
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
