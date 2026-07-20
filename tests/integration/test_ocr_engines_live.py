"""Live OCR-engine tests against real services.

llama_cpp: requires a running OpenAI-compatible server (LLAMA_CPP_OCR_TEST_URL,
default http://127.0.0.1:8123 — see offlineOcrBench/start_surya_server.ps1).
sarvam: requires SARVAM_API_KEY.
azure_di: requires AZURE_DI_OCR_ENDPOINT + AZURE_DI_OCR_KEY.
azure_openai: requires AZURE_OPENAI_OCR_ENDPOINT + AZURE_OPENAI_OCR_KEY + AZURE_OPENAI_OCR_DEPLOYMENT.
azure_ml: requires AZURE_ML_OCR_SCORING_URI + AZURE_ML_OCR_KEY (custom-container BYOC endpoint).
"""

import os

import httpx
import pymupdf
import pytest

from omni_ingest.port.ocr import AzureDocIntelOcrEngine, AzureMlOcrEngine, AzureOpenAIOcrEngine, LlamaCppOcrEngine, SarvamOcrEngine

SURYA_HIGH_ACCURACY_BBOX_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
)


@pytest.fixture
def middle_page_image():
    with pymupdf.open("tests/resources/textbook-scan-middle-pages.pdf") as pdf:
        return pdf[0].get_pixmap(dpi=200, alpha=False).tobytes("png")


async def test_llama_cpp_ocr_engine_extracts_real_page(middle_page_image):
    base_url = os.environ.get("LLAMA_CPP_OCR_TEST_URL", "http://127.0.0.1:8123")
    try:
        async with httpx.AsyncClient(timeout=2) as probe:
            await probe.get(f"{base_url}/health")
    except httpx.HTTPError:
        pytest.skip(f"No llama.cpp server reachable at {base_url}")

    engine = LlamaCppOcrEngine(base_url=base_url, model="datalab-to/surya-ocr-2", prompt=SURYA_HIGH_ACCURACY_BBOX_PROMPT)

    async with engine:
        markdown = await engine.extract(middle_page_image)

    assert len(markdown) > 100
    assert "data-bbox" in markdown


async def test_sarvam_ocr_engine_extracts_real_page(middle_page_image):
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        pytest.skip("SARVAM_API_KEY not configured")

    engine = SarvamOcrEngine(api_key=key, language="hi-IN", poll_interval=3, timeout=180)

    async with engine:
        markdown = await engine.extract(middle_page_image)

    assert len(markdown) > 0


async def test_azure_di_ocr_engine_extracts_real_page(middle_page_image):
    endpoint = os.environ.get("AZURE_DI_OCR_ENDPOINT")
    key = os.environ.get("AZURE_DI_OCR_KEY")
    if not endpoint or not key:
        pytest.skip("AZURE_DI_OCR_ENDPOINT / AZURE_DI_OCR_KEY not configured")

    engine = AzureDocIntelOcrEngine(endpoint=endpoint, api_key=key, poll_interval=2, timeout=180)

    async with engine:
        markdown = await engine.extract(middle_page_image)

    assert len(markdown) > 0
    assert "<table>" in markdown


async def test_azure_openai_ocr_engine_extracts_real_page(middle_page_image):
    endpoint = os.environ.get("AZURE_OPENAI_OCR_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_OCR_KEY")
    deployment = os.environ.get("AZURE_OPENAI_OCR_DEPLOYMENT")
    if not endpoint or not key or not deployment:
        pytest.skip("AZURE_OPENAI_OCR_ENDPOINT / AZURE_OPENAI_OCR_KEY / AZURE_OPENAI_OCR_DEPLOYMENT not configured")

    engine = AzureOpenAIOcrEngine(endpoint=endpoint, api_key=key, deployment=deployment, timeout=180)

    async with engine:
        markdown = await engine.extract(middle_page_image)

    assert len(markdown) > 0


async def test_azure_ml_ocr_engine_extracts_real_page(middle_page_image):
    scoring_uri = os.environ.get("AZURE_ML_OCR_SCORING_URI")
    key = os.environ.get("AZURE_ML_OCR_KEY")
    if not scoring_uri or not key:
        pytest.skip("AZURE_ML_OCR_SCORING_URI / AZURE_ML_OCR_KEY not configured")

    engine = AzureMlOcrEngine(scoring_uri=scoring_uri, api_key=key, deployment=os.environ.get("AZURE_ML_OCR_DEPLOYMENT"), timeout=180)

    async with engine:
        markdown = await engine.extract(middle_page_image)

    assert len(markdown) > 0
