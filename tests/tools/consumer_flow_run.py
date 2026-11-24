"""Run SEEDS-/Shiksha-shaped flows (page_chunking -> rasterize 180 dpi) on a local PDF and print numbers.

Usage: python tests/tools/consumer_flow_run.py <pdf> [--max-pages N]
Local-only helper; prints counts and timings, writes nothing.
"""

import argparse
import asyncio
import io
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader, PdfWriter

from omni_ingest.core.model import ByteContent, ResolvedResource
from omni_ingest.core.pipeline import IngestionContext, create_pipeline_from_config
from omni_ingest.port.metadata_store import NullMetadataStore

YAML = """
pipeline_id: shiksha_shaped
steps:
- agent: page_chunking
  config: {mode: pdf}
- agent: rasterize
  config:
    foreach: .items[]
    input: '[.id]'
    dpi: 180
"""


async def main(pdf: Path, max_pages: int | None) -> None:
    data = pdf.read_bytes()
    if max_pages:
        writer = PdfWriter()
        for page in PdfReader(io.BytesIO(data)).pages[:max_pages]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        data = buf.getvalue()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "p.yaml"
        cfg.write_text(YAML)
        runner = create_pipeline_from_config(cfg)
        ctx = IngestionContext(tenant_id=uuid4(), store=NullMetadataStore(), resource=ResolvedResource(uri=pdf.name, content=ByteContent.from_bytes(data, content_type="application/pdf")))
        start = time.perf_counter()
        results = await runner.run(ctx)
        elapsed = time.perf_counter() - start
        sizes = [len(await i.content.read(ctx)) for i in ctx.items]
        print(f"{pdf.name}: status={results[-1].status.value} items={len(ctx.items)} seconds={elapsed:.1f} mb_out={sum(sizes) / 1e6:.1f} max_item_mb={max(sizes, default=0) / 1e6:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--max-pages", type=int)
    a = ap.parse_args()
    asyncio.run(main(a.pdf, a.max_pages))
