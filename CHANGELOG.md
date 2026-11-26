# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [2.0.1] - 2026-11-26

### Changed

- PDF handling no longer uses PyMuPDF (AGPL-3.0). Rendering and page-object editing use `pypdfium2`, PNG output uses Pillow and hidden-layer lookup and redaction use `pypdf`, so the package stays MIT. All PDF code lives in `omni_ingest/pdf.py`. Behaviour differences from PyMuPDF:
  - Rotated pages report image boxes in visible coordinates.
  - Redact mode writes the `<image id=.../>` marker on one line, also removes duplicate images on the page, and edits the page content stream (text per glyph, images, shadings, vector paths fully inside the region, overlapping annotations and the page thumbnail).
  - Damaged or truncated PDFs, PDFs with a broken `startxref` and PDFs with zero pages raise a `ValueError` with a next action. PyMuPDF repaired the first two and returned no items for the last.
  - Owner-password PDFs keep their encryption when saved.
  - WebP and CMYK JPEG images in markdown output are converted to PNG.
  - Images on an optional-content layer that is switched off are skipped.
  - Tables are ruled grids of at least 3 rows and 3 columns.
  - All pdfium work runs on a single thread, because pdfium is not thread-safe. `workers` limits queued jobs, not parallel pdfium work.
- The version in `pyproject.toml` is now the released version. It previously read 0.1.2 and was rewritten from the git tag at release.
- liblouis (BRF support) is now an optional extra (`braille`) instead of a core dependency. `pip install omni-ingest` no longer fails on Linux and Windows. Without liblouis, BRF reading and writing raise a clear `RuntimeError`; see the Braille section in `docs/advanced.md`.
- The BRF test fixture is now generated from synthetic text.
- Core dependencies were upgraded to clear the advisories `pip-audit` reported (pypdf, pillow, urllib3, aiohttp, cryptography and others).

### Added

- `--max-concurrent-resources` on the CLI (default 8) and a `max_concurrent` argument on `run_flattened` limit how many resources are processed at the same time.
- A licence check in CI that fails on GPL, LGPL, AGPL and SSPL dependencies in the core install and uploads the licence report.
- `THIRD_PARTY_LICENSES.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, pull request template and dependabot config.
- Characterization tests for PDF handling (image extraction, vector figures, redact and caption modes, rasterize, PNG output) and consumer-shaped pipeline flows.
- Unit tests for the concurrency limits of the OCR, rasterize and resource runners.

### Fixed

- The `llm` and `sarvam` OCR engines now honour `concurrency`, and `RasterizeAgent` reads PDF content only while it holds a worker slot, which bounds memory on large batches.
- `run_flattened(max_concurrent=0)` hung forever. Values below 1 now raise a `ValueError`, as the CLI already did.

### Removed

- The `pymupdf` dependency.
- `.pre-commit-config.yaml`. Linting and type checks run in CI (`ruff`, `mypy`).
