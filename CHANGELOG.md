# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- liblouis (BRF support) is now an optional extra (`braille`) instead of a core dependency. `pip install omni-ingest` no longer fails on Linux and Windows. Without liblouis, BRF reading and writing raise a clear `RuntimeError`; see the Braille section in `docs/advanced.md`.
- The BRF test fixture is now generated from synthetic text.

### Added

- `THIRD_PARTY_LICENSES.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, pull request template and dependabot config.
- Characterization tests for PDF handling (image extraction, vector figures, redact and caption modes, rasterize, PNG output) and consumer-shaped pipeline flows.

### Removed

- `.pre-commit-config.yaml`. Linting and type checks run in CI (`ruff`, `mypy`).
