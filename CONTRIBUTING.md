# Contributing to OmniIngest

Thanks for helping. OmniIngest is MIT licensed and maintained by A4I, IIIT Bangalore. By contributing you agree that your contribution is licensed under the same MIT licence. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Ways to help

- Report a bug or request a feature by writing to dev.a4i@iiitb.ac.in, with steps to reproduce and the OmniIngest version.
- Fix a bug, add a test, improve the docs. Small pull requests are welcome without prior discussion.
- For a new pipeline step, adapter (OCR, speech, translation, vector store) or an API change, get in touch first so we can agree on the design.
- Report a security problem privately. See [SECURITY.md](SECURITY.md).

## Development setup

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/A4i-tech/OmniIngest.git
cd OmniIngest
uv sync --dev --extra storage
```

`python-magic` needs `libmagic` (`brew install libmagic` on macOS, `apt install libmagic1` on Debian or Ubuntu). Audio tests need `ffmpeg`.

BRF (braille) support is optional and needs liblouis; see the Braille section in `docs/advanced.md`. Those tests skip when it is missing.

## Checks before you open a pull request

```bash
uv run ruff check src      # lint
uv run mypy src            # types
uv run pytest tests/unit   # unit tests, no network or keys needed
```

Integration tests under `tests/integration` call real cloud services and need the keys listed in `.env.example`. Run the ones you can; CI runs the rest on trusted branches. Pull requests from forks do not receive secrets, so integration tests that need keys will not run there.

## Pull request guidelines

- Branch from the default branch and keep one change per pull request.
- Add or update tests for behaviour you change. Fixtures in `tests/resources` must be synthetic or freely licensed. Do not add partner, student or other personal data, or copyrighted material.
- Update `docs/` and `CHANGELOG.md` when behaviour or the public API changes. The public API is `create_pipeline_from_config`, `register_step`, `IngestionContext`, `Step`, `StepResult` and `StepStatus`.
- Never commit secrets. A gitleaks scan runs on every pull request.
- Do not add dependencies licensed under AGPL, GPL or SSPL to the core package. Optional extras need a note in `THIRD_PARTY_LICENSES.md`.
- Write commit messages that say what changed and why.

## Documentation

The docs live in `docs/` and are built with MkDocs (`uv run mkdocs serve` if you have the docs tooling installed).

## Questions

Write to dev.a4i@iiitb.ac.in.
