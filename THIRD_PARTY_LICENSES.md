# Third-party licences

OmniIngest is MIT licensed (see `LICENSE`). This file lists components that are distributed with, or required by, this repository under a different licence.

## liblouis (LGPL-2.1-or-later)

- Used for: reading and writing BRF (braille) files, in `omni_ingest/parser.py` (`parse_brf`) and `omni_ingest/core/output.py` (`BrfFileOutput`). Imported lazily; nothing else depends on it.
- Upstream: https://github.com/liblouis/liblouis (source, tables and Python bindings)
- Distributed here as: prebuilt wheels `dependencies/liblouis-3.38.0-*.whl` (Linux x86_64 and Windows x86_64), built from liblouis 3.38.0 according to the wheel metadata ("batteries included"), containing the Python `louis` bindings, the shared library (`liblouis.so` / `liblouis.dll`) and the translation tables. The upstream project does not publish these wheels on PyPI. The wheels contain the upstream `COPYING` and `COPYING.LESSER` files.
- The library is loaded dynamically as a separate shared library through the `louis` module and is not statically linked into OmniIngest. You may replace it with any compatible liblouis build, for example your distribution's `liblouis` package and its `python3-louis` bindings.
- Corresponding source for the shipped binaries: https://github.com/liblouis/liblouis/releases/tag/v3.38.0

## Runtime dependencies with notable licences

Found by scanning the installed core dependency set (`pip-licenses`). All other packages are permissive (MIT, BSD, Apache-2.0, ISC, PSF).

- **azure-cognitiveservices-speech** (Microsoft Speech SDK): proprietary Microsoft licence, installed from PyPI and not redistributed here. Use is governed by Microsoft's licence terms.
- **mistralai**: no licence in package metadata. Its upstream repository (`mistralai/client-python`) is Apache-2.0.
- **certifi**, **pathspec**, **tqdm** (MPL-2.0 AND MIT): MPL-2.0 components, used unmodified.
