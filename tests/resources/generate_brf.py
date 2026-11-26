"""Regenerates blueberries-are-now-green.brf from the synthetic text fixture.

One-off developer script. Needs liblouis (`louis`). Run from the repo root:
    python tests/resources/generate_brf.py
Grade 2 English, 40-cell lines, en-us-brf.dis display table.
"""

import textwrap
from pathlib import Path

import louis

HERE = Path(__file__).parent
TABLES = ["en-us-brf.dis", "en-us-g2.ctb"]
PARAGRAPHS = 2


def main() -> None:
    text = (HERE / "blueberries-are-now-green.txt").read_text(encoding="utf-8")
    paragraphs = [p.strip().replace("’", "'") for p in text.split("\n\n") if p.strip()][:PARAGRAPHS]
    cells = [textwrap.fill(str(louis.translateString(TABLES, p)), width=40) for p in paragraphs]
    (HERE / "blueberries-are-now-green.brf").write_text("\n\n".join(cells) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
