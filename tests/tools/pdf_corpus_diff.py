"""Compare two snapshots made by pdf_corpus_snapshot.py (old backend vs new backend).

Usage:  python tests/tools/pdf_corpus_diff.py old.json new.json [--tol 2.0]

Boxes match when every coordinate is within --tol points. Prints per file and per extraction mode: counts, matched,
only-in-old, only-in-new, plus changes in text loss, marker survival and rasterize output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXTRACTIONS = ("embedded_images", "vector_default", "vector_with_tables")


def close(a: list[float], b: list[float], tol: float) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def match(old: list[dict], new: list[dict], tol: float) -> tuple[int, int, int]:
    remaining = list(new)
    matched = 0
    for item in old:
        for candidate in remaining:
            if candidate["page"] == item["page"] and close(candidate["bbox"], item["bbox"], tol):
                remaining.remove(candidate)
                matched += 1
                break
    return matched, len(old) - matched, len(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--tol", type=float, default=2.0)
    args = parser.parse_args()
    old, new = json.loads(args.old.read_text()), json.loads(args.new.read_text())

    for name in sorted(old.keys() | new.keys()):
        a, b = old.get(name), new.get(name)
        print(f"\n== {name}")
        if a is None or b is None or "error" in a or "error" in b:
            print("  old:", (a or {}).get("error", "ok" if a else "missing"), "| new:", (b or {}).get("error", "ok" if b else "missing"))
            continue
        for key in EXTRACTIONS:
            if isinstance(a[key], dict) or isinstance(b[key], dict):
                print(f"  {key:20s} old={a[key] if isinstance(a[key], dict) else len(a[key])} new={b[key] if isinstance(b[key], dict) else len(b[key])}")
                continue
            matched, only_old, only_new = match(a[key], b[key], args.tol)
            flag = "" if only_old == only_new == 0 else "  <-- DIFFERS"
            print(f"  {key:20s} old={len(a[key]):5d} new={len(b[key]):5d} matched={matched:5d} only_old={only_old:4d} only_new={only_new:4d}{flag}")
        for mode in ("redact", "caption"):
            if mode in a and mode in b and "error" not in a[mode] and "error" not in b[mode]:
                print(f"  {mode:8s} text_chars {a[mode]['text_chars']} -> {b[mode]['text_chars']}  markers {a[mode]['markers_found']}/{a[mode]['items']} -> {b[mode]['markers_found']}/{b[mode]['items']}  size_ratio {a[mode]['bytes_ratio']} -> {b[mode]['bytes_ratio']}")
        if "rasterize" in a and "rasterize" in b and "error" not in a["rasterize"] and "error" not in b["rasterize"]:
            ra, rb = a["rasterize"], b["rasterize"]
            print(f"  rasterize pages {ra['pages']} -> {rb['pages']}  first sizes {ra['sizes_first_5'][:2]} -> {rb['sizes_first_5'][:2]}  seconds {ra['seconds']} -> {rb['seconds']}")
        print(f"  seconds old={a['seconds']} new={b['seconds']}")


if __name__ == "__main__":
    main()
