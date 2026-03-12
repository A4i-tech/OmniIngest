"""Call the lesson plan API for each payload JSON and save the results.

Ported from UKI_NEEDS/lp-gen-social.py (process_single_api_call and
call_api_and_save_results functions).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp

from .config import ApiConfig


async def _process_single(
    session: aiohttp.ClientSession,
    payload_path: Path,
    payloads_dir: Path,
    results_dir: Path,
    cfg: ApiConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file_path": str(payload_path),
        "status": "failed",
        "message": "",
        "output_path": "",
    }

    try:
        payload: dict[str, Any] = json.loads(payload_path.read_text(encoding="utf-8"))
        original_id: str = payload.get("_id", "")

        if "workflow" not in payload or "_id" not in payload.get("workflow", {}):
            result["message"] = "Invalid payload schema: workflow._id missing"
            return result

        print(f"  [Sending] {payload_path.name}...")

        async with session.post(
            cfg.url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=cfg.post_timeout_secs),
        ) as resp:
            if resp.status == 400:
                error_msg = await resp.text()
                result["message"] = f"Bad Request: {error_msg}"
                return result
            resp.raise_for_status()
            api_response: dict[str, Any] = await resp.json()

        status_uri: str | None = api_response.get("status_query_get_uri")
        if not status_uri:
            result["message"] = "No status_query_get_uri in response"
            return result

        # Poll for completion
        for _ in range(cfg.max_poll_retries):
            try:
                async with session.get(
                    status_uri,
                    timeout=aiohttp.ClientTimeout(total=cfg.poll_timeout_secs),
                ) as status_resp:
                    status_data: dict[str, Any] = await status_resp.json()
            except Exception:
                await asyncio.sleep(cfg.poll_interval_secs)
                continue

            runtime_status = status_data.get("runtimeStatus")

            if runtime_status == "Completed":
                final_output: dict[str, Any] = status_data.get("output", status_data)

                # ID replacement logic (preserve original behavior)
                if not final_output.get("_id"):
                    final_output["_id"] = original_id
                for field in ("_id", "chapter_id"):
                    if field in final_output and isinstance(final_output[field], str):
                        final_output[field] = final_output[field].replace("Board=SCERT", "Board=BSE-TG")
                if "index_path" in final_output and isinstance(final_output["index_path"], str):
                    final_output["index_path"] = final_output["index_path"].replace("SCERT", "BSE-TG")

                rel = payload_path.relative_to(payloads_dir)
                out_path = results_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(final_output, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                result["status"] = "completed"
                result["output_path"] = str(out_path)
                return result

            elif runtime_status in ("Failed", "Terminated"):
                error_detail = status_data.get("output", "No details provided")
                result["message"] = f"Status: {runtime_status} | Details: {error_detail}"
                return result

            await asyncio.sleep(cfg.poll_interval_secs)

        result["message"] = "Timeout: max poll retries exceeded"

    except Exception as exc:
        result["message"] = str(exc)

    return result


async def run_async(
    payloads_dir: Path,
    results_dir: Path,
    cfg: ApiConfig,
) -> list[dict[str, Any]]:
    all_files = sorted(payloads_dir.rglob("*.json"))
    if not all_files:
        print("No payload files found.")
        return []

    to_process: list[Path] = []
    skipped = 0
    for p in all_files:
        rel = p.relative_to(payloads_dir)
        if (results_dir / rel).exists():
            skipped += 1
        else:
            to_process.append(p)

    print(
        f"Found {len(all_files)} payloads; "
        f"skipping {skipped} (already processed); "
        f"queued {len(to_process)}."
    )
    if not to_process:
        return []

    api_results: list[dict[str, Any]] = []
    connector = aiohttp.TCPConnector(limit=cfg.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(cfg.concurrency)

        async def bounded(p: Path) -> dict[str, Any]:
            async with sem:
                return await _process_single(session, p, payloads_dir, results_dir, cfg)

        tasks = [asyncio.create_task(bounded(p)) for p in to_process]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            api_results.append(res)
            name = Path(res["file_path"]).name
            if res["status"] == "completed":
                print(f"  [OK]   {name}")
            else:
                print(f"  [FAIL] {name}: {res['message']}")

    return api_results


def run(payloads_dir: Path, results_dir: Path, cfg: ApiConfig) -> list[dict[str, Any]]:
    """Synchronous entry point. Runs the async batch processor."""
    return asyncio.run(run_async(payloads_dir, results_dir, cfg))
