"""Auditable streaming JSON transport without retries, repair or response caching."""

import json
import time

import httpx

from .metrics import strict_json


def sse_events(lines):
    parts = []
    for line in lines:
        if line == "":
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].removeprefix(" "))
    if parts:
        yield "\n".join(parts)


def stream_json(client, request, *, clock=time.perf_counter):
    started = clock()
    row = {
        "request": request,
        "usage": {},
        "first_content_ms": None,
        "last_content_ms": None,
        "finish_reason": None,
        "error": None,
        "stream_complete": False,
        "chunks": [],
    }
    content, done = [], False
    try:
        with client.stream("POST", "/chat/completions", json=request) as response:
            row["http_status"] = response.status_code
            if response.status_code == 429:
                row["retry_after"] = response.headers.get("retry-after")
            response.raise_for_status()
            for event in sse_events(response.iter_lines()):
                elapsed = (clock() - started) * 1000
                if event == "[DONE]":
                    done = True
                    break
                chunk = json.loads(event)
                for key in ("id", "model", "provider", "usage"):
                    if chunk.get(key):
                        row[key] = chunk[key]
                if chunk.get("error"):
                    row["error"] = "stream_error"
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
                    text = choice.get("delta", {}).get("content")
                    if text:
                        if row["first_content_ms"] is None:
                            row["first_content_ms"] = elapsed
                        row["last_content_ms"] = elapsed
                        row["chunks"].append({"ms": elapsed, "utf8_bytes": len(text.encode())})
                        content.append(text)
        row["stream_complete"] = done and row["finish_reason"] == "stop"
        if not row["stream_complete"]:
            row["error"] = row["error"] or "incomplete_stream"
        row["value"] = strict_json("".join(content))
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        row["error"] = row["error"] or type(exc).__name__
    row["total_ms"] = (clock() - started) * 1000
    row["raw_output"] = "".join(content)
    row["raw_output_bytes"] = len(row["raw_output"].encode())
    return row
