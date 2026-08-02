"""
Local model unload helpers for Ollama and llama-swap.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

UNLOAD_DELAY_SECONDS = 1.0
OLLAMA_SERVICE_IDS = {"ollama"}
LLAMA_SWAP_SERVICE_IDS = {"llama_swap", "service_355"}


def normalize_service_root_url(base_url: str, default: str = "") -> str:
    """Strip trailing slashes and optional /v1 suffix from a service base URL."""
    root = (base_url or "").strip()
    if not root:
        return default
    root = root.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    # Some users point base_url at full chat endpoints.
    for endpoint in ("/chat/completions", "/v1/chat/completions", "/completions"):
        if endpoint in root:
            root = root.split(endpoint, 1)[0].rstrip("/")
            break
    return root or default


def resolve_unload_backend(service_or_config: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Detect unload backend from service/config metadata.

    Returns:
        'ollama' | 'llama_swap' | None
    """
    if not isinstance(service_or_config, dict):
        return None

    explicit = str(service_or_config.get("unload_backend") or "").strip().lower()
    if explicit in {"ollama", "llama_swap"}:
        return explicit

    service_type = str(service_or_config.get("type") or "").strip().lower()
    service_id = str(service_or_config.get("id") or "").strip().lower()
    service_name = str(service_or_config.get("name") or "").strip().lower()
    base_url = str(service_or_config.get("base_url") or "").strip().lower()
    provider = str(service_or_config.get("provider") or "").strip().lower()

    if service_type == "ollama" or service_id in OLLAMA_SERVICE_IDS or provider == "ollama":
        return "ollama"
    if service_type == "llama_swap" or service_id in LLAMA_SWAP_SERVICE_IDS or provider in LLAMA_SWAP_SERVICE_IDS:
        return "llama_swap"

    compact_name = service_name.replace(" ", "").replace("_", "-")
    if "llama-swap" in compact_name or "llamaswap" in compact_name:
        return "llama_swap"
    if "ollama" in service_name or "ollama" in provider:
        return "ollama"

    # Heuristic for custom OpenAI-compatible entries pointed at llama-swap.
    if "llama-swap" in base_url or "llamaswap" in base_url:
        return "llama_swap"
    if ":11434" in base_url:
        return "ollama"

    return None


def service_supports_auto_unload(service_or_config: Optional[Dict[str, Any]]) -> bool:
    return resolve_unload_backend(service_or_config) is not None


async def wait_before_unload(delay_seconds: float = UNLOAD_DELAY_SECONDS) -> None:
    if delay_seconds and delay_seconds > 0:
        await asyncio.sleep(delay_seconds)


def _fmt_payload(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return "-"
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(payload)


async def unload_local_model(model: str, provider_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Unload a local model to free VRAM/memory when supported.

    - Ollama: POST /api/generate {"model": ..., "keep_alive": 0}
    - llama-swap: POST /api/models/unload/{model}
    """
    from .common import PROCESS_PREFIX, WARN_PREFIX

    provider_config = provider_config or {}
    backend = resolve_unload_backend(provider_config)
    if not backend:
        return

    auto_unload = provider_config.get("auto_unload", True)
    label = "Ollama" if backend == "ollama" else "llama-swap"
    if auto_unload is False:
        print(f"{PROCESS_PREFIX} {label} model kept loaded | model:{model}", flush=True)
        return

    await wait_before_unload()

    default_root = "http://localhost:11434" if backend == "ollama" else "http://127.0.0.1:8080"
    root = normalize_service_root_url(provider_config.get("base_url", ""), default=default_root)

    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            if backend == "ollama":
                url = f"{root}/api/generate"
                payload = {"model": model, "keep_alive": 0}
                print(
                    f"{PROCESS_PREFIX} {label} unload signal | POST {url} | body:{_fmt_payload(payload)}",
                    flush=True,
                )
                response = await client.post(url, json=payload)
                ok = response.status_code == 200
                detail = f"status:{response.status_code}"
                try:
                    body_preview = (response.text or "").strip().replace("\n", " ")
                    if body_preview:
                        detail += f" | resp:{body_preview[:120]}"
                except Exception:
                    pass
            else:
                # Prefer model-specific unload; fall back to unload-all.
                encoded_model = quote(str(model or "").strip(), safe="")
                urls = []
                if encoded_model:
                    urls.append(f"{root}/api/models/unload/{encoded_model}")
                urls.append(f"{root}/api/models/unload")
                ok = False
                last_status = None
                used_url = None
                resp_preview = ""
                for url in urls:
                    print(
                        f"{PROCESS_PREFIX} {label} unload signal | POST {url} | body:- | model:{model}",
                        flush=True,
                    )
                    response = await client.post(url)
                    last_status = response.status_code
                    used_url = url
                    try:
                        resp_preview = (response.text or "").strip().replace("\n", " ")[:120]
                    except Exception:
                        resp_preview = ""
                    if 200 <= response.status_code < 300:
                        ok = True
                        break
                    print(
                        f"{WARN_PREFIX} {label} unload attempt failed | POST {url} | status:{response.status_code}"
                        + (f" | resp:{resp_preview}" if resp_preview else ""),
                        flush=True,
                    )
                detail = f"status:{last_status} | url:{used_url}"
                if resp_preview:
                    detail += f" | resp:{resp_preview}"
                if not ok and last_status is not None:
                    raise RuntimeError(f"HTTP {last_status}")

            if ok:
                print(
                    f"{PROCESS_PREFIX} {label} model unloaded | model:{model} | {detail}",
                    flush=True,
                )
            else:
                print(
                    f"{WARN_PREFIX} {label} unload failed (ignored) | model:{model} | {detail}",
                    flush=True,
                )
    except Exception as e:
        print(
            f"{WARN_PREFIX} {label} unload failed (ignored) | model:{model} | error:{str(e)[:120]}",
            flush=True,
        )


# Backward-compatible aliases used by older import sites.
wait_before_ollama_unload = wait_before_unload
