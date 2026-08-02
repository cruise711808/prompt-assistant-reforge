"""Persist measured local-LLM VRAM needs and provide first-run estimates.

Rules:
1) Prefer cache (survives restart) once a run has been measured.
2) First run with size in name (e.g. 27b): fixed table estimate;
   if estimate > 80% of total VRAM, declare 80% of total.
3) Name has no size field: declare 80% of total VRAM.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

# First-run fixed estimates (weights + modest runtime headroom), bytes
_FIXED_BY_SIZE_B = {
    "120b": int(65 * 1024**3),
    "72b": int(42 * 1024**3),
    "70b": int(40 * 1024**3),
    "34b": int(26 * 1024**3),
    "32b": int(24 * 1024**3),
    "31b": int(23 * 1024**3),
    "27b": int(20 * 1024**3),  # Q5-class 27B default
    "22b": int(16 * 1024**3),
    "14b": int(12 * 1024**3),
    "13b": int(11 * 1024**3),
    "12b": int(10 * 1024**3),
    "9b": int(8 * 1024**3),
    "8b": int(7 * 1024**3),
    "7b": int(7 * 1024**3),
    "4b": int(4 * 1024**3),
    "3b": int(3 * 1024**3),
    "1b": int(2 * 1024**3),
}

_QUANT_FACTOR = {
    "q2": 0.70,
    "q3": 0.80,
    "q4": 0.90,
    "q5": 1.00,
    "q6": 1.15,
    "q8": 1.35,
    "f16": 1.70,
    "bf16": 1.70,
    "fp16": 1.70,
}

_TOTAL_RATIO_CAP = 0.80
_MARGIN_BYTES = int(1.0 * 1024**3)  # +1GB when declaring from estimate/cache
_MIN_MEASURE_BYTES = int(1.5 * 1024**3)
_MAX_REASONABLE_BYTES = int(90 * 1024**3)

# session: free VRAM right after Comfy prepare (before LLM load)
_session: Dict[str, Any] = {}


def cache_path() -> str:
    """Plugin-local measured VRAM cache (not under user/default/prompt-assistant).

    Path: <plugin>/cache/llm_vram_cache.json
    Survives Comfy restart; kept out of user config so API/rules stay clean.
    """
    plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(plugin_root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "llm_vram_cache.json")

    # one-time migrate if an older build wrote into user config dir
    if not os.path.isfile(path):
        try:
            from ..config_manager import config_manager
            old = os.path.join(config_manager.config_dir, "llm_vram_cache.json")
            if os.path.isfile(old):
                import shutil
                shutil.copy2(old, path)
                try:
                    os.remove(old)
                except Exception:
                    pass
        except Exception:
            pass
    return path


def make_key(service_id: Optional[str], model: Optional[str]) -> str:
    sid = (service_id or "").strip().lower() or "unknown"
    mid = (model or "").strip().lower() or "unknown"
    return f"{sid}::{mid}"


def _load_cache() -> Dict[str, Any]:
    path = cache_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(data: Dict[str, Any]) -> None:
    path = cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def get_total_vram_bytes() -> Optional[int]:
    """Best-effort total GPU memory in bytes."""
    try:
        import torch
        if torch.cuda.is_available():
            _free_b, total_b = torch.cuda.mem_get_info(0)
            if total_b and total_b > 0:
                return int(total_b)
    except Exception:
        pass
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device() if hasattr(mm, "get_torch_device") else None
        if dev is not None and hasattr(mm, "get_total_memory"):
            total = int(mm.get_total_memory(dev) or 0)
            if total > 0:
                return total
    except Exception:
        pass
    return None


def _cap80(total_bytes: Optional[int]) -> Optional[int]:
    if not total_bytes or total_bytes <= 0:
        return None
    return int(total_bytes * _TOTAL_RATIO_CAP)


def estimate_llm_vram_bytes(model: Optional[str]) -> Tuple[Optional[int], str, bool]:
    """
    Return (bytes_or_None, note, has_size_in_name).
    has_size_in_name=False when name has no 27b/7b/... field.
    """
    name = (model or "").strip().lower().replace("_", "-")
    if not name:
        return None, "no_model_name", False

    matched_tag = None
    base = None
    # longer tags first (120b before 20b-like false friends; we only have listed tags)
    for tag, val in sorted(_FIXED_BY_SIZE_B.items(), key=lambda kv: -len(kv[0])):
        # match as token-ish: start, separator, or end
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(tag)}(?:[^a-z0-9]|$)", name):
            matched_tag = tag
            base = val
            break
        if tag in name:
            matched_tag = tag
            base = val
            break

    if matched_tag is None or base is None:
        return None, "no_size_in_name", False

    factor = 1.0
    qtag = ""
    for q, f in _QUANT_FACTOR.items():
        if q in name:
            factor = f
            qtag = q
            break

    est = int(base * factor)
    note = f"fixed:{matched_tag}" + (f"*{qtag}" if qtag else "")
    return est, note, True


def get_llm_need_bytes(
    service_id: Optional[str],
    model: Optional[str],
    total_vram_bytes: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Bytes of free VRAM we want Comfy to free for this LLM.

    Priority:
      1) measured cache (restart-safe)
      2) fixed from name size field, capped at 80% total
      3) no size in name -> 80% total
    """
    total = total_vram_bytes if total_vram_bytes and total_vram_bytes > 0 else get_total_vram_bytes()
    cap = _cap80(total)

    key = make_key(service_id, model)
    cache = _load_cache()
    entry = cache.get(key)
    if isinstance(entry, dict) and entry.get("bytes"):
        try:
            measured = int(entry["bytes"])
            if _MIN_MEASURE_BYTES <= measured <= _MAX_REASONABLE_BYTES:
                need = measured + _MARGIN_BYTES
                if cap is not None and need > cap:
                    return cap, f"cache:{key}|capped:80%_total"
                return need, f"cache:{key}"
        except Exception:
            pass

    est, note, has_size = estimate_llm_vram_bytes(model)

    # No size field in name -> declare 80% of card
    if not has_size or est is None:
        if cap is not None:
            return cap, "fallback:80%_total(no_size_in_name)"
        # last resort if total VRAM unknown
        return int(16 * 1024**3), "fallback:16GB(no_total_vram)"

    need = int(est) + _MARGIN_BYTES
    if cap is not None and need > cap:
        return cap, f"{note}|capped:80%_total"
    return need, note


def begin_llm_vram_session(
    service_id: Optional[str],
    model: Optional[str],
    free_bytes_after_prepare: Optional[float],
    used_bytes_after_prepare: Optional[float] = None,
) -> None:
    """Call after Comfy unload, before LLM request."""
    global _session
    _session = {
        "key": make_key(service_id, model),
        "service_id": service_id or "",
        "model": model or "",
        "free_after_prepare": float(free_bytes_after_prepare or 0.0),
        "used_after_prepare": float(used_bytes_after_prepare or 0.0),
        "t0": time.time(),
    }


def record_llm_vram_after_load(
    service_id: Optional[str] = None,
    model: Optional[str] = None,
    free_bytes_now: Optional[float] = None,
    used_bytes_now: Optional[float] = None,
) -> Optional[int]:
    """
    Call while LLM is (still) loaded, before unload.
    measured ≈ free_after_prepare - free_now  (or used_now - used_after_prepare)
    Persists to plugin cache/llm_vram_cache.json (restart-safe; not user config).
    """
    from .common import PROCESS_PREFIX, WARN_PREFIX

    global _session
    sess = _session or {}
    key = make_key(service_id or sess.get("service_id"), model or sess.get("model"))

    free_prep = float(sess.get("free_after_prepare") or 0.0)
    used_prep = float(sess.get("used_after_prepare") or 0.0)

    measured = 0.0
    if free_bytes_now is not None and free_prep > 0:
        measured = max(measured, free_prep - float(free_bytes_now))
    if used_bytes_now is not None and used_prep >= 0:
        measured = max(measured, float(used_bytes_now) - used_prep)

    if measured < _MIN_MEASURE_BYTES or measured > _MAX_REASONABLE_BYTES:
        return None

    measured_i = int(measured)
    cache = _load_cache()
    prev = cache.get(key) if isinstance(cache.get(key), dict) else None
    if prev and prev.get("source") == "measured" and prev.get("bytes"):
        try:
            old = int(prev["bytes"])
            measured_i = int(old * 0.4 + measured_i * 0.6)
        except Exception:
            pass

    cache[key] = {
        "bytes": measured_i,
        "mb": round(measured_i / (1024.0 * 1024.0), 1),
        "source": "measured",
        "model": model or sess.get("model") or "",
        "service_id": service_id or sess.get("service_id") or "",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_path": cache_path(),
    }
    try:
        _save_cache(cache)
        print(
            f"{PROCESS_PREFIX} LLM显存已记录 | key:{key} | "
            f"{cache[key]['mb']:.0f}MB | source:measured | file:{os.path.basename(cache_path())}",
            flush=True,
        )
    except Exception as exc:
        print(f"{WARN_PREFIX} LLM显存缓存写入失败 | {str(exc)[:100]}", flush=True)
        return None
    return measured_i


def clear_session() -> None:
    global _session
    _session = {}