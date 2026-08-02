"""Helpers for switching VRAM between local LLMs and ComfyUI image models."""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

from .model_unload import resolve_unload_backend, service_supports_auto_unload

# Combo / setting values
UNLOAD_CLIP_WITH_MAIN = "with_main"
KEEP_CLIP_VAE = "keep_clip"

CLIP_NAME_HINTS = (
    "clip",
    "temodel",
    "textencoder",
    "text_encoder",
    "t5xxl",
    "t5_",
    "umt5",
    "hydit",
    "sdxlclip",
    "sd1clip",
    "sd2clip",
    "llama_tokenizer",
)
VAE_NAME_HINTS = (
    "vae",
    "autoencoder",
    "taesd",
    "wanvae",
)


def coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled", "with_main", "together"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled", "keep_clip", "keep", "main_only"}:
        return False
    return default


def resolve_unload_clip_with_image(value: Any, default: bool = False) -> bool:
    """
    True  = unload CLIP/VAE together with main image model
    False = keep CLIP/VAE, only unload main diffusion model
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {
        UNLOAD_CLIP_WITH_MAIN,
        "together",
        "main_and_clip",
        "with main",
        "yes",
        "true",
        "1",
        "随主模型一起卸载",
        "随主模型卸载",
        "unload with main",
    }:
        return True
    if text in {
        KEEP_CLIP_VAE,
        "keep",
        "keep_clip",
        "main_only",
        "only_main",
        "no",
        "false",
        "0",
        "保留clip/vae",
        "保留 clip/vae",
        "保留clip",
        "keep clip/vae",
    }:
        return False
    return coerce_bool(value, default)


def resolve_node_unload_flags(
    unload_language_model: Any = None,
    unload_image_model: Any = None,
    ollama_auto_unload: Any = None,
    unload_clip_with_image: Any = None,
    default_language: bool = True,
    default_image: bool = True,
    default_clip: bool = False,
) -> Tuple[bool, bool, bool]:
    """Return (unload_language, unload_image, unload_clip_with_image)."""
    language = unload_language_model
    image = unload_image_model

    if language is None and ollama_auto_unload is not None:
        language = ollama_auto_unload
    if image is None and ollama_auto_unload is not None and unload_image_model is None:
        image = default_image

    return (
        coerce_bool(language, default_language),
        coerce_bool(image, default_image),
        resolve_unload_clip_with_image(unload_clip_with_image, default_clip),
    )


def apply_local_unload_config(
    provider_config: Optional[Dict[str, Any]],
    service: Optional[Dict[str, Any]],
    unload_language_model: bool,
) -> Dict[str, Any]:
    """Attach local unload metadata so Ollama / llama-swap honor the node toggle."""
    cfg = dict(provider_config or {})
    service = service or {}

    for key in ("type", "id", "name", "unload_backend", "base_url", "provider"):
        if key not in cfg and service.get(key) is not None:
            cfg[key] = service.get(key)

    if "provider" not in cfg:
        cfg["provider"] = service.get("id") or cfg.get("id")

    backend = resolve_unload_backend(cfg) or resolve_unload_backend(service)
    if backend:
        cfg["unload_backend"] = backend
        cfg["auto_unload"] = bool(unload_language_model)
    elif "auto_unload" in service or service_supports_auto_unload(service):
        cfg["auto_unload"] = bool(unload_language_model)

    return cfg


def _collect_type_names(loaded_model: Any) -> List[str]:
    names: List[str] = []
    try:
        patcher = getattr(loaded_model, "model", None)
        if patcher is not None:
            names.append(type(patcher).__name__)
            inner = getattr(patcher, "model", None)
            if inner is not None:
                names.append(type(inner).__name__)
        real_ref = getattr(loaded_model, "real_model", None)
        if real_ref is not None:
            try:
                real = real_ref() if callable(real_ref) else real_ref
            except Exception:
                real = None
            if real is not None:
                names.append(type(real).__name__)
    except Exception:
        pass
    out = []
    seen = set()
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _model_type_names(loaded_model: Any) -> str:
    return " ".join(_collect_type_names(loaded_model)).lower()


def _model_display_name(loaded_model: Any) -> str:
    names = _collect_type_names(loaded_model)
    if not names:
        return "UnknownModel"
    for n in reversed(names):
        if n not in {"ModelPatcher", "ModelPatcherDynamic", "LoadedModel"}:
            return n
    return names[-1]


def _model_size_mb(loaded_model: Any) -> Optional[float]:
    try:
        if hasattr(loaded_model, "model_loaded_memory"):
            return float(loaded_model.model_loaded_memory()) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        if hasattr(loaded_model, "model_memory"):
            return float(loaded_model.model_memory()) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def _model_loaded_bytes(loaded_model: Any) -> float:
    try:
        if hasattr(loaded_model, "model_loaded_memory"):
            return float(loaded_model.model_loaded_memory() or 0)
    except Exception:
        pass
    try:
        patcher = getattr(loaded_model, "model", None)
        if patcher is not None and hasattr(patcher, "loaded_size"):
            return float(patcher.loaded_size() or 0)
    except Exception:
        pass
    return 0.0


def classify_comfy_model(loaded_model: Any) -> str:
    """
    Classify a Comfy loaded model entry.
    Returns: 'clip' | 'vae' | 'main'
    """
    blob = _model_type_names(loaded_model)
    if any(h in blob for h in CLIP_NAME_HINTS):
        return "clip"
    if any(h in blob for h in VAE_NAME_HINTS) or "autoencoderkl" in blob:
        return "vae"
    return "main"


def _soft_empty_cache(mm) -> None:
    """Ask Comfy to flush CUDA cache; do not invent a second memory manager."""
    if hasattr(mm, "soft_empty_cache"):
        try:
            mm.soft_empty_cache(True)
        except TypeError:
            try:
                mm.soft_empty_cache()
            except Exception:
                pass


def _vram_stats_from_nvsmi() -> Optional[Dict[str, float]]:
    """Board-level VRAM via nvidia-smi (sees llama-server / other processes)."""
    try:
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        line = (out or "").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        free_mb = float(parts[0])
        used_mb = float(parts[1])
        total_mb = float(parts[2])
        if total_mb <= 0:
            return None
        return {
            "free": free_mb,
            "total": total_mb,
            "used": used_mb,
            "alloc": 0.0,
            "reserved": 0.0,
            "free_bytes": free_mb * 1024.0 * 1024.0,
            "used_bytes": used_mb * 1024.0 * 1024.0,
            "source": "nvidia-smi",
        }
    except Exception:
        return None


def _vram_stats_mb() -> Optional[Dict[str, float]]:
    # Prefer nvidia-smi: torch.cuda.mem_get_info on some Windows setups only
    # reflects the current process and misses external llama-server VRAM.
    stats = _vram_stats_from_nvsmi()
    alloc = 0.0
    reserved = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            alloc = float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0)
            reserved = float(torch.cuda.memory_reserved(device)) / (1024.0 * 1024.0)
            if stats is None:
                free_b, total_b = torch.cuda.mem_get_info(device)
                stats = {
                    "free": free_b / (1024.0 * 1024.0),
                    "total": total_b / (1024.0 * 1024.0),
                    "used": (total_b - free_b) / (1024.0 * 1024.0),
                    "free_bytes": float(free_b),
                    "used_bytes": float(total_b - free_b),
                    "source": "torch",
                }
    except Exception:
        pass
    if not stats:
        return None
    stats["alloc"] = alloc
    stats["reserved"] = reserved
    return stats


def _fmt_vram(stats: Optional[Dict[str, float]]) -> str:
    if not stats:
        return "vram:n/a"
    src = stats.get("source") or "?"
    return (
        f"used:{stats['used']:.0f}MB free:{stats['free']:.0f}MB "
        f"alloc:{stats['alloc']:.0f}MB reserved:{stats['reserved']:.0f}MB "
        f"/ {stats['total']:.0f}MB ({src})"
    )


def _fmt_item(kind: str, name: str, size_mb: Optional[float]) -> str:
    kind_label = {"main": "主模型", "clip": "CLIP/TE", "vae": "VAE"}.get(kind, kind)
    if size_mb is None:
        return f"{kind_label}:{name}"
    return f"{kind_label}:{name}({size_mb:.1f}MB)"


def _primary_torch_device(mm):
    if hasattr(mm, "get_torch_device"):
        try:
            dev = mm.get_torch_device()
            if dev is not None:
                return dev
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda:0")
    except Exception:
        pass
    return None


def _resolve_offload_device(patcher: Any, mm):
    """Prefer the patcher's own offload_device (usually CPU)."""
    try:
        off = getattr(patcher, "offload_device", None)
        if off is not None:
            return off
    except Exception:
        pass
    if hasattr(mm, "unet_offload_device"):
        try:
            return mm.unet_offload_device()
        except Exception:
            pass
    try:
        import torch
        return torch.device("cpu")
    except Exception:
        return "cpu"


def _soft_offload_one(loaded: Any, mm, want_free_bytes: float) -> Tuple[float, str]:
    """
    Move weights to CPU/offload_device while keeping the LoadedModel entry alive.

    Returns (bytes_reported_freed, detail_tag).
    Does NOT detach and does NOT pop from current_loaded_models.
    """
    patcher = getattr(loaded, "model", None)
    if patcher is None:
        return 0.0, "dead"

    try:
        loaded.currently_used = False
    except Exception:
        pass

    before = _model_loaded_bytes(loaded)
    if before <= 1.0:
        return 0.0, "already_offloaded"

    # Request enough to park nearly all VRAM-resident weights.
    # partially_unload stops once memory_freed meets the request; oversized is OK.
    request = max(float(want_free_bytes), before + 64.0 * 1024.0 * 1024.0, 1.0)
    offload_device = _resolve_offload_device(patcher, mm)

    freed = 0.0
    method = "none"
    try:
        if hasattr(patcher, "partially_unload"):
            raw = patcher.partially_unload(offload_device, request)
            freed = float(raw or 0)
            method = f"partially_unload->{offload_device}"
        elif hasattr(loaded, "model_unload"):
            # Fallback only: model_unload(memory < loaded) may soft-offload without detach.
            soft_need = max(1.0, before - 1.0)
            fully = loaded.model_unload(soft_need)
            if fully:
                method = "model_unload_detach"
            else:
                method = "model_unload_partial"
            after_fb = _model_loaded_bytes(loaded)
            freed = max(0.0, before - after_fb)
        else:
            return 0.0, "no_api"
    except Exception as exc:
        return 0.0, f"error:{type(exc).__name__}"

    after = _model_loaded_bytes(loaded)
    observed = max(0.0, before - after)
    report = max(freed, observed)
    remain_mb = after / (1024.0 * 1024.0)
    return report, f"{method}|remain:{remain_mb:.1f}MB"


def unload_comfy_image_models(
    reason: str = "prepare_for_llm",
    unload_clip_with_image: bool = False,
    llm_model: Optional[str] = None,
    service_id: Optional[str] = None,
) -> bool:
    """
    Soft-offload Comfy image models to CPU/offload_device.

    Keeps entries in current_loaded_models so weights can stay resident in RAM
    and reload without re-reading from disk. Avoids free_memory()/detach() which
    pop models and force a cold restage.
    """
    from .common import PROCESS_PREFIX, WARN_PREFIX
    
    try:
        import comfy.model_management as mm
    except Exception as exc:  # pragma: no cover
        print(f"{WARN_PREFIX} Comfy model unload unavailable | {str(exc)[:120]}", flush=True)
        return False

    try:
        loaded_list = getattr(mm, "current_loaded_models", None)
        if loaded_list is None:
            return False

        before_count = len(loaded_list)
        vram_before = _vram_stats_mb()

        if before_count == 0:
            print(
                f"{PROCESS_PREFIX} Comfy 软卸载 | 无需卸载 | 当前无已加载模型 | "
                f"{_fmt_vram(vram_before)} | {reason}",
                flush=True,
            )
            return True

        snapshot = []
        targets = []
        kept = []
        for loaded in list(loaded_list):
            kind = classify_comfy_model(loaded)
            name = _model_display_name(loaded)
            size_mb = _model_size_mb(loaded)
            snapshot.append((kind, name, size_mb, loaded))
            if (not unload_clip_with_image) and kind in ("clip", "vae"):
                kept.append((kind, name, size_mb, loaded))
            else:
                targets.append((kind, name, size_mb, loaded))

        before_desc = ", ".join(_fmt_item(k, n, s) for k, n, s, _ in snapshot) or "-"
        scope = "all_soft" if unload_clip_with_image else "main_soft"
        print(
            f"{PROCESS_PREFIX} Comfy 软卸载准备 | scope:{scope} | 已加载[{before_count}]: {before_desc} "
            f"| {_fmt_vram(vram_before)} | {reason}",
            flush=True,
        )

        # Soft-offload largest first so VRAM frees early for the LLM.
        targets_sorted = sorted(
            targets,
            key=lambda item: float(item[2] or 0.0),
            reverse=True,
        )

        unloaded_items: List[str] = []
        kept_items = [_fmt_item(k, n, s) for k, n, s, _ in kept]
        method_bits: List[str] = []
        total_api_freed = 0.0
        still_hot = 0

        for kind, name, size_mb, loaded in targets_sorted:
            # Always fully soft-offload selected models (main / optional clip+vae).
            # Using free_memory(need) would detach when need >= model size.
            want = _model_loaded_bytes(loaded) + 64.0 * 1024.0 * 1024.0
            freed_b, tag = _soft_offload_one(loaded, mm, want)
            total_api_freed += freed_b
            after_mb = _model_size_mb(loaded)
            label = _fmt_item(kind, name, size_mb)
            after_label = f"{label}->VRAM余{after_mb:.1f}MB" if after_mb is not None else label
            unloaded_items.append(after_label)
            method_bits.append(f"{name}:{tag}")
            if (after_mb or 0) > 64.0:
                still_hot += 1

        _soft_empty_cache(mm)
        gc.collect()
        _soft_empty_cache(mm)

        after_count = len(loaded_list)
        vram_after = _vram_stats_mb()
        freed_used = None
        if vram_before and vram_after:
            freed_used = vram_before["used"] - vram_after["used"]

        unloaded_text = ", ".join(unloaded_items) if unloaded_items else "无"
        kept_text = ", ".join(kept_items) if kept_items else "无"
        freed_txt = f"释放约:{freed_used:.0f}MB" if freed_used is not None else "释放约:n/a"
        api_txt = (
            f"api:partially_unload_to_cpu keep_loaded=1 "
            f"api_freed:{(total_api_freed / (1024.0 * 1024.0)):.0f}MB "
        )
        methods_txt = "; ".join(method_bits) if method_bits else "-"

        print(
            f"{PROCESS_PREFIX} Comfy 软卸载完成 | scope:{scope} | {api_txt} "
            f"| 卸下[{len(unloaded_items)}]: {unloaded_text} "
            f"| 保留[{len(kept_items)}]: {kept_text} | list:{before_count}->{after_count} "
            f"| {freed_txt} | 前[{_fmt_vram(vram_before)}] -> 后[{_fmt_vram(vram_after)}] "
            f"| methods: {methods_txt} | {reason}",
            flush=True,
        )

        if after_count < before_count:
            print(
                f"{WARN_PREFIX} 软卸载后 loaded 列表变短 ({before_count}->{after_count})，"
                f"可能有路径触发了 detach；下次重载或将重新读盘。",
                flush=True,
            )
        if freed_used is not None and freed_used < 256 and targets_sorted:
            print(
                f"{WARN_PREFIX} 软卸载后显存几乎未下降（{freed_used:.0f}MB）。"
                f"可尝试开启设置「CLIP随主模型卸载」。仍热模型数:{still_hot}",
                flush=True,
            )
        return True
    except Exception as exc:
        print(f"{WARN_PREFIX} Comfy image model unload failed | {str(exc)[:160]}", flush=True)
        return False


def get_global_unload_clip_with_image(default: bool = False) -> bool:
    """Read ComfyUI setting: PromptAssistant.Settings.UnloadClipWithImage (default off)."""
    try:
        from ..config_manager import config_manager
        settings = config_manager.get_settings() or {}
        if not isinstance(settings, dict):
            return default
        if "PromptAssistant.Settings.UnloadClipWithImage" not in settings:
            return default
        return resolve_unload_clip_with_image(
            settings.get("PromptAssistant.Settings.UnloadClipWithImage"),
            default=default,
        )
    except Exception:
        return default


def prepare_for_local_llm(
    unload_image_model: bool = True,
    unload_clip_with_image: Any = None,
    reason: str = "prepare_for_llm",
    llm_model: Optional[str] = None,
    service_id: Optional[str] = None,
) -> bool:
    if not unload_image_model:
        return False
    if unload_clip_with_image is None:
        clip_flag = get_global_unload_clip_with_image(default=False)
    else:
        clip_flag = resolve_unload_clip_with_image(unload_clip_with_image, default=False)
    return unload_comfy_image_models(
        reason=reason,
        unload_clip_with_image=bool(clip_flag),
        llm_model=llm_model,
        service_id=service_id,
    )


