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
    # unique preserve order
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
    # Prefer concrete model class over ModelPatcher wrapper
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


def _vram_stats_mb() -> Optional[Dict[str, float]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda:0")
        free_b, total_b = torch.cuda.mem_get_info(device)
        return {
            "free": free_b / (1024.0 * 1024.0),
            "total": total_b / (1024.0 * 1024.0),
            "used": (total_b - free_b) / (1024.0 * 1024.0),
            "alloc": float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0),
            "reserved": float(torch.cuda.memory_reserved(device)) / (1024.0 * 1024.0),
            "free_bytes": float(free_b),
            "used_bytes": float(total_b - free_b),
        }
    except Exception:
        return None


def _fmt_vram(stats: Optional[Dict[str, float]]) -> str:
    if not stats:
        return "vram:n/a"
    return (
        f"used:{stats['used']:.0f}MB free:{stats['free']:.0f}MB "
        f"alloc:{stats['alloc']:.0f}MB reserved:{stats['reserved']:.0f}MB "
        f"/ {stats['total']:.0f}MB"
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


def unload_comfy_image_models(
    reason: str = "prepare_for_llm",
    unload_clip_with_image: bool = False,
    llm_model: Optional[str] = None,
    service_id: Optional[str] = None,
) -> bool:
    """
    NORMAL-friendly unload via Comfy free_memory / unload_all_models only.

    free_memory(need) means: I want at least `need` bytes free.
    `need` comes from LLM size cache (measured) or fixed estimate by model name.
    """
    from .common import PROCESS_PREFIX, WARN_PREFIX
    from .llm_vram_cache import begin_llm_vram_session, get_llm_need_bytes

    try:
        import comfy.model_management as mm
    except Exception as exc:  # pragma: no cover
        print(f"{WARN_PREFIX} Comfy model unload unavailable | {str(exc)[:120]}", flush=True)
        return False

    try:
        loaded_list = getattr(mm, "current_loaded_models", None)
        if loaded_list is None:
            return False

        before = len(loaded_list)
        vram_before = _vram_stats_mb()
        total_vram_b = None
        if vram_before and vram_before.get("total"):
            total_vram_b = float(vram_before["total"]) * 1024.0 * 1024.0
        llm_need, llm_note = get_llm_need_bytes(service_id, llm_model, total_vram_bytes=total_vram_b)
        llm_need_mb = llm_need / (1024.0 * 1024.0)

        if before == 0:
            print(
                f"{PROCESS_PREFIX} Comfy 显存卸载 | 无需卸载 | 当前无已加载模型 | "
                f"LLM申报:{llm_need_mb:.0f}MB({llm_note}) | {_fmt_vram(vram_before)} | {reason}",
                flush=True,
            )
            begin_llm_vram_session(
                service_id,
                llm_model,
                (vram_before or {}).get("free_bytes"),
                (vram_before or {}).get("used_bytes"),
            )
            return True

        snapshot = []
        keep_loaded = []
        for loaded in list(loaded_list):
            kind = classify_comfy_model(loaded)
            name = _model_display_name(loaded)
            size_mb = _model_size_mb(loaded)
            snapshot.append((kind, name, size_mb, loaded))
            if (not unload_clip_with_image) and kind in ("clip", "vae"):
                keep_loaded.append(loaded)

        before_desc = ", ".join(_fmt_item(k, n, s) for k, n, s, _ in snapshot) or "-"
        scope = "all" if unload_clip_with_image else "main_only"
        print(
            f"{PROCESS_PREFIX} Comfy 显存卸载准备 | scope:{scope} | 已加载[{before}]: {before_desc} "
            f"| LLM申报:{llm_need_mb:.0f}MB({llm_note}) | {_fmt_vram(vram_before)} | {reason}",
            flush=True,
        )

        device = _primary_torch_device(mm)
        freed_models = []
        need = float(llm_need)

        if unload_clip_with_image:
            if hasattr(mm, "unload_all_models"):
                mm.unload_all_models()
            elif device is not None:
                freed_models = list(mm.free_memory(1e30, device) or [])
            else:
                print(f"{WARN_PREFIX} Comfy unload skipped | no torch device", flush=True)
                return False
            unloaded_items = [_fmt_item(k, n, s) for k, n, s, _ in snapshot]
            kept_items = []
        else:
            if device is None:
                print(f"{WARN_PREFIX} Comfy unload skipped | no torch device", flush=True)
                return False
            try:
                freed_models = list(mm.free_memory(need, device, keep_loaded=keep_loaded) or [])
            except TypeError:
                freed_models = list(mm.free_memory(need, device) or [])

            freed_ids = set(id(x) for x in freed_models)
            unloaded_items = []
            kept_items = []
            for kind, name, size_mb, loaded in snapshot:
                label = _fmt_item(kind, name, size_mb)
                if id(loaded) in freed_ids:
                    unloaded_items.append(label)
                else:
                    still = any(loaded is x for x in list(loaded_list))
                    if still:
                        kept_items.append(label)
                    else:
                        unloaded_items.append(label)

        _soft_empty_cache(mm)
        gc.collect()
        _soft_empty_cache(mm)

        after = len(loaded_list)
        vram_after = _vram_stats_mb()
        freed_used = None
        if vram_before and vram_after:
            freed_used = vram_before["used"] - vram_after["used"]

        begin_llm_vram_session(
            service_id,
            llm_model,
            (vram_after or vram_before or {}).get("free_bytes"),
            (vram_after or vram_before or {}).get("used_bytes"),
        )

        unloaded_text = ", ".join(unloaded_items) if unloaded_items else "无"
        kept_text = ", ".join(kept_items) if kept_items else "无"
        freed_txt = f"释放约:{freed_used:.0f}MB" if freed_used is not None else "释放约:n/a"
        if unload_clip_with_image:
            api_txt = "api:unload_all_models"
        else:
            api_txt = f"free_memory返回:{len(freed_models)} 申报需空闲:{llm_need_mb:.0f}MB({llm_note})"

        print(
            f"{PROCESS_PREFIX} Comfy 显存已卸载 | scope:{scope} | {api_txt} "
            f"| 卸下[{len(unloaded_items)}]: {unloaded_text} "
            f"| 保留[{len(kept_items)}]: {kept_text} | before:{before} after:{after} "
            f"| {freed_txt} | 前[{_fmt_vram(vram_before)}] -> 后[{_fmt_vram(vram_after)}] | {reason}",
            flush=True,
        )
        if freed_used is not None and freed_used < 256 and before == after and not unload_clip_with_image:
            print(
                f"{WARN_PREFIX} free_memory 未卸下模型（{freed_used:.0f}MB）。"
                f"可能当前空闲已满足 LLM 申报，或请开启「CLIP随主模型卸载」。",
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


def note_llm_loaded_vram(
    model: Optional[str] = None,
    service_id: Optional[str] = None,
) -> None:
    """Sample VRAM while local LLM is loaded; persist measured size."""
    from .llm_vram_cache import record_llm_vram_after_load

    stats = _vram_stats_mb() or {}
    record_llm_vram_after_load(
        service_id=service_id,
        model=model,
        free_bytes_now=stats.get("free_bytes"),
        used_bytes_now=stats.get("used_bytes"),
    )