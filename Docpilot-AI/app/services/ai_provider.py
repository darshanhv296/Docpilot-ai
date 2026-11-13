"""
DocPilot AI / VISION AI — Centralized AI Provider Manager (ai_provider.py)
Single source of truth for AI provider credentials, live model discovery,
credential validation, text/vision generation, streaming, and embeddings.
Used across Study Copilot, Vision Solver, Recall Cards, Interactive Quiz,
Cheat Sheet, Document Analysis, and Summarization.
"""

from __future__ import annotations

import base64
import hashlib
import email.utils
import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from app.database.db import BASE_DIR, SUPPORTED_PROVIDERS, db, now_iso


def _load_dotenv_into_environ() -> None:
    """Load key=value pairs from .env if present without overwriting existing environment vars."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            trimmed = line.strip()
            if not trimmed or trimmed.startswith("#") or "=" not in trimmed:
                continue
            k, v = trimmed.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


_load_dotenv_into_environ()

logger = logging.getLogger(__name__)

GEMINI_CHAT_MODEL_IDS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-4-hydrogen",
)
GEMINI_PAID_MODEL_IDS = ("gemini-4-carbon", "gemini-4-argon")
GEMINI_LIVE_AUDIO_MODEL_IDS = (
    "gemini-3.8-live",
    "gemini-3.8-live-extended-thinking",
    "gemini-3-flash-live",
    "gemini-3.5-live-translate-preview",
    "gemini-2.5-flash-native-audio-latest",
    "gemini-3.8-flash-tts",
    "gemini-3.5-transcribe",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
)
GEMINI_CURATED_MODEL_IDS = frozenset(
    (*GEMINI_CHAT_MODEL_IDS, *GEMINI_PAID_MODEL_IDS, *GEMINI_LIVE_AUDIO_MODEL_IDS)
)
GROQ_TEXT_MODEL_IDS = (
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "minimaxai/minimax-m2.7",
)
GROQ_VISION_MODEL_IDS = ("qwen/qwen3.8-27b", "meta-models/muse-spark-open")
OPENROUTER_VISION_MODEL_IDS = ("nvidia/nemotron-3.5-super-vl:free",)
OPENROUTER_TEXT_MODEL_IDS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "cohere/north-mini-code:free",
    "particle-ai/particle-1:free",
    "recursive/flint-v1.0:free",
    "baidu/ernie-5.5-flash-preview:free",
    "dots-studio/dots-3-note-preview:free",
    "thinkingmachines/inkling-2-small:free",
    "liquid/lfm-3-1.7b:free",
)
OPENROUTER_CURATED_MODEL_IDS = frozenset(
    ("openrouter/free", *OPENROUTER_VISION_MODEL_IDS, *OPENROUTER_TEXT_MODEL_IDS)
)
CURATED_VISION_IDS = {
    *GEMINI_CHAT_MODEL_IDS,
    *GEMINI_PAID_MODEL_IDS,
    *GROQ_VISION_MODEL_IDS,
    *OPENROUTER_VISION_MODEL_IDS,
    "openrouter/free",
}
VISION_FALLBACK_PRIORITY = (
    ("gemini", "gemini-3.8-flash"),
    ("groq", "qwen/qwen3.8-27b"),
    ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
    ("openrouter", "openrouter/free"),
)
UNSAFE_MODEL_MARKERS = ("nano-banana", "-image", "antigravity", "lyria")
GEMINI_MODEL_CURATED_CATALOG: List[Dict[str, Any]] = []


def _canonical_model_id(provider: str, model_id: str) -> str:
    model_id = str(model_id or "").strip()
    if provider == "gemini" and model_id.startswith("models/"):
        return model_id[len("models/") :]
    return model_id


class ProviderRequestError(RuntimeError):
    """Provider response with an HTTP status that can be classified for retries."""

    def __init__(
        self,
        provider: str,
        model: str,
        status_code: int,
        reason: str,
        retry_after: Optional[float] = None,
        response_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.reason = sanitize_log_text(reason)
        lower_reason = self.reason.lower()
        if status_code == 413 or any(
            marker in lower_reason
            for marker in ("context length", "context_length", "request too large", "too many tokens")
        ):
            self.failure_kind = "context_capacity"
        elif status_code == 429 or any(
            marker in lower_reason
            for marker in ("rate limit", "rate_limit", "tokens per minute", "tpm")
        ):
            self.failure_kind = "rate_limit"
        else:
            self.failure_kind = "provider"
        self.retryable = status_code in (0, 408, 429, 500, 502, 503, 504)
        self.retry_after = max(0.0, retry_after) if retry_after is not None else None
        self.response_headers = response_headers or {}
        super().__init__(self.reason)


def model_pricing_metadata(provider: str, raw_model: Dict[str, Any]) -> Dict[str, Any]:
    """Return pricing only when the provider's model response makes it verifiable."""
    prov = (provider or "").strip().lower()
    if prov == "local":
        return {"pricing_status": "local", "pricing_label": "LOCAL", "isFree": False}
    pricing = raw_model.get("pricing")
    if prov == "openrouter" and isinstance(pricing, dict):
        try:
            prompt = float(pricing.get("prompt", ""))
            completion = float(pricing.get("completion", ""))
        except (TypeError, ValueError):
            prompt = completion = None
        if prompt is not None and completion is not None:
            if prompt == 0 and completion == 0:
                return {"pricing_status": "free", "pricing_label": "FREE", "isFree": True}
            return {"pricing_status": "paid", "pricing_label": "PAID", "isFree": False}
    return {"pricing_status": "unknown", "pricing_label": "Pricing unknown", "isFree": False}


def normalize_model_record(provider: str, model: Dict[str, Any]) -> Dict[str, Any]:
    """Attach stable identity, availability, capabilities, and verified pricing fields."""
    model_id = str(model.get("id") or "")
    pricing = model_pricing_metadata(provider, model)
    if model.get("pricing_status") in ("free", "paid", "local", "unknown"):
        pricing_status = model["pricing_status"]
        pricing = {
            "pricing_status": pricing_status,
            "pricing_label": model.get("pricing_label")
            or ("LOCAL" if pricing_status == "local" else pricing_status.upper()),
            "isFree": pricing_status == "free",
        }
    capabilities = {
        key: model.get(key)
        for key in (
            "text", "vision", "document", "image_generation", "audio_input",
            "audio_generation", "reasoning", "local", "context_window",
            "supported_generation_methods", "input_modalities", "output_modalities",
        )
        if key in model
    }
    return {
        **model,
        "provider_id": provider,
        "exact_model_id": model_id,
        "display_name": model.get("name") or model_id,
        "availability_status": model.get("status") or "unknown",
        "verification_status": model.get("verification_status") or "untested",
        "capability_metadata": capabilities,
        **pricing,
    }


def sort_gemini_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort versioned Gemini IDs numerically, then place aliases/unversioned IDs last."""
    version_pattern = re.compile(r"(?i)gemini[-\s]*(\d+(?:\.\d+)*)")
    fallback_pattern = re.compile(r"(?<!\d)(\d+\.\d+(?:\.\d+)*)(?!\d)")

    def sort_key(model: Dict[str, Any]) -> Tuple[Any, ...]:
        model_id = str(model.get("id") or "")
        display_name = str(model.get("name") or model.get("display_name") or "")
        metadata_version = str(model.get("version") or "")
        identity = f"{model_id} {display_name} {metadata_version}"
        match = version_pattern.search(identity) or fallback_pattern.search(identity)
        if match:
            version = tuple(int(part) for part in match.group(1).split("."))
            version = (version + (0,) * 8)[:8]
            # Stable, preview, and experimental builds of the same version
            # remain distinct; aliases retain their original IDs and sort first.
            lower = identity.lower()
            if "latest" in lower:
                maturity = 4
            elif any(tag in lower for tag in ("preview", "pre-release", "prerelease")):
                maturity = 2
            elif any(tag in lower for tag in ("experimental", "exp-")):
                maturity = 1
            else:
                maturity = 3
            return (0, tuple(-part for part in version), -maturity, model_id.casefold())
        alias_rank = 0 if "latest" in identity.lower() else 1
        return (1, alias_rank, model_id.casefold())

    return sorted(models, key=sort_key)


def resolve_real_gemini_model(requested_model: Optional[str]) -> str:
    model = (requested_model or "").strip()
    if not model:
        raise ValueError("A discovered Gemini model ID is required.")
    if model.startswith("models/"):
        model = model[len("models/") :]
    return model


def sanitize_log_text(msg: str) -> str:
    if not msg:
        return "Unknown error"
    cleaned = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED_GEMINI_KEY]", str(msg))
    cleaned = re.sub(r"sk-[0-9A-Za-z\-_]{15,}", "[REDACTED_API_KEY]", cleaned)
    cleaned = re.sub(r"gsk_[0-9A-Za-z\-_]{15,}", "[REDACTED_GROQ_KEY]", cleaned)
    cleaned = re.sub(r"Bearer\s+[0-9A-Za-z\-_.]{12,}", "Bearer [REDACTED]", cleaned, flags=re.I)
    return cleaned


def _header_int(headers: Dict[str, str], *names: str) -> Optional[int]:
    for name in names:
        try:
            return int(headers[name])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _header_float(headers: Dict[str, str], *names: str) -> Optional[float]:
    for name in names:
        try:
            return float(headers[name])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def estimate_request_tokens(messages: List[Dict[str, Any]]) -> int:
    """Conservative lightweight estimate for text plus multimodal image inputs."""
    char_count = 0
    image_count = 0
    for message in messages:
        char_count += len(str(message.get("content") or ""))
        for image in message.get("images") or []:
            if isinstance(image, dict):
                image_count += 1
    return max(1, math.ceil(char_count / 4) + image_count * 1024)


class AIProviderManager:
    """
    Centralized manager for AI provider credentials, live model discovery,
    active provider/model state, and LLM calls across all features.
    """

    def __init__(self) -> None:
        self.active_provider: str = "gemini"
        self.active_model: str = ""
        self.active_model_name: str = ""
        self.local_endpoint: str = "http://localhost:11434"
        self._model_usage_counts: Dict[Tuple[str, str], int] = {}
        self._model_latency_totals: Dict[Tuple[str, str], float] = {}
        self._model_latency_counts: Dict[Tuple[str, str], int] = {}
        self._model_request_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
        self._model_rate_limits: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._blocked_models: set[Tuple[str, str]] = set()
        try:
            configured_margin = float(os.environ.get("DOCPILOT_TPM_SAFETY_MARGIN", "0.1"))
        except ValueError:
            logger.warning("Invalid DOCPILOT_TPM_SAFETY_MARGIN; using the 0.1 default.")
            configured_margin = 0.1
        if not 0.0 <= configured_margin <= 0.5:
            logger.warning("DOCPILOT_TPM_SAFETY_MARGIN must be between 0 and 0.5; clamping it.")
        self._tpm_safety_margin = min(0.5, max(0.0, configured_margin))
        self._usage_lock = None
        self.initialize_from_db()

    def initialize_from_db(self) -> None:
        """Restore saved provider and model configuration from db.py on startup."""
        settings = db.get_user_settings("usr-default")
        self.active_provider = str(settings.get("active_provider") or "gemini").strip().lower()
        self.active_model = settings.get("active_model") or ""
        self.active_model_name = settings.get("active_model_name") or ""
        self.local_endpoint = settings.get("local_endpoint") or "http://localhost:11434"
        canonical_model = _canonical_model_id(self.active_provider, self.active_model)
        if canonical_model != self.active_model:
            self.active_model = canonical_model
            db.update_user_settings({"active_model": self.active_model})
        if self.active_model and not self._is_model_allowed(
            self.active_provider, self.active_model
        ):
            self._reset_active_model_to_safe_fallback()

    @staticmethod
    def _is_model_allowed(provider: str, model_id: str) -> bool:
        prov = (provider or "").strip().lower()
        model = _canonical_model_id(prov, model_id)
        lowered = model.lower()
        if any(marker in lowered for marker in UNSAFE_MODEL_MARKERS):
            return False
        if prov == "gemini":
            return model in GEMINI_CURATED_MODEL_IDS
        if prov == "groq":
            return model in (*GROQ_TEXT_MODEL_IDS, *GROQ_VISION_MODEL_IDS)
        if prov == "openrouter":
            return model in OPENROUTER_CURATED_MODEL_IDS
        if prov == "openai":
            return lowered.startswith(("gpt-4", "gpt-5", "o1", "o3")) and not any(
                marker in lowered
                for marker in ("-audio", "-realtime", "-transcribe", "-tts", "-search")
            )
        return True

    def _safe_fallback_selection(self) -> Dict[str, str]:
        gemini_status = db.get_provider_public_status("gemini")
        if gemini_status.get("verified") and self.resolve_api_key("gemini")[0]:
            gemini_model = next(
                (
                    item
                    for item in gemini_status.get("available_models") or []
                    if isinstance(item, dict)
                    and _canonical_model_id("gemini", item.get("id", ""))
                    == "gemini-3.8-flash"
                    and item.get("text") is True
                ),
                None,
            )
            if gemini_model:
                return {
                    "provider": "gemini",
                    "model": "gemini-3.8-flash",
                    "model_name": gemini_model.get("name") or "gemini-3.8-flash",
                }

        for provider in SUPPORTED_PROVIDERS:
            if provider == "local":
                continue
            status = self.sanitize_provider_status(
                provider, db.get_provider_public_status(provider)
            )
            if not status.get("verified") or not self.resolve_api_key(provider)[0]:
                continue
            capabilities = status.get("model_capabilities") or {}
            for item in status.get("available_models") or []:
                if not isinstance(item, dict):
                    continue
                raw_model_id = str(item.get("id") or "")
                model_id = _canonical_model_id(provider, raw_model_id)
                model_caps = (
                    capabilities.get(raw_model_id)
                    or capabilities.get(model_id)
                    or item
                )
                if (
                    self._is_model_allowed(provider, model_id)
                    and model_caps.get("text") is True
                    and item.get("pricing_status") == "free"
                    and item.get("availability_status")
                    not in ("unavailable", "temporarily_unavailable")
                ):
                    return {
                        "provider": provider,
                        "model": model_id,
                        "model_name": item.get("name") or model_id,
                    }
        return {
            "provider": "gemini",
            "model": "gemini-3.8-flash",
            "model_name": "gemini-3.8-flash",
        }

    def _reset_active_model_to_safe_fallback(self) -> Dict[str, str]:
        selection = self._safe_fallback_selection()
        self.active_provider = selection["provider"]
        self.active_model = selection["model"]
        self.active_model_name = selection["model_name"]
        db.update_user_settings(
            {
                "active_provider": self.active_provider,
                "active_model": self.active_model,
                "active_model_name": self.active_model_name,
            }
        )
        return selection

    @staticmethod
    def _filter_curated_models(
        provider: str, models: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for item in models:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            model_id = _canonical_model_id(provider, str(item["id"]))
            if not AIProviderManager._is_model_allowed(provider, model_id):
                continue
            if provider == "gemini" and item.get("image_generation") is True:
                continue
            model = {**item, "id": model_id, "exact_model_id": model_id}
            if provider == "gemini":
                model["text"] = (
                    model_id in GEMINI_CHAT_MODEL_IDS
                    or model_id in GEMINI_PAID_MODEL_IDS
                    or model_id in ("gemma-4-31b-it", "gemma-4-26b-a4b-it")
                )
                model["vision"] = model_id in CURATED_VISION_IDS
                model["document"] = model["vision"]
                model["pricing_status"] = (
                    "paid"
                    if model_id in GEMINI_PAID_MODEL_IDS
                    else "free"
                    if model_id in GEMINI_CHAT_MODEL_IDS
                    else "unknown"
                )
            elif provider in ("groq", "openrouter"):
                model["text"] = True
                model["vision"] = model_id in CURATED_VISION_IDS
                model["document"] = model["vision"]
                model["pricing_status"] = "free"
            elif provider == "openai":
                model["text"] = True
                model["pricing_status"] = "paid"
            if model.get("pricing_status") in ("free", "paid"):
                model["pricing_label"] = model["pricing_status"].upper()
                model["isFree"] = model["pricing_status"] == "free"
            model["capability_metadata"] = {
                **(model.get("capability_metadata") or {}),
                **{
                    key: model[key]
                    for key in ("text", "vision", "document", "pricing_status")
                    if key in model
                },
            }
            filtered.append(model)
        return filtered

    def sanitize_provider_status(
        self, provider: str, status: Dict[str, Any]
    ) -> Dict[str, Any]:
        models = self._filter_curated_models(
            provider, status.get("available_models") or []
        )
        if (
            provider == "openrouter"
            and status.get("verified")
            and self.resolve_api_key(provider)[0]
            and not any(item.get("id") == "openrouter/free" for item in models)
        ):
            models.insert(
                0,
                {
                    "id": "openrouter/free",
                    "name": "OpenRouter Free Auto-Router",
                    "text": True,
                    "vision": True,
                    "document": True,
                    "pricing_status": "free",
                    "pricing_label": "FREE",
                    "isFree": True,
                    "availability_status": "discovered",
                    "verification_status": "verified",
                },
            )
        allowed_ids = {item["id"] for item in models}
        capabilities = status.get("model_capabilities") or {}
        filtered_capabilities = {}
        for model_id in allowed_ids:
            model = next(item for item in models if item["id"] == model_id)
            capability_data = {
                key: model[key]
                for key in ("text", "vision", "document")
                if key in model
            }
            capability_data["free_tier"] = (
                True if model.get("pricing_status") == "free"
                else False if model.get("pricing_status") == "paid"
                else None
            )
            filtered_capabilities[model_id] = {
                **(capabilities.get(model_id) or capabilities.get(f"models/{model_id}") or {}),
                **capability_data,
            }
        return {
            **status,
            "available_models": models,
            "model_capabilities": filtered_capabilities,
        }

    def _record_model_state(
        self,
        provider: str,
        model_id: str,
        availability_status: str,
        verification_status: Optional[str] = None,
        vision_capable: Optional[bool] = None,
    ) -> None:
        status = db.get_provider_public_status(provider)
        models = status.get("available_models") or []
        model = next(
            (
                item
                for item in models
                if isinstance(item, dict) and item.get("id") == model_id
            ),
            None,
        )
        if not model:
            return
        model["availability_status"] = availability_status
        if verification_status:
            model["verification_status"] = verification_status
        if vision_capable is not None:
            model["vision"] = vision_capable
            model["document"] = vision_capable
            model["capability_metadata"] = {
                **(model.get("capability_metadata") or {}),
                "vision": vision_capable,
                "document": vision_capable,
            }
        capabilities = status.get("model_capabilities") or {}
        if verification_status or vision_capable is not None:
            model_capabilities = capabilities.get(model_id) or {}
            if verification_status:
                model_capabilities["verification_status"] = verification_status
            if vision_capable is not None:
                model_capabilities["vision"] = vision_capable
            capabilities[model_id] = model_capabilities
        db.save_provider_credentials(
            provider=provider,
            api_key=None,
            base_url=status.get("base_url"),
            verified=bool(status.get("verified")),
            authentication_method=status.get("authentication_method") or "user_stored_key",
            available_models=models,
            model_capabilities=capabilities,
        )

    def get_model_usage_stats(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        lock = getattr(self, "_usage_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._usage_lock = lock
        with lock:
            counts = dict(getattr(self, "_model_usage_counts", {}))
            latency_totals = dict(getattr(self, "_model_latency_totals", {}))
            latency_counts = dict(getattr(self, "_model_latency_counts", {}))
        return {
            key: {
                "request_count": count,
                "success_count": getattr(self, "_model_request_counts", {}).get(key, {}).get("success", 0),
                "failure_count": getattr(self, "_model_request_counts", {}).get(key, {}).get("failure", 0),
                "average_latency_ms": (
                    latency_totals[key] / latency_counts[key]
                    if latency_counts.get(key)
                    else None
                ),
                "rate_limits": getattr(self, "_model_rate_limits", {}).get(key, {}),
            }
            for key, count in counts.items()
        }

    def _record_model_usage(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        success: bool = True,
        response_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        lock = getattr(self, "_usage_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._usage_lock = lock
        key = (provider, model)
        with lock:
            counts = getattr(self, "_model_usage_counts", None)
            if counts is None:
                counts = {}
                self._model_usage_counts = counts
            latency_totals = getattr(self, "_model_latency_totals", None)
            if latency_totals is None:
                latency_totals = {}
                self._model_latency_totals = latency_totals
            latency_counts = getattr(self, "_model_latency_counts", None)
            if latency_counts is None:
                latency_counts = {}
                self._model_latency_counts = latency_counts
            counts[key] = counts.get(key, 0) + 1
            latency_totals[key] = latency_totals.get(key, 0.0) + max(0.0, latency_ms)
            latency_counts[key] = latency_counts.get(key, 0) + 1
            outcomes = getattr(self, "_model_request_counts", None)
            if outcomes is None:
                outcomes = {}
                self._model_request_counts = outcomes
            outcome = outcomes.setdefault(key, {"success": 0, "failure": 0})
            outcome["success" if success else "failure"] += 1
            if response_headers:
                limits = getattr(self, "_model_rate_limits", None)
                if limits is None:
                    limits = {}
                    self._model_rate_limits = limits
                normalized = {str(name).lower(): str(value) for name, value in response_headers.items()}
                rate_info = {
                    "limit_tokens": _header_int(normalized, "x-ratelimit-limit-tokens", "anthropic-ratelimit-tokens-limit"),
                    "remaining_tokens": _header_int(normalized, "x-ratelimit-remaining-tokens", "anthropic-ratelimit-tokens-remaining"),
                    "reset_tokens": _header_float(normalized, "x-ratelimit-reset-tokens", "anthropic-ratelimit-tokens-reset"),
                    "observed_at": time.time(),
                }
                limits[key] = {name: value for name, value in rate_info.items() if value is not None}

    @staticmethod
    def _retry_after_seconds(data: Dict[str, Any]) -> Optional[float]:
        headers = data.get("__response_headers__") if isinstance(data, dict) else None
        if not isinstance(headers, dict):
            return None
        normalized = {str(name).lower(): str(value).strip() for name, value in headers.items()}
        value = normalized.get("retry-after")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
                return max(0.0, parsed.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _choose_chat_model(
        models: List[Dict[str, Any]],
        preferred_model: Optional[str],
        current_model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        chat_models = [item for item in models if item.get("text") is True]
        for model_id in (preferred_model, current_model):
            if model_id:
                match = next(
                    (item for item in chat_models if item.get("id") == model_id),
                    None,
                )
                if match:
                    return match
        return chat_models[0] if chat_models else None

    def _find_vision_fallback(
        self,
        provider: str,
        excluded_model: str,
        base_url: str = "",
        allow_cross_provider: bool = False,
    ) -> Optional[Dict[str, Any]]:
        for candidate_provider, candidate_model in VISION_FALLBACK_PRIORITY:
            if (candidate_provider, candidate_model) == (provider, excluded_model):
                continue
            if not allow_cross_provider and candidate_provider != provider:
                continue
            candidate = self._configured_curated_vision_model(
                candidate_provider, candidate_model
            )
            if candidate:
                logger.info(
                    "Vision fallback candidate selected: previous_provider=%s previous_model=%s provider=%s model=%s",
                    provider,
                    excluded_model,
                    candidate["provider"],
                    candidate["model"],
                )
                return {
                    **candidate,
                    "base_url": candidate.get("base_url") or base_url or self.local_endpoint,
                }
        return None

    def _configured_curated_vision_model(
        self, provider: str, model: str
    ) -> Optional[Dict[str, Any]]:
        if model not in CURATED_VISION_IDS or not self._is_model_allowed(provider, model):
            return None
        if (provider, model) in getattr(self, "_blocked_models", set()):
            return None
        status = db.get_provider_public_status(provider)
        if not status.get("verified") or not self.resolve_api_key(provider)[0]:
            return None
        entry = next(
            (
                item
                for item in status.get("available_models") or []
                if isinstance(item, dict)
                and _canonical_model_id(provider, item.get("id", "")) == model
            ),
            {},
        )
        if entry.get("availability_status") in ("unavailable", "temporarily_unavailable"):
            return None
        return {
            "provider": provider,
            "model": model,
            "model_name": entry.get("name") or model,
            "base_url": status.get("base_url") or "",
        }


    def resolve_api_key(self, provider: str, explicit_key: Optional[str] = None) -> Tuple[str, str]:
        """
        Resolves the API key for `provider` without ever exposing it to the client.
        Priority:
        1. Explicit unmasked key passed during verification
        2. Encrypted user key stored in SQLite `provider_vault` via `db.py`
        3. Server environment variable (unless provider credentials were explicitly deleted)
        """
        prov = (provider or self.active_provider or "gemini").strip().lower()
        clean_explicit = (explicit_key or "").strip()
        if clean_explicit and not clean_explicit.startswith("••"):
            return clean_explicit, "user_stored_key"

        stored = db.get_decrypted_api_key(prov)
        if stored:
            return stored, "user_stored_key"

        if db.is_provider_explicitly_deleted(prov):
            return "", "none"

        placeholders = {
            "your_gemini_api_key_here",
            "todo_keyhere",
            "my_gemini_api_key",
            "your_api_key_here",
            "changeme",
        }
        if prov == "gemini":
            for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY"):
                val = os.environ.get(env_name, "").strip()
                if val and val.lower() not in placeholders:
                    return val, "built_in_server_key"
        elif prov == "groq":
            val = os.environ.get("GROQ_API_KEY", "").strip()
            if val and val.lower() not in placeholders:
                return val, "built_in_server_key"
        elif prov == "openrouter":
            val = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if val and val.lower() not in placeholders:
                return val, "built_in_server_key"
        elif prov == "openai":
            val = os.environ.get("OPENAI_API_KEY", "").strip()
            if val and val.lower() not in placeholders:
                return val, "built_in_server_key"
        elif prov == "anthropic":
            val = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if val and val.lower() not in placeholders:
                return val, "built_in_server_key"

        return "", "none"

    def _http_json(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = 8.0,
    ) -> Tuple[int, Dict[str, Any]]:
        req_headers = {"Accept": "application/json", "User-Agent": "DocPilot-VISION-AI/2.0"}
        if headers:
            req_headers.update(headers)
        data_bytes = None
        if body is not None:
            data_bytes = json.dumps(body).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    result = json.loads(raw)
                except Exception:
                    result = {"raw": raw}
                if isinstance(result, dict):
                    result["__response_headers__"] = dict(resp.headers.items())
                return resp.status, result
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            try:
                result = json.loads(raw)
            except Exception:
                result = {"error": {"message": raw or str(e.reason)}}
            if isinstance(result, dict):
                result["__response_headers__"] = dict(e.headers.items()) if e.headers else {}
            return e.code, result
        except Exception as e:
            return 0, {"error": {"message": sanitize_log_text(str(e))}}

    @staticmethod
    def _capabilities_from_metadata(
        provider: str, item: Dict[str, Any], model_id: str
    ) -> Dict[str, Any]:
        """Normalize capability metadata without guessing from model names."""
        raw_methods = item.get("supportedGenerationMethods") or item.get(
            "supported_generation_methods"
        ) or []
        if isinstance(raw_methods, str):
            raw_methods = [raw_methods]
        methods = {
            str(value).lower()
            for value in raw_methods
        }
        modalities = (
            item.get("modalities")
            or item.get("input_modalities")
            or item.get("inputModalities")
            or []
        )
        output_modalities = item.get("output_modalities") or item.get("outputModalities") or []
        if isinstance(modalities, str):
            modalities = [modalities]
        if isinstance(output_modalities, str):
            output_modalities = [output_modalities]
        all_modalities = {str(value).lower() for value in [*modalities, *output_modalities]}
        architecture = item.get("architecture") or {}
        if isinstance(architecture, dict):
            all_modalities.update(
                str(value).lower()
                for value in (architecture.get("input_modalities") or [])
            )
            all_modalities.update(
                str(value).lower()
                for value in (architecture.get("output_modalities") or [])
            )

        text = bool(
            "generatecontent" in methods
            or "chat.completions" in methods
            or "text" in all_modalities
            or (
                provider in ("openai", "groq", "openrouter", "anthropic", "local")
                and not methods
                and not all_modalities
            )
        )
        input_modality_set = {str(value).lower() for value in modalities}
        output_modality_set = {str(value).lower() for value in output_modalities}
        vision = True if (
            "image" in input_modality_set
            or "image_url" in input_modality_set
            or "vision" in input_modality_set
            or item.get("vision") is True
        ) else (
            False
            if item.get("vision") is False
            or bool(modalities)
            or any(key in item for key in ("input_modalities", "inputModalities"))
            else None
        )
        image_generation = bool(
            "image_generation" in output_modality_set
            or "image" in output_modality_set
            or "image_generation" in methods
            or item.get("image_generation") is True
        )
        audio_input = bool("audio" in input_modality_set or "audio_input" in input_modality_set)
        audio_output = bool(
            "audio" in output_modality_set
            or "audio_output" in output_modality_set
            or "tts" in methods
        )
        return {
            "text": text,
            "vision": vision,
            "document": vision or "document" in all_modalities,
            "image_generation": image_generation,
            "audio_input": audio_input,
            "audio_generation": audio_output,
            "reasoning": bool(item.get("reasoning") or item.get("thinking")),
            "local": provider == "local",
            "supported_generation_methods": list(raw_methods),
            "input_modalities": list(modalities),
            "output_modalities": list(output_modalities),
            "context_window": int(
                item.get("inputTokenLimit")
                or item.get("context_length")
                or item.get("context_window")
                or 0
            ),
        }

    @staticmethod
    def _cached_models_are_fresh(status: Dict[str, Any], max_age_seconds: int = 900) -> bool:
        timestamp = status.get("verification_timestamp")
        if not timestamp or not status.get("available_models"):
            return False
        try:
            from datetime import datetime, timezone

            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - parsed).total_seconds() < max_age_seconds
        except (TypeError, ValueError):
            return False

    def verify_and_discover_provider(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        activate: bool = True,
        selected_model: Optional[str] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Validates credentials for `provider`, discovers available models,
        saves the encrypted key in `db.py` ONLY if valid (preserving previous
        working configuration if the new key fails validation!), and updates
        the active provider immediately without restarting the server.
        """
        prov = (provider or "gemini").strip().lower()
        if prov not in SUPPORTED_PROVIDERS:
            return {
                "verified": False,
                "provider": prov,
                "available_models": [],
                "safe_error_code": "PROVIDER_UNAVAILABLE",
                "user_friendly_message": f"Unsupported provider '{prov}'.",
            }

        prev_status = db.get_provider_public_status(prov)
        if (
            not refresh
            and not api_key
            and not base_url
            and prev_status.get("verified")
            and self._cached_models_are_fresh(prev_status)
        ):
            cached_status = self.sanitize_provider_status(prov, prev_status)
            return {
                "verified": True,
                "provider": prov,
                "authentication_method": prev_status.get("authentication_method"),
                "verification_timestamp": prev_status.get("verification_timestamp"),
                "available_models": cached_status["available_models"],
                "model_capabilities": cached_status["model_capabilities"],
                "user_friendly_message": "Using recently discovered provider models.",
                "cached": True,
            }
        resolved_key, auth_method = self.resolve_api_key(prov, api_key)
        resolved_base = (
            (base_url or "").strip()
            or prev_status.get("base_url")
            or ("http://localhost:11434" if prov == "local" else "")
        )

        models: List[Dict[str, Any]] = []
        caps: Dict[str, Any] = {}

        if prov == "local":
            clean_base = resolved_base.rstrip("/")
            status_code, data = self._http_json(f"{clean_base}/api/tags", timeout=4.0)
            if status_code == 200 and isinstance(data.get("models"), list):
                for m in data["models"]:
                    mid = m.get("name") or m.get("model")
                    if not mid:
                        continue
                    show_code, show_data = self._http_json(
                        f"{clean_base}/api/show",
                        method="POST",
                        body={"name": mid},
                        timeout=4.0,
                    )
                    raw_capabilities = (
                        show_data.get("capabilities")
                        if show_code == 200 and isinstance(show_data.get("capabilities"), list)
                        else None
                    )
                    vision_capable = (
                        "vision" in raw_capabilities if raw_capabilities is not None else None
                    )
                    models.append(
                        {
                            "id": mid,
                            "name": f"{mid} (Ollama Local)",
                            **model_pricing_metadata("local", m),
                            "vision": vision_capable,
                            "document": vision_capable,
                            "image_generation": False,
                            "audio_input": False,
                            "audio_generation": False,
                            "reasoning": None,
                            "status": "discovered",
                            "verification_status": "untested" if raw_capabilities is None else "verified",
                            "desc": f"Local Ollama model • {m.get('details', {}).get('parameter_size', 'Local')}",
                            "context_length": None,
                        }
                    )
            else:
                status_v1, data_v1 = self._http_json(f"{clean_base}/v1/models", timeout=4.0)
                if status_v1 == 200 and isinstance(data_v1.get("data"), list):
                    for m in data_v1["data"]:
                        mid = m.get("id")
                        if not mid:
                            continue
                        models.append(
                            {
                                "id": mid,
                                "name": f"{mid} (Local Server)",
                                **model_pricing_metadata("local", m),
                                "vision": None,
                                "document": None,
                                "image_generation": False,
                                "audio_input": False,
                                "audio_generation": False,
                                "reasoning": False,
                                "status": "discovered",
                                "verification_status": "untested",
                                "desc": "Local OpenAI-compatible model",
                                "context_length": None,
                            }
                        )
                else:
                    return {
                        "verified": False,
                        "provider": "local",
                        "authentication_method": "local_endpoint",
                        "verification_timestamp": now_iso(),
                        "available_models": [],
                        "model_capabilities": {},
                        "safe_error_code": "LOCAL_OFFLINE",
                        "user_friendly_message": f"Local AI server at {clean_base} is offline or unreachable.",
                    }

            models = [normalize_model_record("local", model) for model in models]
            for m in models:
                caps[m["id"]] = {
                    **self._capabilities_from_metadata("local", m, m["id"]),
                    "free_tier": None,
                }
            db.save_provider_credentials(
                provider="local",
                api_key="",
                base_url=clean_base,
                verified=True,
                authentication_method="local_endpoint",
                available_models=models,
                model_capabilities=caps,
            )
            if activate:
                self.active_provider = "local"
                self.local_endpoint = clean_base
                chosen = self._choose_chat_model(models, selected_model, self.active_model)
                self.active_model = chosen["id"] if chosen else ""
                self.active_model_name = chosen["name"] if chosen else ""
                db.update_user_settings(
                    {
                        "active_provider": self.active_provider,
                        "active_model": self.active_model,
                        "active_model_name": self.active_model_name,
                        "active_model_is_free": bool(
                            chosen and chosen.get("pricing_status") == "free"
                        ),
                        "local_endpoint": self.local_endpoint,
                    }
                )
            return {
                "verified": True,
                "provider": "local",
                "authentication_method": "local_endpoint",
                "verification_timestamp": now_iso(),
                "available_models": models,
                "model_capabilities": caps,
                "user_friendly_message": (
                    f"Connected to Local AI Server at {clean_base} ({len(models)} models)."
                    if any(item.get("text") is True for item in models)
                    else f"Connected to Local AI Server at {clean_base}, but no discovered text chat model is available."
                ),
            }

        # Non-local providers require a key
        if not resolved_key:
            return {
                "verified": False,
                "provider": prov,
                "authentication_method": "none",
                "verification_timestamp": now_iso(),
                "available_models": self._filter_curated_models(
                    prov, prev_status.get("available_models") or []
                ),
                "model_capabilities": self.sanitize_provider_status(prov, prev_status)[
                    "model_capabilities"
                ],
                "safe_error_code": "MISSING_CREDENTIALS",
                "user_friendly_message": f"No {prov.upper()} API key configured. Enter your key in Settings to connect.",
            }

        # Validate key against provider API
        if prov == "gemini":
            models = []
            seen_model_ids = set()
            page_token = ""
            seen_page_tokens = set()
            while True:
                query = {"key": resolved_key, "pageSize": 100}
                if page_token:
                    query["pageToken"] = page_token
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models?"
                    + urllib.parse.urlencode(query)
                )
                code, data = self._http_json(url, timeout=8.0)
                if code != 200 or not isinstance(data.get("models"), list):
                    err_msg = (
                        (data.get("error") or {}).get("message")
                        if isinstance(data.get("error"), dict)
                        else "Invalid Gemini API key or network error."
                    )
                    if not prev_status.get("verified"):
                        db.save_provider_credentials(
                            provider="gemini",
                            api_key=None,
                            verified=False,
                            authentication_method=auth_method,
                            last_error_code="AUTH_FAILED",
                            last_error_message=sanitize_log_text(str(err_msg)),
                        )
                    return {
                        "verified": False,
                        "provider": "gemini",
                        "authentication_method": auth_method,
                        "verification_timestamp": now_iso(),
                        "available_models": self._filter_curated_models(
                            prov, prev_status.get("available_models") or []
                        ),
                        "model_capabilities": self.sanitize_provider_status(prov, prev_status)[
                            "model_capabilities"
                        ],
                        "safe_error_code": "AUTH_FAILED" if code in (400, 401, 403) else "PROVIDER_UNAVAILABLE",
                        "user_friendly_message": sanitize_log_text(str(err_msg)),
                    }

                for item in data["models"]:
                    if not isinstance(item, dict):
                        continue
                    model_id = _canonical_model_id(
                        "gemini", str(item.get("name") or item.get("id") or "")
                    )
                    if not model_id or model_id in seen_model_ids:
                        continue
                    if model_id not in GEMINI_CURATED_MODEL_IDS:
                        continue
                    raw_methods = item.get("supportedGenerationMethods") or []
                    if isinstance(raw_methods, str):
                        raw_methods = [raw_methods]
                    metadata_caps = self._capabilities_from_metadata("gemini", item, model_id)
                    if metadata_caps.get("image_generation"):
                        continue
                    is_text_chat = (
                        model_id in GEMINI_CHAT_MODEL_IDS
                        or model_id in GEMINI_PAID_MODEL_IDS
                        or model_id in ("gemma-4-31b-it", "gemma-4-26b-a4b-it")
                    )
                    pricing_status = (
                        "paid"
                        if model_id in GEMINI_PAID_MODEL_IDS
                        else "free"
                        if model_id in GEMINI_CHAT_MODEL_IDS
                        else "unknown"
                    )
                    model = {
                        **item,
                        "id": model_id,
                        "name": item.get("displayName") or item.get("display_name") or model_id,
                        **metadata_caps,
                        "text": is_text_chat,
                        "vision": model_id in CURATED_VISION_IDS,
                        "document": model_id in CURATED_VISION_IDS,
                        "pricing_status": pricing_status,
                        "pricing_label": pricing_status.upper() if pricing_status != "unknown" else "Pricing unknown",
                        "isFree": pricing_status == "free",
                        "audio_input": model_id in (
                            "gemini-3.8-live",
                            "gemini-3.8-live-extended-thinking",
                            "gemini-3-flash-live",
                            "gemini-3.5-live-translate-preview",
                            "gemini-2.5-flash-native-audio-latest",
                            "gemini-3.5-transcribe",
                        ),
                        "audio_generation": model_id in (
                            "gemini-3.8-live",
                            "gemini-3.8-live-extended-thinking",
                            "gemini-3-flash-live",
                            "gemini-3.5-live-translate-preview",
                            "gemini-2.5-flash-native-audio-latest",
                            "gemini-3.8-flash-tts",
                        ),
                        "status": "discovered",
                        "availability_status": "discovered",
                        "verification_status": "untested",
                        "supported_generation_methods": raw_methods,
                        "desc": item.get("description") or "Listed by Google Gemini API",
                        "context_length": int(item.get("inputTokenLimit") or 0),
                    }
                    models.append(normalize_model_record("gemini", model))
                    seen_model_ids.add(model_id)

                page_token = str(data.get("nextPageToken") or "")
                if not page_token:
                    break
                if page_token in seen_page_tokens:
                    raise RuntimeError("Gemini model listing returned a repeated pagination token.")
                seen_page_tokens.add(page_token)

            gemini_order = (
                *GEMINI_CHAT_MODEL_IDS,
                *GEMINI_PAID_MODEL_IDS,
                *GEMINI_LIVE_AUDIO_MODEL_IDS,
            )
            order = {model_id: index for index, model_id in enumerate(gemini_order)}
            models.sort(key=lambda item: order.get(str(item.get("id") or ""), len(order)))
            caps = {
                model["id"]: {
                    **model.get("capability_metadata", {}),
                    "free_tier": (
                        True if model.get("pricing_status") == "free"
                        else False if model.get("pricing_status") == "paid"
                        else None
                    ),
                    "pricing_status": model.get("pricing_status", "unknown"),
                    "availability_status": "listed",
                }
                for model in models
            }
            if not models:
                return {
                    "verified": False,
                    "provider": "gemini",
                    "authentication_method": auth_method,
                    "verification_timestamp": now_iso(),
                    "available_models": [],
                    "model_capabilities": {},
                    "safe_error_code": "NO_WORKING_MODELS",
                    "user_friendly_message": "The Gemini key was accepted, but the model-listing endpoint returned no models.",
                }

        elif prov in ("groq", "openrouter", "openai"):
            endpoint_map = {
                "groq": "https://api.groq.com/openai/v1/models",
                "openrouter": "https://openrouter.ai/api/v1/models",
                "openai": "https://api.openai.com/v1/models",
            }
            after = ""
            seen_cursors = set()
            while True:
                listing_url = endpoint_map[prov]
                if prov == "openai":
                    query = {"limit": 100}
                    if after:
                        query["after"] = after
                    listing_url = f"{listing_url}?{urllib.parse.urlencode(query)}"
                code, data = self._http_json(
                    listing_url,
                    headers={"Authorization": "Bearer " + resolved_key},
                    timeout=8.0,
                )
                if code != 200 or not isinstance(data.get("data"), list):
                    err_msg = (
                        (data.get("error") or {}).get("message")
                        if isinstance(data.get("error"), dict)
                        else f"{prov.upper()} model discovery failed (HTTP {code})."
                    )
                    return {
                        "verified": False,
                        "provider": prov,
                        "authentication_method": auth_method,
                        "verification_timestamp": now_iso(),
                        "available_models": self._filter_curated_models(
                            prov, prev_status.get("available_models") or []
                        ),
                        "model_capabilities": self.sanitize_provider_status(prov, prev_status)[
                            "model_capabilities"
                        ],
                        "safe_error_code": "AUTH_FAILED" if code in (401, 403) else "PROVIDER_UNAVAILABLE",
                        "user_friendly_message": sanitize_log_text(str(err_msg)),
                    }

                page = data["data"]
                for m in page:
                    mid = str(m.get("id") or "").strip()
                    if not mid:
                        continue
                    if prov == "groq" and mid not in (*GROQ_TEXT_MODEL_IDS, *GROQ_VISION_MODEL_IDS):
                        continue
                    if prov == "openrouter" and mid not in OPENROUTER_CURATED_MODEL_IDS:
                        continue
                    if prov == "openai" and not self._is_model_allowed(prov, mid):
                        continue
                    low = mid.lower()
                    if any(x in low for x in ("whisper", "tts", "dall-e", "embedding", "moderation")):
                        continue
                    metadata_caps = self._capabilities_from_metadata(prov, m, mid)
                    is_curated_vision = mid in CURATED_VISION_IDS
                    curated_free = prov == "groq" or prov == "openrouter"
                    models.append(
                        {
                            "id": mid,
                            "name": m.get("name") or mid,
                            **metadata_caps,
                            "text": True if prov in ("groq", "openrouter", "openai") else metadata_caps["text"],
                            "vision": is_curated_vision,
                            "document": is_curated_vision,
                            "pricing_status": "free" if curated_free else "paid",
                            "pricing_label": "FREE" if curated_free else "PAID",
                            "isFree": curated_free,
                            "status": "discovered",
                            "verification_status": "untested",
                            "desc": f"{prov.upper()} live model",
                            "context_length": int(m.get("context_length") or m.get("context_window") or 0) or None,
                        }
                    )
                cursor = str(data.get("last_id") or (page[-1].get("id") if page else ""))
                if prov != "openai" or not data.get("has_more"):
                    break
                if not cursor or cursor in seen_cursors:
                    raise RuntimeError(f"{prov.upper()} model listing returned a repeated pagination cursor.")
                seen_cursors.add(cursor)
                after = cursor

            if prov == "openrouter" and not any(
                item.get("id") == "openrouter/free" for item in models
            ):
                models.append(
                    {
                        "id": "openrouter/free",
                        "name": "OpenRouter Free Auto-Router",
                        "text": True,
                        "vision": True,
                        "document": True,
                        "pricing_status": "free",
                        "pricing_label": "FREE",
                        "isFree": True,
                        "image_generation": False,
                        "status": "discovered",
                        "availability_status": "discovered",
                        "verification_status": "verified",
                        "desc": "Routes requests to an available free OpenRouter model.",
                    }
                )

        elif prov == "anthropic":
            after = ""
            seen_cursors = set()
            while True:
                query = {"limit": 100}
                if after:
                    query["after_id"] = after
                listing_url = "https://api.anthropic.com/v1/models?" + urllib.parse.urlencode(query)
                code, data = self._http_json(
                    listing_url,
                    headers={
                        "x-api-key": resolved_key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=8.0,
                )
                if code != 200:
                    err_msg = (
                        (data.get("error") or {}).get("message")
                        if isinstance(data.get("error"), dict)
                        else f"Anthropic model discovery failed (HTTP {code})."
                    )
                    return {
                        "verified": False,
                        "provider": "anthropic",
                        "authentication_method": auth_method,
                        "verification_timestamp": now_iso(),
                        "available_models": self._filter_curated_models(
                            prov, prev_status.get("available_models") or []
                        ),
                        "model_capabilities": self.sanitize_provider_status(prov, prev_status)[
                            "model_capabilities"
                        ],
                        "safe_error_code": "AUTH_FAILED" if code in (401, 403) else "PROVIDER_UNAVAILABLE",
                        "user_friendly_message": sanitize_log_text(str(err_msg)),
                    }
                page = data.get("data") if isinstance(data.get("data"), list) else []
                for item in page:
                    mid = item.get("id")
                    if not mid:
                        continue
                    models.append(
                        {
                            "id": mid,
                            "name": item.get("display_name") or mid,
                            **self._capabilities_from_metadata(prov, item, mid),
                            "pricing_status": "paid",
                            "pricing_label": "PAID",
                            "isFree": False,
                            "status": "discovered",
                            "verification_status": "untested",
                            "desc": "Anthropic model",
                            "context_length": int(item.get("context_length") or item.get("context_window") or 0) or None,
                        }
                    )
                cursor = str(data.get("last_id") or (page[-1].get("id") if page else ""))
                if not data.get("has_more"):
                    break
                if not cursor or cursor in seen_cursors:
                    raise RuntimeError("Anthropic model listing returned a repeated pagination cursor.")
                seen_cursors.add(cursor)
                after = cursor
            if not models:
                return {
                    "verified": False,
                    "provider": prov,
                    "authentication_method": auth_method,
                    "verification_timestamp": now_iso(),
                    "available_models": [],
                    "model_capabilities": {},
                    "safe_error_code": "NO_DISCOVERED_MODELS",
                    "user_friendly_message": "The provider returned no accessible models.",
                }

        models = [normalize_model_record(prov, model) for model in models]
        models = self._filter_curated_models(prov, models)
        if prov in ("groq", "openrouter"):
            curated_order = (
                (*GROQ_TEXT_MODEL_IDS, *GROQ_VISION_MODEL_IDS)
                if prov == "groq"
                else ("openrouter/free", *OPENROUTER_VISION_MODEL_IDS, *OPENROUTER_TEXT_MODEL_IDS)
            )
            order = {model_id: index for index, model_id in enumerate(curated_order)}
            models.sort(key=lambda item: order.get(str(item.get("id") or ""), len(order)))
        for m in models:
            caps[m["id"]] = {
            **self._capabilities_from_metadata(prov, m, m["id"]),
            "free_tier": (
                True if m.get("pricing_status") == "free"
                else False if m.get("pricing_status") == "paid"
                else None
            ),
            }

        # Persist encrypted credentials to db.py (saving resolved_key so it persists across restarts)
        db.save_provider_credentials(
            provider=prov,
            api_key=resolved_key if resolved_key else None,
            base_url=resolved_base,
            verified=True,
            authentication_method="user_stored_key" if resolved_key else auth_method,
            available_models=models,
            model_capabilities=caps,
            last_error_code="",
            last_error_message="",
        )

        if activate:
            self.active_provider = prov
            chosen = self._choose_chat_model(models, selected_model, self.active_model)
            self.active_model = chosen["id"] if chosen else ""
            self.active_model_name = chosen["name"] if chosen else ""
            db.update_user_settings(
                {
                    "active_provider": self.active_provider,
                    "active_model": self.active_model,
                    "active_model_name": self.active_model_name,
                    "active_model_is_free": bool(
                        chosen and chosen.get("pricing_status") == "free"
                    ),
                }
            )

        return {
            "verified": True,
            "provider": prov,
            "has_stored_key": bool(resolved_key),
            "masked_key": db.mask_api_key(resolved_key) if resolved_key else "",
            "authentication_method": "user_stored_key" if resolved_key else auth_method,
            "verification_timestamp": now_iso(),
            "available_models": models,
            "model_capabilities": caps,
            "total_models": len(models),
            "user_friendly_message": (
                f"{prov.upper()} verified ({len(models)} models ready)."
                if any(item.get("text") is True for item in models)
                else f"{prov.upper()} credentials verified, but no discovered text chat model is available."
            ),
        }

    def check_vision_compatibility(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        prov = (provider or self.active_provider or "gemini").strip().lower()
        mod = _canonical_model_id(prov, (model or self.active_model or "").strip())

        status = db.get_provider_public_status(prov)
        verified_model = next(
            (
                item
                for item in (status.get("available_models") or [])
                if isinstance(item, dict)
                and _canonical_model_id(prov, item.get("id", "")) == mod
            ),
            None,
        )
        if verified_model and verified_model.get("availability_status") in (
            "temporarily_unavailable",
            "unavailable",
        ):
            return {
                "compatible": False,
                "provider": prov,
                "model": mod,
                "reason": f"Model '{mod}' is currently {verified_model['availability_status'].replace('_', ' ')}.",
                "recommended_models": [
                    _canonical_model_id(prov, item.get("id", ""))
                    for item in (status.get("available_models") or [])
                    if isinstance(item, dict)
                    and item.get("vision") is True
                    and item.get("availability_status") not in ("temporarily_unavailable", "unavailable")
                    and item.get("id")
                ],
            }
        curated_provider = {
            "gemini": mod.startswith("gemini-"),
            "groq": mod in GROQ_VISION_MODEL_IDS,
            "openrouter": mod in (*OPENROUTER_VISION_MODEL_IDS, "openrouter/free"),
        }
        if mod in CURATED_VISION_IDS and curated_provider.get(prov, False):
            return {"compatible": True, "provider": prov, "model": mod, "recommended_models": []}
        if verified_model and verified_model.get("vision") is True:
            return {"compatible": True, "provider": prov, "model": mod, "recommended_models": []}

        rec = [
            _canonical_model_id(prov, str(m.get("id") or m.get("name") or ""))
            for m in (status.get("available_models") or [])
            if isinstance(m, dict) and m.get("vision") and (m.get("id") or m.get("name"))
        ]
        return {
            "compatible": False,
            "provider": prov,
            "model": mod,
            "suggested_model": rec[0] if rec else None,
            "reason": f"Selected model '{mod}' on {prov.upper()} is text-only and does not accept image inputs.",
            "recommended_models": rec,
        }

    def resolve_vision_model(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        allow_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Select a curated vision model without probing providers or starting local inference."""
        prov = (provider or self.active_provider or "gemini").strip().lower()
        mod = _canonical_model_id(prov, (model or self.active_model or "").strip())
        selected = self._configured_curated_vision_model(prov, mod)
        if selected:
            return {
                "compatible": True,
                "routed": False,
                **selected,
                "original_provider": prov,
                "original_model": mod,
                "automatic": True,
                "notice": "",
                "recommended_models": [mod],
            }
        if allow_fallback:
            for candidate_provider, candidate_model in VISION_FALLBACK_PRIORITY:
                selected = self._configured_curated_vision_model(
                    candidate_provider, candidate_model
                )
                if not selected:
                    continue
                logger.info(
                    "Vision model selected automatically: provider=%s model=%s",
                    selected["provider"],
                    selected["model"],
                )
                routed = (selected["provider"], selected["model"]) != (prov, mod)
                return {
                    "compatible": True,
                    "routed": routed,
                    **selected,
                    "original_provider": prov,
                    "original_model": mod,
                    "automatic": True,
                    "notice": (
                        f"Vision/file analysis automatically uses {selected['model_name']} "
                        f"({selected['provider'].upper()}); your chat model remains unchanged."
                        if routed
                        else ""
                    ),
                    "recommended_models": [selected["model"]],
                }
        return {
            "compatible": False,
            "routed": False,
            "provider": prov,
            "model": mod,
            "model_name": mod,
            "original_provider": prov,
            "original_model": mod,
            "automatic": True,
            "notice": "No configured curated vision model is currently available.",
            "recommended_models": [],
        }

    def select_model(
        self,
        task: str = "text",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        allow_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Select a model only from verified provider discovery results."""
        requested_provider = (provider or self.active_provider or "").strip().lower()
        requested_model = _canonical_model_id(
            requested_provider, (model or self.active_model or "").strip()
        )
        required_capability = {
            "text": "text",
            "chat": "text",
            "vision": "vision",
            "document": "document",
            "image": "image_generation",
            "image_generation": "image_generation",
            "audio_input": "audio_input",
            "audio_generation": "audio_generation",
        }.get(task.strip().lower())
        if not required_capability:
            return {"selected": False, "error": f"Unsupported routing task: {task}."}

        if (
            requested_provider == "openrouter"
            and requested_model == "openrouter/free"
            and required_capability in ("text", "vision", "document")
        ):
            status = db.get_provider_public_status("openrouter")
            if status.get("verified") and self.resolve_api_key("openrouter")[0]:
                return {
                    "selected": True,
                    "provider": "openrouter",
                    "model": "openrouter/free",
                    "model_name": "OpenRouter Free Auto-Router",
                    "routed": False,
                }

        providers = [requested_provider]
        if allow_fallback:
            providers.extend(
                p for p in SUPPORTED_PROVIDERS
                if p != "local" and p not in providers
            )
        for current_provider in providers:
            status = self.sanitize_provider_status(
                current_provider, db.get_provider_public_status(current_provider)
            )
            capabilities = status.get("model_capabilities") or {}
            for entry in status.get("available_models") or []:
                if not isinstance(entry, dict):
                    continue
                candidate_id = str(entry.get("id") or "").strip()
                candidate_id = _canonical_model_id(current_provider, candidate_id)
                if not candidate_id:
                    continue
                if not self._is_model_allowed(current_provider, candidate_id):
                    continue
                if current_provider == requested_provider and requested_model and candidate_id != requested_model:
                    continue
                model_caps = capabilities.get(candidate_id) or entry
                if (
                    model_caps.get(required_capability) is True
                    and entry.get("availability_status") != "unavailable"
                ):
                    return {
                        "selected": True,
                        "provider": current_provider,
                        "model": candidate_id,
                        "model_name": entry.get("name") or candidate_id,
                        "routed": current_provider != requested_provider
                        or candidate_id != requested_model,
                    }
            if current_provider == requested_provider:
                requested_model = ""
        return {
            "selected": False,
            "provider": requested_provider,
            "error": f"No verified {task} model is available for the configured provider.",
        }

    def rank_text_models(
        self,
        required_tokens: int = 0,
        excluded: Optional[set[Tuple[str, str]]] = None,
        capability: str = "text",
    ) -> List[Dict[str, Any]]:
        """Rank verified cloud models only; local inference requires explicit selection."""
        excluded_models = excluded or set()
        usage = self.get_model_usage_stats()
        candidates: List[Dict[str, Any]] = []
        for provider_index, provider in enumerate(SUPPORTED_PROVIDERS):
            # Never auto-start Ollama/local models: they can consume substantial RAM/VRAM.
            # Local inference is available only when the user explicitly selects it.
            if provider == "local":
                continue
            status = self.sanitize_provider_status(
                provider, db.get_provider_public_status(provider)
            )
            if not status.get("verified"):
                continue
            api_key, _ = self.resolve_api_key(provider)
            if provider != "local" and not api_key:
                continue
            capabilities = status.get("model_capabilities") or {}
            for entry in status.get("available_models") or []:
                if not isinstance(entry, dict):
                    continue
                model_id = str(entry.get("id") or "").strip()
                model_id = _canonical_model_id(provider, model_id)
                if not model_id or (provider, model_id) in excluded_models:
                    continue
                if not self._is_model_allowed(provider, model_id):
                    continue
                if (provider, model_id) in getattr(self, "_blocked_models", set()):
                    continue
                if capability == "text":
                    lowered_id = model_id.lower()
                    if lowered_id.endswith(":batch") or any(
                        marker in lowered_id
                        for marker in ("-image", "-image-preview", "-tts", "text-to-speech",
                                       "computer-use", "deep-research", "nano-banana")
                    ):
                        continue
                if entry.get("availability_status") in ("unavailable", "temporarily_unavailable"):
                    continue
                model_caps = capabilities.get(model_id) or entry
                if capability == "vision":
                    if model_caps.get("vision") is False:
                        continue
                    try:
                        compatibility = self.check_vision_compatibility(provider, model_id)
                    except ProviderRequestError as exc:
                        logger.warning(
                            "Vision candidate verification failed: provider=%s model=%s error_kind=%s http_status=%s",
                            provider,
                            model_id,
                            exc.failure_kind,
                            exc.status_code,
                        )
                        continue
                    if not compatibility.get("compatible"):
                        continue
                elif model_caps.get(capability) is not True:
                    continue
                context_capacity = (
                    model_caps.get("context_window")
                    or entry.get("context_window")
                    or entry.get("context_length")
                )
                try:
                    context_capacity = int(context_capacity) if context_capacity else None
                except (TypeError, ValueError):
                    context_capacity = None
                if (
                    context_capacity
                    and required_tokens
                    > int(context_capacity * (1 - getattr(self, "_tpm_safety_margin", 0.1)))
                ):
                    continue
                key = (provider, model_id)
                stats = usage.get(key, {})
                rate_limits = stats.get("rate_limits") or {}
                remaining_tokens = rate_limits.get("remaining_tokens")
                observation_age = time.time() - rate_limits.get("observed_at", 0)
                if remaining_tokens is not None and required_tokens and observation_age <= 60:
                    if required_tokens > int(
                        remaining_tokens * (1 - getattr(self, "_tpm_safety_margin", 0.1))
                    ):
                        continue
                pricing_status = entry.get("pricing_status")
                if pricing_status == "free" or provider == "local":
                    cost_rank = 0.0
                elif pricing_status == "paid":
                    pricing = entry.get("pricing") or {}
                    try:
                        cost_rank = float(pricing.get("prompt", 0)) + float(
                            pricing.get("completion", 0)
                        )
                    except (TypeError, ValueError):
                        cost_rank = 1000.0
                else:
                    cost_rank = 500.0
                failures = stats.get("failure_count", 0)
                attempts = stats.get("request_count", 0)
                failure_rate = failures / attempts if attempts else 0.0
                candidates.append(
                    {
                        "provider": provider,
                        "model": model_id,
                        "model_name": entry.get("display_name") or entry.get("name") or model_id,
                        "base_url": status.get("base_url") or "",
                        "context_capacity": context_capacity,
                        "remaining_tokens": remaining_tokens,
                        "cost_rank": cost_rank,
                        "latency_ms": stats.get("average_latency_ms"),
                        "failure_rate": failure_rate,
                        "request_count": attempts,
                        "provider_index": provider_index,
                    }
                )
        candidates.sort(
            key=lambda item: (
                item["failure_rate"],
                item["cost_rank"],
                item["latency_ms"] if item["latency_ms"] is not None else math.inf,
                item["request_count"],
                item["provider_index"],
                item["model"],
            )
        )
        return candidates

    def _model_capacity_check(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
    ) -> Tuple[bool, int, str]:
        estimate = estimate_request_tokens(messages)
        status = db.get_provider_public_status(provider)
        entry = next(
            (
                item
                for item in status.get("available_models") or []
                if isinstance(item, dict)
                and _canonical_model_id(provider, item.get("id", "")) == model
            ),
            {},
        )
        caps = status.get("model_capabilities") or {}
        model_caps = (
            caps.get(model) or caps.get(f"models/{model}")
            if isinstance(caps, dict)
            else {}
        )
        context_capacity = (
            (model_caps or {}).get("context_window")
            or entry.get("context_window")
            or entry.get("context_length")
        )
        try:
            context_capacity = int(context_capacity) if context_capacity else None
        except (TypeError, ValueError):
            context_capacity = None
        required = estimate + max_tokens
        safety_margin = getattr(self, "_tpm_safety_margin", 0.1)
        if context_capacity and required > int(context_capacity * (1 - safety_margin)):
            return (
                False,
                required,
                f"Estimated request ({required} tokens) exceeds {provider}/{model} "
                f"context capacity ({context_capacity} tokens with safety margin).",
            )
        stats = self.get_model_usage_stats().get((provider, model), {})
        rate_limits = stats.get("rate_limits") or {}
        remaining = rate_limits.get("remaining_tokens")
        observation_age = time.time() - rate_limits.get("observed_at", 0)
        if remaining is not None and observation_age <= 60 and required > int(remaining * (1 - safety_margin)):
            return (
                False,
                required,
                f"Estimated request ({required} tokens) exceeds the currently reported "
                f"TPM capacity ({remaining} tokens) for {provider}/{model}.",
            )
        return True, required, ""

    # ========================================================================
    # UNIFIED GENERATION & STREAMING ENGINE
    # ========================================================================

    def _generate_with_retries(
        self,
        messages: List[Dict[str, Any]],
        provider: str,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        on_retry: Optional[Callable[[int, ProviderRequestError], None]],
        retry_rate_limits: bool = True,
    ) -> Dict[str, Any]:
        max_attempts = 1 if provider == "local" else 2
        for attempt in range(1, max_attempts + 1):
            request_started = time.perf_counter()
            try:
                if provider == "gemini":
                    result = self._generate_gemini(
                        messages, model, temperature, max_tokens, json_mode, api_key
                    )
                elif provider in ("groq", "openrouter", "openai", "local"):
                    result = self._generate_openai_compatible(
                        provider,
                        messages,
                        model,
                        base_url,
                        temperature,
                        max_tokens,
                        json_mode,
                        api_key,
                    )
                elif provider == "anthropic":
                    result = self._generate_anthropic(
                        messages, model, temperature, max_tokens, api_key
                    )
                else:
                    raise ValueError(f"Unsupported AI provider: {provider}.")
                self._record_model_usage(
                    provider,
                    model,
                    (time.perf_counter() - request_started) * 1000,
                    True,
                    result.get("_rate_limit_headers"),
                )
                self._record_model_state(provider, model, "available", "verified")
                return result
            except ProviderRequestError as exc:
                self._record_model_usage(
                    provider,
                    model,
                    (time.perf_counter() - request_started) * 1000,
                    False,
                    exc.response_headers,
                )
                logger.warning(
                    "AI request attempt %d/%d failed: provider=%s model=%s http_status=%s reason=%s",
                    attempt,
                    max_attempts,
                    provider,
                    model,
                    exc.status_code or "network",
                    exc.reason,
                )
                if (
                    not exc.retryable
                    or attempt == max_attempts
                    or exc.failure_kind == "context_capacity"
                    or (
                        exc.failure_kind == "rate_limit"
                        and exc.retry_after is not None
                        and exc.retry_after > 30
                    )
                    or (exc.failure_kind == "rate_limit" and not retry_rate_limits)
                ):
                    if exc.status_code in (0, 408, 500, 502, 503, 504):
                        self._record_model_state(
                            provider, model, "temporarily_unavailable"
                        )
                    elif exc.status_code in (404, 410):
                        self._record_model_state(provider, model, "unavailable")
                        self._blocked_models.add((provider, model))
                    elif exc.status_code == 402:
                        self._blocked_models.add((provider, model))
                        self._record_model_state(provider, model, "temporarily_unavailable")
                    elif exc.status_code == 400 and any(
                        marker in exc.reason.lower()
                        for marker in ("unsupported image", "image not supported", "does not support image")
                    ):
                        self._record_model_state(
                            provider,
                            model,
                            "available",
                            "unsupported",
                            vision_capable=False,
                        )
                    raise
                if on_retry:
                    on_retry(attempt + 1, exc)
                delay = min(
                    1.5,
                    exc.retry_after
                    if exc.retry_after is not None
                    else 0.25 * (2 ** (attempt - 1)),
                )
                time.sleep(delay)
        raise RuntimeError("AI generation ended without a response.")

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
        on_retry: Optional[Callable[[int, ProviderRequestError], None]] = None,
        vision_provider: Optional[str] = None,
        vision_model: Optional[str] = None,
        allow_vision_fallback: bool = False,
        operation: str = "chat",
        selection_mode: str = "manual",
        allow_model_fallback: bool = False,
        document_context: bool = False,
    ) -> Dict[str, Any]:
        routing_started = time.perf_counter()
        requested_provider = (provider or self.active_provider or "gemini").strip().lower()
        requested_model = _canonical_model_id(
            requested_provider, (model or self.active_model or "").strip()
        )
        original_provider = requested_provider
        original_model = requested_model
        unsafe_active_model = bool(
            self.active_model
            and not self._is_model_allowed(self.active_provider, self.active_model)
        )
        if unsafe_active_model or not self._is_model_allowed(requested_provider, requested_model):
            selection = self._reset_active_model_to_safe_fallback()
            requested_provider = selection["provider"]
            requested_model = selection["model"]
            if requested_provider != original_provider:
                base_url = None
        mode = selection_mode.strip().lower()
        automatic = mode == "automatic"
        has_images = any(bool(m.get("images")) for m in messages if isinstance(m, dict))
        has_document_context = bool(
            document_context
            or any(
                isinstance(message, dict)
                and (
                    message.get("document_context")
                    or message.get("file_context")
                    or message.get("files")
                )
                for message in messages
            )
            or any(
                marker in str(message.get("content") or "").upper()
                for message in messages
                if isinstance(message, dict)
                for marker in (
                    "ATTACHED DOCUMENT CONTEXT",
                    "ACTIVE DOCUMENT METADATA",
                    "DOCUMENT GROUNDING:",
                )
            )
        )
        has_vision_context = has_images or has_document_context
        fallback_allowed = automatic or allow_model_fallback or has_vision_context
        routing_info: Optional[Dict[str, Any]] = None
        if has_vision_context:
            routing_info = self.resolve_vision_model(
                original_provider,
                original_model,
                allow_fallback=True,
            )
        elif not automatic:
            selection = self.select_model(
                "text",
                requested_provider,
                requested_model,
                allow_fallback=False,
            )
            if not selection.get("selected"):
                if not allow_model_fallback:
                    raise RuntimeError(selection.get("error") or "No verified text model is available.")
                candidates = self.rank_text_models(capability="text")
                if not candidates:
                    raise RuntimeError(selection.get("error") or "No verified text model is available.")
                requested_provider = candidates[0]["provider"]
                requested_model = candidates[0]["model"]
                base_url = candidates[0].get("base_url") or base_url
            else:
                requested_provider = selection["provider"]
                requested_model = selection["model"]
        if routing_info is not None:
            if not routing_info.get("compatible"):
                raise RuntimeError(routing_info.get("notice") or "No verified vision model is available.")
            requested_provider = routing_info["provider"]
            requested_model = routing_info["model"]
            base_url = routing_info.get("base_url") or base_url
            routing_info["original_provider"] = original_provider
            routing_info["original_model"] = original_model
            routing_info["routed"] = bool(
                routing_info.get("routed")
                or requested_provider != original_provider
                or requested_model != original_model
            )
            if routing_info["routed"] and not routing_info.get("notice"):
                routing_info["notice"] = (
                    f"Image understanding uses {routing_info.get('model_name') or requested_model} "
                    f"({requested_provider.upper()}) while chat remains on {original_model or 'the selected text model'}."
                )
        preferred_operation_model = (requested_provider, requested_model)
        requested_capacity_ok, required_tokens, capacity_reason = self._model_capacity_check(
            requested_provider, requested_model, messages, max_tokens
        )
        if automatic:
            requested_capability = "vision" if has_vision_context else "text"
            candidates = self.rank_text_models(
                required_tokens,
                capability=requested_capability,
            )
            if candidates and not has_vision_context:
                selected = candidates[0]
                requested_provider = selected["provider"]
                requested_model = selected["model"]
                base_url = selected.get("base_url") or base_url
                requested_capacity_ok, required_tokens, capacity_reason = self._model_capacity_check(
                    requested_provider, requested_model, messages, max_tokens
                )
                if routing_info is not None and (
                    requested_provider != routing_info["provider"]
                    or requested_model != routing_info["model"]
                ):
                    routing_info.update(
                        {
                            "provider": requested_provider,
                            "model": requested_model,
                            "model_name": selected.get("model_name") or requested_model,
                            "routed": True,
                        }
                    )

        candidates_to_try: List[Dict[str, Any]] = []
        if requested_capacity_ok:
            candidates_to_try.append(
                {
                    "provider": requested_provider,
                    "model": requested_model,
                    "base_url": base_url,
                }
            )
        elif not fallback_allowed:
            raise RuntimeError(capacity_reason)

        if fallback_allowed:
            excluded = {(requested_provider, requested_model)}
            if has_vision_context:
                for candidate_provider, candidate_model in VISION_FALLBACK_PRIORITY:
                    if (candidate_provider, candidate_model) in excluded:
                        continue
                    candidate = self._configured_curated_vision_model(
                        candidate_provider, candidate_model
                    )
                    if candidate:
                        candidates_to_try.append(candidate)
                        excluded.add((candidate_provider, candidate_model))
            else:
                fallback_candidates = self.rank_text_models(
                    required_tokens,
                    excluded,
                    capability="text",
                )
                for candidate in fallback_candidates[:2]:
                    candidates_to_try.append(candidate)
        if not candidates_to_try:
            raise RuntimeError(
                capacity_reason
                or f"No discovered compatible model has capacity for the {operation} request."
            )

        last_error: Optional[ProviderRequestError] = None
        res: Optional[Dict[str, Any]] = None
        selected_provider = requested_provider
        selected_model = requested_model
        selected_model_name = requested_model
        for index, candidate in enumerate(candidates_to_try):
            candidate_provider = candidate["provider"]
            candidate_model = candidate["model"]
            candidate_base_url = candidate.get("base_url") or base_url
            fits, _, reason = self._model_capacity_check(
                candidate_provider, candidate_model, messages, max_tokens
            )
            if not fits:
                capacity_reason = reason
                continue
            candidate_key = (
                api_key
                if candidate_provider == original_provider
                else None
            )
            try:
                res = self._generate_with_retries(
                    messages,
                    candidate_provider,
                    candidate_model,
                    candidate_key,
                    candidate_base_url,
                    temperature,
                    max_tokens,
                    json_mode,
                    on_retry,
                    retry_rate_limits=not fallback_allowed,
                )
                selected_provider = candidate_provider
                selected_model = candidate_model
                selected_model_name = candidate.get("model_name") or candidate_model
                if index or (
                    not automatic
                    and (candidate_provider, candidate_model) != preferred_operation_model
                ):
                    res["model_fallback"] = True
                    cause = (
                        f"after {last_error.failure_kind} from "
                        f"{last_error.provider}/{last_error.model} "
                        f"(HTTP {last_error.status_code})"
                        if last_error is not None
                        else "after the preferred candidate lacked available capacity"
                    )
                    res["model_fallback_reason"] = (
                        f"Allocated {operation} to {candidate_provider}/{candidate_model} {cause}."
                    )
                break
            except ProviderRequestError as exc:
                last_error = exc
                if exc.status_code in (401, 403):
                    logger.error(
                        "Model routing stopped after provider authentication failure: operation=%s provider=%s model=%s status=%s",
                        operation,
                        candidate_provider,
                        candidate_model,
                        exc.status_code,
                    )
                    raise
                model_unavailable = exc.status_code in (404, 410, 500, 502, 503, 504) or (
                    exc.status_code == 400
                    and "image" in exc.reason.lower()
                    and any(
                        marker in exc.reason.lower()
                        for marker in ("unsupported", "not support", "does not accept")
                    )
                )
                if has_vision_context and model_unavailable:
                    vision_fallback = self._find_vision_fallback(
                        candidate_provider,
                        candidate_model,
                        candidate_base_url or "",
                        allow_cross_provider=allow_vision_fallback or automatic or has_vision_context,
                    )
                    if vision_fallback and not any(
                        item["provider"] == vision_fallback["provider"]
                        and item["model"] == vision_fallback["model"]
                        for item in candidates_to_try
                    ):
                        candidates_to_try.append(
                            {
                                "provider": vision_fallback["provider"],
                                "model": vision_fallback["model"],
                                "model_name": vision_fallback["model_name"],
                                "base_url": vision_fallback["base_url"],
                            }
                        )
                        routing_info = {
                            "compatible": True,
                            "routed": True,
                            "provider": vision_fallback["provider"],
                            "model": vision_fallback["model"],
                            "model_name": vision_fallback["model_name"],
                            "original_provider": original_provider,
                            "original_model": original_model,
                            "notice": (
                                f"Vision model {candidate_model} was unavailable; "
                                f"the request was routed to {vision_fallback['model_name']}."
                            ),
                        }
                        continue
                if not fallback_allowed:
                    raise
                logger.info(
                    "Model fallback for operation=%s after %s status=%s retry_after=%s: provider=%s model=%s",
                    operation,
                    exc.failure_kind,
                    exc.status_code,
                    exc.retry_after,
                    candidate_provider,
                    candidate_model,
                )
        if res is None:
            if last_error is not None:
                if has_vision_context:
                    raise RuntimeError(
                        "No available compatible vision model could process this image or document. "
                        f"The last attempt failed on {last_error.provider}/{last_error.model} "
                        f"({last_error.failure_kind}, HTTP {last_error.status_code}): "
                        f"{last_error.reason}"
                    ) from last_error
                raise last_error
            raise RuntimeError(
                (
                    "No available compatible vision model has sufficient context or provider capacity."
                    if has_vision_context
                    else capacity_reason
                    or f"No available compatible model could handle {operation}."
                )
            )
        res.setdefault("operation", operation)
        res.setdefault("estimated_input_tokens", estimate_request_tokens(messages))
        if selected_provider != original_provider or selected_model != original_model:
            res["operation_model_changed"] = True
            res["requested_model"] = original_model
        if has_vision_context:
            res["vision_selection_mode"] = "automatic"
            res["vision_model_name"] = (
                routing_info.get("model_name") if routing_info else selected_model
            )
            res["vision_selection_reason"] = (
                "Ranked from discovered models with verified image-input support, capacity, "
                "availability, reliability, usage, latency, and pricing metadata."
            )
        reason = (
            res.get("model_fallback_reason")
            or (
                "Selected from discovered compatible models using availability, capacity, "
                "usage, latency, and pricing metadata."
                if automatic
                else "Used the manually preferred model."
            )
        )
        res["model_selection_reason"] = reason
        logger.info(
            "AI routing complete: operation=%s mode=%s intent_model=%s/%s selected_model=%s/%s fallback=%s latency_ms=%.1f reason=%s",
            operation,
            mode,
            original_provider,
            original_model,
            selected_provider,
            selected_model,
            bool(res.get("model_fallback")),
            (time.perf_counter() - routing_started) * 1000,
            reason,
        )
        if routing_info is not None:
            routing_info.update(
                {
                    "provider": selected_provider,
                    "model": selected_model,
                    "model_name": selected_model_name,
                    "routed": (
                        selected_provider != original_provider
                        or selected_model != original_model
                    ),
                }
            )
        if routing_info is not None and routing_info.get("routed"):
            routing_info.update(
                {
                    "notice": routing_info.get("notice")
                    or (
                        f"Vision analysis was routed from {preferred_operation_model[0]}/"
                        f"{preferred_operation_model[1]} to {selected_provider}/{selected_model} "
                        "for availability or capacity."
                    ),
                }
            )

        if routing_info:
            res["routed_vision"] = bool(routing_info.get("routed"))
            res["routed_vision_model"] = routing_info["model"]
            res["routed_vision_model_name"] = routing_info["model_name"]
            res["routed_vision_provider"] = routing_info["provider"]
            res["original_chat_model"] = routing_info["original_model"]
            res["vision_routing_notice"] = routing_info.get("notice", "")
        return res

    def _generate_gemini(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        explicit_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        api_key, _ = self.resolve_api_key("gemini", explicit_api_key)
        if not api_key:
            raise RuntimeError(
                "Gemini API key is not configured. Open Settings -> AI Providers to enter and verify your API key."
            )

        real_model = resolve_real_gemini_model(model)
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content_text = str(msg.get("content") or "")
            if role == "system":
                if content_text.strip():
                    system_parts.append(content_text.strip())
                continue

            parts: List[Dict[str, Any]] = []
            if content_text:
                parts.append({"text": content_text})

            for img in msg.get("images") or []:
                data_url = img.get("dataUrl") or img.get("data") or ""
                mime = img.get("mimeType") or "image/png"
                b64_data = data_url
                if data_url.startswith("data:"):
                    m = re.match(r"^data:([^;]+);base64,(.+)$", data_url, re.S)
                    if m:
                        mime = m.group(1)
                        b64_data = m.group(2)
                if b64_data:
                    parts.append({"inlineData": {"mimeType": mime, "data": b64_data}})

            if parts:
                contents.append(
                    {
                        "role": "model" if role == "assistant" else "user",
                        "parts": parts,
                    }
                )

        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(real_model)}:generateContent?key={urllib.parse.quote(api_key)}"
        code, data = self._http_json(url, method="POST", body=payload, timeout=60.0)

        if code != 200:
            err_msg = (
                (data.get("error") or {}).get("message")
                if isinstance(data.get("error"), dict)
                else f"Gemini API error (HTTP {code})"
            )
            raise ProviderRequestError(
                "gemini",
                model,
                code,
                str(err_msg),
                self._retry_after_seconds(data),
                data.get("__response_headers__", {}),
            )

        candidates = data.get("candidates") or []
        text_out = ""
        if candidates:
            c_parts = (candidates[0].get("content") or {}).get("parts") or []
            text_out = "".join(p.get("text", "") for p in c_parts if isinstance(p, dict))

        return {
            "text": text_out,
            "model": model,
            "provider": "gemini",
            "_rate_limit_headers": data.get("__response_headers__", {}),
        }

    def _generate_openai_compatible(
        self,
        provider: str,
        messages: List[Dict[str, Any]],
        model: str,
        base_url: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        explicit_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        api_key, _ = self.resolve_api_key(provider, explicit_api_key)
        if provider != "local" and not api_key:
            raise RuntimeError(
                f"{provider.upper()} API key is not configured. Open Settings -> AI Providers to enter and verify your key."
            )

        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
        elif provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
        else:
            clean_base = (base_url or self.local_endpoint or "http://localhost:11434").rstrip("/")
            url = f"{clean_base}/v1/chat/completions"

        oai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = str(msg.get("content") or "")
            imgs = msg.get("images") or []
            if imgs and role == "user":
                content_arr: List[Dict[str, Any]] = [{"type": "text", "text": text or "Analyze this image."}]
                for img in imgs:
                    d_url = img.get("dataUrl") or img.get("data") or ""
                    if not d_url.startswith("data:"):
                        d_url = f"data:{img.get('mimeType', 'image/png')};base64,{d_url}"
                    content_arr.append({"type": "image_url", "image_url": {"url": d_url}})
                oai_messages.append({"role": role, "content": content_arr})
            else:
                oai_messages.append({"role": role, "content": text})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and provider in ("openai", "groq"):
            payload["response_format"] = {"type": "json_object"}

        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        if provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "DocPilot VISION AI"

        request_timeout = 18.0 if provider == "local" else 45.0
        code, data = self._http_json(
            url, method="POST", headers=headers, body=payload, timeout=request_timeout
        )
        if code != 200:
            err_msg = (
                (data.get("error") or {}).get("message")
                if isinstance(data.get("error"), dict)
                else f"{provider.upper()} API error (HTTP {code})"
            )
            raise ProviderRequestError(
                provider,
                model,
                code,
                str(err_msg),
                self._retry_after_seconds(data),
                data.get("__response_headers__", {}),
            )

        choices = data.get("choices") or []
        text_out = ""
        if choices:
            text_out = ((choices[0].get("message") or {}).get("content")) or ""
        return {
            "text": text_out,
            "model": model,
            "provider": provider,
            "_rate_limit_headers": data.get("__response_headers__", {}),
        }

    def _generate_anthropic(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        explicit_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        api_key, _ = self.resolve_api_key("anthropic", explicit_api_key)
        if not api_key:
            raise RuntimeError(
                "Anthropic API key is not configured. Open Settings -> AI Providers to enter and verify your key."
            )

        system_parts: List[str] = []
        anth_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = str(msg.get("content") or "")
            if role == "system":
                if text.strip():
                    system_parts.append(text.strip())
                continue
            content: Any = text or "..."
            images = msg.get("images") or []
            if images and role == "user":
                content_blocks: List[Dict[str, Any]] = []
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for image in images:
                    data_url = image.get("dataUrl") or image.get("data") or ""
                    mime_type = image.get("mimeType") or "image/png"
                    encoded = data_url
                    if data_url.startswith("data:"):
                        match = re.match(r"^data:([^;]+);base64,(.+)$", data_url, re.S)
                        if match:
                            mime_type, encoded = match.groups()
                    if encoded:
                        content_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": encoded,
                                },
                            }
                        )
                content = content_blocks
            anth_messages.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content,
                }
            )

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anth_messages or [{"role": "user", "content": "Hello"}],
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        code, data = self._http_json(
            "https://api.anthropic.com/v1/messages",
            method="POST",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            body=payload,
            timeout=60.0,
        )
        if code != 200:
            err_msg = (
                (data.get("error") or {}).get("message")
                if isinstance(data.get("error"), dict)
                else f"Anthropic API error (HTTP {code})"
            )
            raise ProviderRequestError(
                "anthropic",
                model,
                code,
                str(err_msg),
                self._retry_after_seconds(data),
                data.get("__response_headers__", {}),
            )

        blocks = data.get("content") or []
        text_out = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        return {
            "text": text_out,
            "model": model,
            "provider": "anthropic",
            "_rate_limit_headers": data.get("__response_headers__", {}),
        }

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "You are a precise academic AI. Output valid JSON only.",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        operation: str = "chat",
        selection_mode: str = "manual",
        allow_model_fallback: bool = False,
        return_metadata: bool = False,
    ) -> Any:
        res = self.generate_text(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=0.2,
            max_tokens=max_tokens,
            json_mode=True,
            operation=operation,
            selection_mode=selection_mode,
            allow_model_fallback=allow_model_fallback,
        )
        raw = res.get("text", "").strip()
        parsed = parse_json_safely(raw)
        if not return_metadata:
            return parsed
        return parsed, {
            "provider": res.get("provider"),
            "model": res.get("model"),
            "operation": res.get("operation"),
            "selection_reason": res.get("model_selection_reason"),
            "fallback": bool(res.get("model_fallback")),
            "fallback_reason": res.get("model_fallback_reason"),
        }

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a normalized 64-dimensional semantic/hashed vector for fast local RAG
        that works both online and offline without external latency spikes.
        """
        vec = [0.0] * 64
        tokens = re.findall(r"[a-z0-9_]{2,}", (text or "").lower())
        if not tokens:
            return vec
        for tok in tokens[:500]:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % 64
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 5) for v in vec]


def parse_json_safely(raw_text: str) -> Any:
    """Extracts and parses JSON object or array from an LLM response."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Find first { ... } or [ ... ]
    obj_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if obj_match:
        candidate = obj_match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            # Repair trailing commas
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except Exception:
                pass
    return None


# Singleton AI Provider Manager instance
ai_manager = AIProviderManager()
