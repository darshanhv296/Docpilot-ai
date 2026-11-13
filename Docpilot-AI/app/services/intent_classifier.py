"""Lightweight, context-aware routing for incoming chat messages."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.database.db import SUPPORTED_PROVIDERS, db
from app.services.ai_provider import ProviderRequestError, ai_manager

logger = logging.getLogger(__name__)

INTENTS = {
    "NORMAL_CONVERSATION",
    "STUDY_EXPLANATION",
    "QUIZ_GENERATION",
    "CHEAT_SHEET_GENERATION",
    "RECALL_CARD_GENERATION",
    "DOCUMENT_SUMMARY",
    "IMAGE_UNDERSTANDING",
    "OTHER_APPLICATION_ACTIONS",
}
STUDY_INTENTS = {
    "STUDY_EXPLANATION",
    "QUIZ_GENERATION",
    "CHEAT_SHEET_GENERATION",
    "RECALL_CARD_GENERATION",
    "DOCUMENT_SUMMARY",
}
_last_assignment: Dict[str, Any] = {
    "provider": None,
    "model": None,
    "selection_reason": "No classifier model has been selected yet.",
}
_rejected_classifier_models: set[Tuple[str, str]] = set()
_classification_count = 0
_classification_latency_ms = 0.0
_state_lock = threading.Lock()

_FALLBACK_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "QUIZ_GENERATION",
        re.compile(
            r"\b(?:make|create|generate|prepare)\b.{0,50}\b(?:quiz|practice test|practice questions|questions and answers|mcq|mcqs|multiple.choice questions)\b"
            r"|\btest my knowledge\b|\bquiz me\b|\btest me on\b",
            re.I,
        ),
    ),
    (
        "CHEAT_SHEET_GENERATION",
        re.compile(
            r"\b(?:make|create|generate|prepare)\b.{0,45}\b(?:cheat[\s-]?sheet|revision notes|exam notes|exam revision material)\b"
            r"|\bextract important points\b.{0,30}\b(?:revision|exam)\b",
            re.I,
        ),
    ),
    (
        "RECALL_CARD_GENERATION",
        re.compile(r"\b(?:make|create|generate|prepare)\b.{0,35}\b(?:flashcards?|recall cards?|memory cards?)\b", re.I),
    ),
    (
        "DOCUMENT_SUMMARY",
        re.compile(r"\b(?:summari[sz]e|give me a summary of|extract key points from)\b.{0,50}\b(?:document|chapter|file|notes|material|this)\b", re.I),
    ),
    (
        "STUDY_EXPLANATION",
        re.compile(r"\b(?:teach me|explain|help me understand|walk me through|study)\b.{0,60}\w", re.I),
    ),
)


def _explicit_study_intent(message: str) -> Optional[str]:
    for candidate_intent, pattern in _FALLBACK_PATTERNS:
        if pattern.search(message):
            return candidate_intent
    return None


def _fallback_topic(message: str) -> Optional[str]:
    match = re.search(r"\b(?:about|on|from|for|using)\s+(.+)$", message, re.I)
    if not match:
        return None
    topic = re.sub(r"[?.!,;]+$", "", match.group(1).strip())
    return topic[:160] or None


def _contextual_followup(message: str, prior_messages: List[Dict[str, Any]]) -> Optional[Tuple[str, Optional[str]]]:
    if not re.search(r"\b(?:give me|create|make)\s+(?:\d+|one|two|three|four|five|some)\s+more\b", message, re.I):
        return None
    context = "\n".join(
        str(item.get("content") or "")
        for item in prior_messages[-6:]
        if isinstance(item, dict)
    )
    if re.search(r"\b(?:quiz|practice test|test questions)\b", context, re.I):
        for item in reversed(prior_messages[-6:]):
            if isinstance(item, dict) and item.get("role") == "user":
                topic = _fallback_topic(str(item.get("content") or ""))
                if topic:
                    return "QUIZ_GENERATION", topic
        topic_match = re.search(r"\bquiz\s+(?:on|about)\s+([^.!?\n]+)", context, re.I)
        topic = topic_match.group(1).strip() if topic_match else None
        if topic:
            topic = re.sub(r"\s+(?:created|generated|made)\b.*$", "", topic, flags=re.I)
        return "QUIZ_GENERATION", topic[:160] if topic else None
    if re.search(r"\b(?:flashcards?|recall cards?|memory cards?)\b", context, re.I):
        for item in reversed(prior_messages[-6:]):
            if isinstance(item, dict) and item.get("role") == "user":
                topic = _fallback_topic(str(item.get("content") or ""))
                if topic:
                    return "RECALL_CARD_GENERATION", topic
        topic_match = re.search(r"\b(?:cards?|flashcards?)\s+(?:on|about|for)\s+([^.!?\n]+)", context, re.I)
        return "RECALL_CARD_GENERATION", topic_match.group(1).strip()[:160] if topic_match else None
    return None


def _small_local_model(model: Dict[str, Any]) -> bool:
    description = " ".join(
        str(model.get(field) or "")
        for field in ("desc", "description", "parameter_size", "name", "id")
    )
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:b|billion)\b", description, re.I)
    return bool(match and float(match.group(1)) <= 8)


def _price_rank(model: Dict[str, Any], provider: str) -> float:
    if provider == "local":
        return 10.0
    status = model.get("pricing_status")
    if status == "free":
        return 0.0
    if status == "paid":
        pricing = model.get("pricing")
        if isinstance(pricing, dict):
            try:
                return 1.0 + float(pricing.get("prompt", 0)) + float(
                    pricing.get("completion", 0)
                )
            except (TypeError, ValueError):
                pass
        return 2.0
    return 1000.0


def _efficiency_rank(model: Dict[str, Any]) -> float:
    for field in ("parameter_size", "parameter_count"):
        value = model.get(field)
        if value is None:
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*(b|billion)?", str(value), re.I)
        if match:
            return float(match.group(1))
    identity = " ".join(
        str(model.get(field) or "")
        for field in ("display_name", "name", "id")
    ).lower()
    if "nano-banana" in identity:
        return math.inf
    for rank, marker in enumerate(("nano", "mini", "small", "flash", "haiku", "instant")):
        if marker in identity:
            return float(rank)
    return math.inf


def _classifier_candidates() -> List[Dict[str, Any]]:
    usage = ai_manager.get_model_usage_stats()
    candidates: List[Dict[str, Any]] = []
    provider_order = {provider: index for index, provider in enumerate(SUPPORTED_PROVIDERS)}
    with _state_lock:
        rejected_models = set(_rejected_classifier_models)
    for provider in SUPPORTED_PROVIDERS:
        # The classifier is an automatic background task; never start Ollama here.
        # Local models are reserved for explicit user-selected inference only.
        if provider == "local":
            continue
        status = db.get_provider_public_status(provider)
        if not status.get("verified"):
            continue
        capabilities = status.get("model_capabilities") or {}
        for model in status.get("available_models") or []:
            if not isinstance(model, dict):
                continue
            model_id = str(model.get("id") or "").strip()
            if (
                not model_id
                or model_id.lower().endswith(":batch")
                or any(marker in model_id.lower() for marker in (
                    "-image", "-image-preview", "-tts", "text-to-speech",
                    "computer-use", "deep-research", "nano-banana",
                ))
                or (provider, model_id) in rejected_models
            ):
                continue
            model_caps = capabilities.get(model_id) or model
            if model_caps.get("text") is not True:
                continue
            availability = model.get("availability_status") or model.get("status")
            if availability in ("unavailable", "temporarily_unavailable"):
                continue
            verification = model.get("verification_status")
            if verification in ("unsupported", "unavailable"):
                continue
            if provider == "local" and not _small_local_model(model):
                continue
            stats = usage.get((provider, model_id), {})
            candidates.append(
                {
                    "provider": provider,
                    "model": model_id,
                    "model_name": model.get("display_name") or model.get("name") or model_id,
                    "base_url": status.get("base_url") or "",
                    "request_count": int(stats.get("request_count") or 0),
                    "average_latency_ms": stats.get("average_latency_ms"),
                    "price_rank": _price_rank(model, provider),
                    "efficiency_rank": _efficiency_rank(model),
                    "provider_order": provider_order[provider],
                }
            )
    has_usage = any(candidate["request_count"] for candidate in candidates)
    candidates.sort(
        key=lambda candidate: (
            candidate["request_count"] if has_usage else 0,
            candidate["price_rank"],
            candidate["average_latency_ms"]
            if candidate["average_latency_ms"] is not None
            else math.inf,
            candidate["efficiency_rank"],
            candidate["provider_order"],
            candidate["model"],
        )
    )
    for candidate in candidates:
        if has_usage:
            candidate["selection_reason"] = (
                f"Least-used compatible model ({candidate['request_count']} recorded "
                "provider requests), with cost and observed latency as tie-breakers."
            )
        elif candidate["price_rank"] < 1000:
            candidate["selection_reason"] = (
                "Usage statistics are not available yet; selected the lowest-cost "
                "compatible discovered model, using observed latency and reported "
                "model size as tie-breakers."
            )
        elif math.isfinite(candidate["efficiency_rank"]):
            candidate["selection_reason"] = (
                "Usage and pricing statistics are unavailable; selected a compatible "
                "model labelled or reported as lightweight, then used observed latency "
                "and stable provider order."
            )
        else:
            candidate["selection_reason"] = (
                "Usage and pricing statistics are unavailable; selected a compatible "
                "discovered model using observed latency and stable provider order."
            )
    return candidates


def _record_classification_latency(elapsed_ms: float) -> None:
    global _classification_count, _classification_latency_ms
    with _state_lock:
        _classification_count += 1
        _classification_latency_ms += elapsed_ms


def _validated_result(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Intent classifier returned a non-object JSON value.")
    intent = value.get("intent")
    confidence = value.get("confidence")
    study_action = value.get("study_action")
    action_type = value.get("action_type")
    topic = value.get("topic")
    if intent not in INTENTS:
        raise ValueError("Intent classifier returned an unsupported intent.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("Intent classifier confidence must be numeric.")
    if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise ValueError("Intent classifier confidence must be between zero and one.")
    if not isinstance(study_action, bool) or study_action != (intent in STUDY_INTENTS):
        raise ValueError("Intent classifier study_action does not match its intent.")
    if intent == "NORMAL_CONVERSATION":
        if action_type is not None:
            raise ValueError("Normal conversation must not request an application action.")
    elif action_type != intent:
        raise ValueError("Intent classifier action_type does not match its intent.")
    if topic is not None and (not isinstance(topic, str) or len(topic) > 160):
        raise ValueError("Intent classifier topic must be null or a short string.")
    return {
        "intent": intent,
        "confidence": float(confidence),
        "study_action": study_action,
        "action_type": action_type,
        "topic": topic.strip() if isinstance(topic, str) and topic.strip() else None,
    }


def _fallback_intent(
    message: str,
    images: List[Any],
    prior_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if images:
        intent = "IMAGE_UNDERSTANDING"
        topic = None
    else:
        contextual = _contextual_followup(message, prior_messages)
        intent = contextual[0] if contextual else "NORMAL_CONVERSATION"
        topic = contextual[1] if contextual else None
        if not contextual:
            for candidate_intent, pattern in _FALLBACK_PATTERNS:
                if pattern.search(message):
                    intent = candidate_intent
                    break
            topic = _fallback_topic(message) if intent in STUDY_INTENTS else None
    is_study_action = intent in STUDY_INTENTS
    return {
        "intent": intent,
        "confidence": 0.55 if intent != "NORMAL_CONVERSATION" else 0.5,
        "study_action": is_study_action,
        "action_type": intent if intent != "NORMAL_CONVERSATION" else None,
        "topic": topic if is_study_action else None,
    }


def classify_user_intent(
    message: str,
    prior_messages: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Classify a chat turn with a dedicated lightweight model and safe fallback."""
    global _classification_count, _classification_latency_ms, _last_assignment
    started = time.perf_counter()
    image_list = images if isinstance(images, list) else []

    # Avoid a second LLM request before ordinary chat. Images and clear study
    # requests are deterministic; reserve model classification for ambiguous app actions.
    lower_message = (message or "").lower()
    action_markers = (
        "generate file", "export ", "create document", "make document",
        "download ", "open settings", "change model", "switch model",
        "delete conversation", "delete chat", "rename chat", "clear history",
        "upload document", "attach document", "remove document", "delete document",
        "pause rag", "resume rag", "detach document", "manage documents",
        "change theme", "change voice", "create workspace", "switch workspace",
    )
    explicit = _explicit_study_intent(message)
    if image_list or explicit or not any(marker in lower_message for marker in action_markers):
        fast_result = _fallback_intent(message, image_list, prior_messages or [])
        elapsed_ms = (time.perf_counter() - started) * 1000
        _record_classification_latency(elapsed_ms)
        logger.info(
            "Intent classified: provider=none model=none reason=deterministic_fast_path latency_ms=%.2f intent=%s",
            elapsed_ms,
            fast_result["intent"],
        )
        return {
            **fast_result,
            "classifier_provider": None,
            "classifier_model": None,
            "selection_reason": "Deterministic fast path; classifier model call skipped.",
            "latency_ms": elapsed_ms,
            "used_fallback": True,
        }
    context_lines = [
        f"{'assistant' if item.get('role') == 'assistant' else 'user'}: "
        f"{str(item.get('content') or '')[:500]}"
        for item in (prior_messages or [])[-6:]
        if isinstance(item, dict)
    ]
    classifier_input = (
        f"Recent conversation:\n{chr(10).join(context_lines) or '(none)'}\n"
        f"Incoming user message:\n{message[:1200]}\n"
        f"Attached image count: {len(image_list)}"
    )
    system_prompt = (
        "Classify the incoming message using the recent conversation only to resolve "
        "references and follow-ups. Categories: NORMAL_CONVERSATION, STUDY_EXPLANATION, "
        "QUIZ_GENERATION, CHEAT_SHEET_GENERATION, RECALL_CARD_GENERATION, "
        "DOCUMENT_SUMMARY, IMAGE_UNDERSTANDING, OTHER_APPLICATION_ACTIONS. "
        "General knowledge questions are NORMAL_CONVERSATION unless the user asks to "
        "learn/study or use a study tool. Greetings do not override an explicit study "
        "request in the same message. Use IMAGE_UNDERSTANDING when an image is attached "
        "and the user asks about it. Return only JSON with exactly these fields: "
        '{"intent":"...","confidence":0.0,"study_action":false,'
        '"action_type":null,"topic":null}. study_action is true only for the five '
        "study intents from STUDY_EXPLANATION through DOCUMENT_SUMMARY. action_type "
        "must equal intent except NORMAL_CONVERSATION, where it is null. topic is a "
        "short extracted subject or null. Do not include reasoning."
    )

    errors: List[str] = []
    try:
        candidates = _classifier_candidates()
    except Exception as exc:
        candidates = []
        errors.append(f"model selection: {exc}")
        logger.exception("Intent classifier model selection failed.")
    for candidate in candidates[:2]:
        try:
            response = ai_manager.generate_text(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": classifier_input},
                ],
                provider=candidate["provider"],
                model=candidate["model"],
                base_url=candidate["base_url"],
                temperature=0,
                max_tokens=120,
                json_mode=True,
            )
            parsed = _validated_result(json.loads(str(response.get("text") or "")))
            explicit_intent = _explicit_study_intent(message)
            if explicit_intent and parsed["intent"] == "NORMAL_CONVERSATION":
                parsed.update(
                    {
                        "intent": explicit_intent,
                        "study_action": True,
                        "action_type": explicit_intent,
                        "topic": parsed.get("topic") or _fallback_topic(message),
                    }
                )
            if image_list and parsed["intent"] not in (
                "QUIZ_GENERATION",
                "CHEAT_SHEET_GENERATION",
                "RECALL_CARD_GENERATION",
            ):
                parsed.update(
                    {
                        "intent": "IMAGE_UNDERSTANDING",
                        "study_action": False,
                        "action_type": "IMAGE_UNDERSTANDING",
                    }
                )
            with _state_lock:
                _last_assignment = {
                    "provider": candidate["provider"],
                    "model": candidate["model"],
                    "selection_reason": candidate["selection_reason"],
                }
            elapsed_ms = (time.perf_counter() - started) * 1000
            _record_classification_latency(elapsed_ms)
            logger.info(
                "Intent classified: provider=%s model=%s reason=%s latency_ms=%.2f intent=%s",
                candidate["provider"],
                candidate["model"],
                candidate["selection_reason"],
                elapsed_ms,
                parsed["intent"],
            )
            return {
                **parsed,
                "classifier_provider": candidate["provider"],
                "classifier_model": candidate["model"],
                "selection_reason": candidate["selection_reason"],
                "latency_ms": elapsed_ms,
                "used_fallback": False,
            }
        except (ProviderRequestError, RuntimeError, ValueError, TypeError) as exc:
            errors.append(f"{candidate['provider']}/{candidate['model']}: {exc}")
            with _state_lock:
                _rejected_classifier_models.add(
                    (candidate["provider"], candidate["model"])
                )
            logger.warning(
                "Intent classifier model failed: provider=%s model=%s error=%s",
                candidate["provider"],
                candidate["model"],
                str(exc),
            )

    elapsed_ms = (time.perf_counter() - started) * 1000
    _record_classification_latency(elapsed_ms)
    fallback = _fallback_intent(message, image_list, prior_messages or [])
    if errors:
        logger.error("Intent classifier exhausted compatible models: %s", "; ".join(errors))
    logger.info(
        "Intent classified: provider=none model=none reason=safe_fallback "
        "latency_ms=%.2f intent=%s",
        elapsed_ms,
        fallback["intent"],
    )
    return {
        **fallback,
        "classifier_provider": None,
        "classifier_model": None,
        "selection_reason": "No compatible classifier model succeeded; used safe fallback.",
        "latency_ms": elapsed_ms,
        "used_fallback": True,
    }


def get_intent_classifier_status() -> Dict[str, Any]:
    """Return safe classifier assignment and aggregate latency metadata."""
    with _state_lock:
        count = _classification_count
        total_latency_ms = _classification_latency_ms
        assignment = dict(_last_assignment)
    return {
        **assignment,
        "classification_count": count,
        "average_latency_ms": total_latency_ms / count if count else None,
    }
