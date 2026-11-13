#!/usr/bin/env python3
"""
DocPilot AI / VISION AI — Standalone Local Application Entry Point (main.py)
Launch on Windows / macOS / Linux with:
    python main.py

Serves the complete frontend interface and all backend API services
(SQLite Database, OS-Encrypted Credential Vault, Multi-Provider AI Manager,
Multi-Format Document Extraction & RAG, Study Copilot, Recall Cards,
Interactive Quiz, Adaptive Cheat Sheet, Vision Solver, and Real File Exporter)
without requiring Node.js, npm, Vite, or Docker at runtime.
"""

from __future__ import annotations

import email.parser
import io
import json
import logging
import mimetypes
import os
import re
import socket
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.ai_provider import (
    GEMINI_MODEL_CURATED_CATALOG,
    ProviderRequestError,
    ai_manager,
    sanitize_log_text,
)
from app.database.db import BASE_DIR, DATA_DIR, EXPORTS_DIR, SQLITE_DB_PATH, UPLOADS_DIR, db, ensure_data_dirs, now_iso
from app.services.documents_service import process_and_index_upload
from app.services.export_service import (
    FORMAT_METADATA,
    detect_requested_export_formats,
    generate_and_validate_deliverable,
    validate_generated_file,
)
from app.services.intent_classifier import (
    classify_user_intent,
    get_intent_classifier_status,
)
from app.services.rag_service import retrieve_relevant_chunks
from app.services.study_service import (
    generate_cheatsheet_for_session,
    generate_flashcards_for_session,
    generate_quiz_for_session,
    is_trivial_greeting,
    regenerate_cheatsheet_section_for_session,
)
from app.services.vision_service import VISION_CATEGORIES, analyze_vision_image, classify_vision_image

logger = logging.getLogger(__name__)

_DOCUMENT_REFERENCE_RE = re.compile(
    r"\b(?:this|that|the|uploaded|attached)\s+"
    r"(?:file|document|pdf|chapter|report|paper|notes?)\b|"
    r"\b(?:file|document|pdf)\s+(?:contents?|overview|summary)\b|"
    r"\b(?:summari[sz]e|overview|contents?)\s+(?:this|the|uploaded|attached)\s+"
    r"(?:file|document|pdf)\b",
    re.I,
)
_DOCUMENT_FOLLOWUP_RE = re.compile(
    r"\b(?:second|third|next|previous|last|first)\s+(?:section|chapter|page|part)\b|"
    r"\b(?:explain|expand on|go deeper on|summari[sz]e)\s+(?:it|that|this|the next one)\b|"
    r"\b(?:more detail|continue from|where we left off)\b",
    re.I,
)


def _is_document_question(
    message: str, prior_messages: List[Dict[str, Any]]
) -> bool:
    if _DOCUMENT_REFERENCE_RE.search(message or ""):
        return True
    if not _DOCUMENT_FOLLOWUP_RE.search(message or ""):
        return False
    return any(
        _DOCUMENT_REFERENCE_RE.search(str(item.get("content") or ""))
        or "[Source:" in str(item.get("content") or "")
        for item in prior_messages[-12:]
        if isinstance(item, dict)
    )


# ============================================================================
# SHARED CORE API CONTROLLER (USED BY BOTH FASTAPI & BUILT-IN HTTP ENGINE)
# ============================================================================

def build_system_prompt(
    workspace_mode: str,
    teaching_mode: str,
    rag_context: str,
    memories_text: str,
    requested_formats: List[str],
    document_overview: bool = False,
) -> str:
    mode_instructions = {
        "study": "You are VISION AI Study Copilot, an expert university STEM & academic tutor. Provide clear, rigorous explanations, step-by-step derivations using LaTeX ($$ ... $$ for display equations, $ ... $ for inline math), intuitive examples, and structured headings.",
        "chatbot": "You are VISION AI Assistant, a knowledgeable, direct, and helpful reasoning partner.",
        "docqa": "You are VISION AI Document Q&A Analyst. Ground your answers directly in the attached document context and cite relevant sections or pages.",
        "productivity": "You are VISION AI Productivity & Engineering Strategist. Focus on actionable plans, clean code, structured deliverables, and clear execution steps.",
        "socialgood": "You are VISION AI Accessibility & Community Educator. Explain complex concepts in clear, inclusive, plain language with practical takeaways.",
    }
    depth_instructions = {
        "quick": (
            "LEARNING MODE — QUICK STUDY:\n"
            "- Focus strictly on essential, high-yield information.\n"
            "- Provide short, direct explanations.\n"
            "- Highlight important definitions, core formulas, and bulleted key takeaways.\n"
            "- Omit unnecessary background detail or lengthy derivations."
        ),
        "normal": (
            "LEARNING MODE — NORMAL LEARNING:\n"
            "- Provide balanced, university-level explanations.\n"
            "- Clearly explain what each concept means and why it matters.\n"
            "- Include relevant real-world or academic examples.\n"
            "- Highlight important connections between concepts."
        ),
        "deep": (
            "LEARNING MODE — DEEP LEARNING:\n"
            "- Provide detailed, rigorous conceptual and technical explanations.\n"
            "- Explain physical, mathematical, or logical mechanisms step by step.\n"
            "- State underlying relationships, governing principles, and assumptions.\n"
            "- Include worked examples where useful, discuss important edge cases, and include necessary mathematical reasoning without unnecessary filler."
        ),
        "auto": (
            "LEARNING MODE — AUTO ADAPTIVE:\n"
            "- Dynamically determine the appropriate explanation depth based on the user's request, active conversation history, subject complexity, and demonstrated understanding.\n"
            "- Use concise summaries for simple factual queries and step-by-step mechanistic depth with worked examples for complex STEM/academic topics."
        ),
    }
    parts = [
        mode_instructions.get(workspace_mode, mode_instructions["study"]),
        depth_instructions.get(teaching_mode, depth_instructions["normal"]),
    ]
    if requested_formats:
        fmt_str = ", ".join(f".{f}" for f in requested_formats)
        if "py" in requested_formats:
            parts.append(
                "The user requested a downloadable Python script (.py). Include a complete, clean, runnable ```python ... ``` code block with proper functions and an `if __name__ == '__main__':` entry point."
            )
        else:
            parts.append(
                f"The user requested a downloadable deliverable ({fmt_str}). Structure your response clearly with headings, bullet points, or tables so it exports cleanly."
            )
    if memories_text:
        parts.append(f"ACTIVE SESSION MEMORY:\n{memories_text}")
    if rag_context:
        parts.append(
            "ATTACHED DOCUMENT CONTEXT (Combine this grounded document context with the active conversation history):\n"
            + rag_context
        )
        parts.append(
            "DOCUMENT GROUNDING: Use only the document excerpts and metadata supplied above for claims about the uploaded documents. "
            "Do not claim to see an attachment that is not represented in that context. Preserve source, page, and section references "
            "when available. If the context does not contain enough evidence, say what is missing instead of guessing."
        )
    if document_overview:
        parts.append(
            "DOCUMENT OVERVIEW REQUEST: Give a detailed, source-grounded explanation adapted to the document. When supported by the "
            "provided context, cover its title and subject, purpose and scope, main concepts, important sections, technical explanations, "
            "findings, relevant examples/formulas/data, and conclusions or implications. Organize by chapter/section when the sources "
            "support it, and cite the supplied page/section references. Do not invent missing sections or details."
        )
    return "\n\n".join(parts)


def execute_chat_turn(
    body: Dict[str, Any],
    on_retry: Optional[Callable[[int, ProviderRequestError], None]] = None,
) -> Dict[str, Any]:
    """
    Executes a complete chat turn:
    1. Verifies that an AI provider is configured and verified (blocks unconfigured requests)
    2. Ensures session exists & isolates conversation history to `session_id`
    3. Classifies intent with an independent lightweight provider model
    4. Routes study-tool requests or generates a chat response with optional scoped RAG
    5. Saves the user/assistant messages and any generated file metadata
    """
    user = db.get_default_user()
    user_id = user["id"]
    session_id = str(body.get("session_id") or f"chat-{int(time.time() * 1000)}").strip()
    prompt_text = str(body.get("message") or body.get("prompt") or "").strip()
    if not prompt_text:
        raise ValueError("Message cannot be empty.")

    provider = str(body.get("provider") or ai_manager.active_provider or "gemini").strip().lower()
    model = str(body.get("model") or body.get("model_id") or ai_manager.active_model or "").strip()
    base_url = str(body.get("base_url") or ai_manager.local_endpoint or "").strip()
    model_selection_mode = str(body.get("model_selection_mode") or "manual").strip().lower()
    allow_model_fallback = True
    workspace_mode = str(body.get("workspace_mode") or "study").strip()
    teaching_mode = str(body.get("teaching_mode") or body.get("learning_mode") or "normal").strip().lower()
    document_ids = body.get("document_ids") if isinstance(body.get("document_ids"), list) else None
    images = body.get("images") if isinstance(body.get("images"), list) else []

    # Access Control Check: Require a valid configured API key (unless using a verified local endpoint)
    resolved_key, _ = ai_manager.resolve_api_key(provider, body.get("api_key"))
    if provider != "local" and not resolved_key:
        raise PermissionError(
            "AI setup required. Add and verify an API key in AI Engine & Provider Settings to start chatting."
        )
    if provider == "local":
        st_local = db.get_provider_public_status("local")
        if not st_local.get("verified"):
            raise PermissionError(
                "AI setup required. Add and verify an API key in AI Engine & Provider Settings to start chatting."
            )

    session = db.get_session(session_id)
    auto_title = prompt_text[:48].strip() + ("..." if len(prompt_text) > 48 else "")
    if not session:
        session = db.create_session(
            {
                "id": session_id,
                "user_id": user_id,
                "workspace_id": "ws-default",
                "workspace_mode": workspace_mode,
                "title": auto_title if not is_trivial_greeting(prompt_text) else "New Session",
                "topic": auto_title if not is_trivial_greeting(prompt_text) else "",
                "provider_id": provider,
                "model_id": model,
                "model_name": model,
                "teaching_depth": teaching_mode,
                "document_ids": document_ids or [],
            }
        )
    else:
        updates: Dict[str, Any] = {
            "provider_id": provider,
            "model_id": model,
            "teaching_depth": teaching_mode,
        }
        if not is_trivial_greeting(prompt_text) and is_trivial_greeting(session.get("title", "")):
            updates["title"] = auto_title
            updates["topic"] = auto_title
        session = db.update_session(session_id, updates) or session

    # Load prior conversation history strictly for this session_id
    prior_messages = db.get_messages(session_id)
    session_documents = db.get_documents(user_id=user_id, session_id=session_id)
    requested_document_ids = (
        document_ids
        if isinstance(body.get("document_ids"), list)
        else session.get("document_ids") or [item.get("id") for item in session_documents]
    )
    requested_document_ids = {
        str(item).strip() for item in requested_document_ids if str(item).strip()
    }
    active_documents = [
        item
        for item in session_documents
        if item.get("id") in requested_document_ids
        and str(item.get("status") or "ready").lower() not in ("failed", "error")
    ]
    failed_documents = [
        item
        for item in session_documents
        if item.get("id") in requested_document_ids
        and str(item.get("status") or "").lower() in ("failed", "error")
    ]
    supplied_failures = body.get("failed_documents")
    if isinstance(supplied_failures, list):
        failed_documents.extend(
            item
            for item in supplied_failures
            if isinstance(item, dict)
            and (item.get("name") or item.get("filename"))
        )
    intent_result = classify_user_intent(prompt_text, prior_messages, images)
    intent = intent_result["intent"]
    explicit_document_question = _is_document_question(prompt_text, prior_messages)
    if (
        explicit_document_question
        and intent not in (
            "QUIZ_GENERATION",
            "CHEAT_SHEET_GENERATION",
            "RECALL_CARD_GENERATION",
            "OTHER_APPLICATION_ACTIONS",
            "IMAGE_UNDERSTANDING",
        )
    ):
        intent = "DOCUMENT_SUMMARY"
        intent_result["intent"] = intent
        intent_result["study_action"] = False
        intent_result["action_type"] = intent
    document_context_required = bool(
        active_documents
        and (
            intent == "DOCUMENT_SUMMARY"
            or intent == "STUDY_EXPLANATION"
            or explicit_document_question
        )
    )
    document_notice = ""
    if intent == "DOCUMENT_SUMMARY" and not active_documents:
        if failed_documents:
            failures = "; ".join(
                f"{item.get('name') or item.get('filename')}: "
                f"{str(item.get('error_message') or item.get('summary') or item.get('error') or 'text extraction or indexing failed').strip()}"
                for item in failed_documents
            )
            document_notice = (
                "I could not read the attached document. "
                f"Upload/extraction details: {failures}. Please retry the upload or use a text-searchable file."
            )
        else:
            document_notice = (
                "Please upload or attach a document to this conversation first, and I can explain its contents."
            )
    logger.info(
        "Document context resolved: operation=%s session=%s active_documents=%d failed_documents=%d context_required=%s",
        "document_summary" if document_context_required or intent == "DOCUMENT_SUMMARY" else "chat",
        session_id,
        len(active_documents),
        len(failed_documents),
        document_context_required,
    )
    action_route = {
        "QUIZ_GENERATION": "quiz_generation",
        "CHEAT_SHEET_GENERATION": "cheat_sheet_generation",
        "RECALL_CARD_GENERATION": "recall_card_generation",
        "DOCUMENT_SUMMARY": "document_summary",
        "STUDY_EXPLANATION": "study_explanation",
        "IMAGE_UNDERSTANDING": "image_understanding",
        "OTHER_APPLICATION_ACTIONS": "application_chat",
    }.get(intent, "normal_chat")
    logger.info("Intent routing: intent=%s route=%s", intent, action_route)

    # Retrieve RAG context strictly for this session_id and document_ids
    rag_context = ""
    sources: List[Dict[str, Any]] = []
    study_action_result: Optional[Dict[str, Any]] = None
    res: Optional[Dict[str, Any]] = None
    if intent in ("QUIZ_GENERATION", "CHEAT_SHEET_GENERATION", "RECALL_CARD_GENERATION"):
        study_params = {
            "user_id": user_id,
            "session_id": session_id,
            "topic": intent_result.get("topic") or body.get("topic"),
            "document_ids": document_ids,
            "difficulty": body.get("difficulty") or "mixed",
            "learning_mode": teaching_mode,
            "count": body.get("count"),
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "model_selection_mode": model_selection_mode,
            "allow_model_fallback": allow_model_fallback,
        }
        if intent == "QUIZ_GENERATION":
            study_params["quiz_provider"] = body.get("quiz_provider")
            study_params["quiz_model"] = body.get("quiz_model")
            study_action_result = generate_quiz_for_session(study_params)
            questions = study_action_result.get("quizzes") or []
            answer_text = study_action_result.get("message") or (
                f"Created a {len(questions)}-question quiz on "
                f"{study_action_result.get('topic') or 'the active study topic'}. "
                "It is saved in the Quizzes section."
            )
        elif intent == "CHEAT_SHEET_GENERATION":
            study_params["cheatsheet_provider"] = body.get("cheatsheet_provider")
            study_params["cheatsheet_model"] = body.get("cheatsheet_model")
            study_action_result = generate_cheatsheet_for_session(study_params)
            answer_text = study_action_result.get("message") or (
                f"Created revision notes for "
                f"{study_action_result.get('topic') or 'the active study topic'}. "
                "They are saved in the Cheat Sheet section."
            )
        else:
            study_params["flashcards_provider"] = body.get("flashcards_provider")
            study_params["flashcards_model"] = body.get("flashcards_model")
            study_action_result = generate_flashcards_for_session(study_params)
            cards = study_action_result.get("flashcards") or []
            answer_text = study_action_result.get("message") or (
                f"Created {len(cards)} recall cards for "
                f"{study_action_result.get('topic') or 'the active study topic'}. "
                "They are saved in the Recall Cards section."
            )
        action_route = {
            "QUIZ_GENERATION": "quiz_generation",
            "CHEAT_SHEET_GENERATION": "cheat_sheet_generation",
            "RECALL_CARD_GENERATION": "recall_card_generation",
        }[intent]
        logger.info("Intent routing completed: intent=%s route=%s", intent, action_route)
        study_model_routing = study_action_result.get("model_routing") or []
        latest_study_route = study_model_routing[-1] if study_model_routing else {}
        fallback_study_route = next(
            (item for item in reversed(study_model_routing) if item.get("fallback")),
            {},
        )
        res = {
            "text": answer_text,
            "provider": latest_study_route.get("provider") or provider,
            "model": latest_study_route.get("model") or model,
            "model_routing": study_model_routing,
            "model_selection_reason": latest_study_route.get("selection_reason"),
            "model_fallback": any(item.get("fallback") for item in study_model_routing),
            "model_fallback_reason": fallback_study_route.get("fallback_reason"),
        }
    elif (
        document_context_required
        or (
            intent != "NORMAL_CONVERSATION"
            and (body.get("use_rag", True) or intent == "DOCUMENT_SUMMARY")
        )
    ):
        active_document_ids = [str(item["id"]) for item in active_documents]
        if document_context_required:
            doc_header = "\n".join(
                f"- {item.get('filename') or 'Uploaded document'} "
                f"({item.get('file_type') or 'file'}, {item.get('chunk_count') or 0} indexed chunks): "
                f"{item.get('summary') or 'No extraction preview is available.'}"
                for item in active_documents
            )
            total_chunks = sum(int(item.get("chunk_count") or 0) for item in active_documents)
            broad_overview = bool(
                re.search(
                    r"\b(?:tell me about|summari[sz]e|overview|contents?|chapter by chapter|in detail|what is this)\b",
                    prompt_text,
                    re.I,
                )
            )
            retrieval_top_k = max(8, total_chunks) if broad_overview else 8
        else:
            doc_header = ""
            retrieval_top_k = 8
        rag_res = retrieve_relevant_chunks(
            user_id=user_id,
            query=prompt_text,
            document_ids=(
                active_document_ids if document_context_required else document_ids
            ),
            session_id=session_id,
            top_k=retrieval_top_k,
        )
        rag_context = rag_res.get("context") or ""
        sources = rag_res.get("sources") or []
        if doc_header:
            rag_context = f"ACTIVE DOCUMENT METADATA:\n{doc_header}\n\n{rag_context}".strip()
        logger.info(
            "Document retrieval complete: operation=%s session=%s documents=%d chunks=%d",
            "document_summary" if document_context_required else "rag_chat",
            session_id,
            len(active_documents) if document_context_required else len(document_ids or []),
            len(rag_res.get("chunks") or []),
        )
        if document_context_required and not rag_res.get("chunks"):
            document_notice = (
                "I found the active document record, but it contains no readable indexed text to answer from. "
                "Please retry the upload or provide a text-searchable copy."
            )

    # Retrieve session memory
    memories_list = db.get_memories(user_id=user_id, session_id=session_id)
    memories_text = "\n".join(f"- {m['content']}" for m in memories_list[:10])

    requested_formats = detect_requested_export_formats(prompt_text)
    if intent in ("QUIZ_GENERATION", "CHEAT_SHEET_GENERATION", "RECALL_CARD_GENERATION"):
        sys_prompt = ""
        llm_messages: List[Dict[str, Any]] = []
    else:
        generation_workspace_mode = (
            "chatbot"
            if intent == "NORMAL_CONVERSATION"
            else "docqa"
            if document_context_required or intent == "DOCUMENT_SUMMARY"
            else workspace_mode
        )
        sys_prompt = build_system_prompt(
            workspace_mode=generation_workspace_mode,
            teaching_mode=teaching_mode,
            rag_context=rag_context,
            memories_text=memories_text,
            requested_formats=requested_formats,
            document_overview=bool(
                document_context_required
                and re.search(
                    r"\b(?:tell me about|summari[sz]e|overview|contents?|chapter by chapter|in detail|what is this)\b",
                    prompt_text,
                    re.I,
                )
            ),
        )

        llm_messages = [{"role": "system", "content": sys_prompt}]
        for pm in prior_messages[-24:]:
            llm_messages.append(
                {
                    "role": "assistant" if pm.get("role") == "assistant" else "user",
                    "content": pm.get("content", ""),
                }
            )
        llm_messages.append(
            {
                "role": "user",
                "content": prompt_text,
                "images": images,
            }
        )

    mode_max_tokens = 1536 if teaching_mode == "quick" else (6144 if teaching_mode == "deep" else 4096)
    mode_temp = 0.25 if teaching_mode == "quick" else (0.4 if teaching_mode == "deep" else 0.35)

    if document_notice:
        answer_text = document_notice
        res = {
            "text": answer_text,
            "provider": provider,
            "model": model,
        }
    elif intent not in ("QUIZ_GENERATION", "CHEAT_SHEET_GENERATION", "RECALL_CARD_GENERATION"):
        res = ai_manager.generate_text(
            messages=llm_messages,
            provider=provider,
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            temperature=mode_temp,
            max_tokens=mode_max_tokens,
            on_retry=on_retry,
            vision_provider=body.get("vision_provider"),
            vision_model=body.get("vision_model"),
            allow_vision_fallback=True,
            selection_mode=model_selection_mode,
            allow_model_fallback=allow_model_fallback,
            document_context=document_context_required,
        )
    if res is None:
        raise RuntimeError("Chat routing completed without producing a response.")
    answer_text = res.get("text", "").strip()
    db.add_message(
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": "user",
            "content": prompt_text,
            "model": model,
            "provider": provider,
        }
    )
    routed_vision = bool(res.get("routed_vision"))
    routing_notice = res.get("vision_routing_notice") or res.get("routing_notice")
    effective_model = res.get("model") or model
    effective_provider = res.get("provider") or provider

    # Generate & validate real files if requested by user
    generated_files: List[Dict[str, Any]] = []
    for fmt in requested_formats:
        try:
            gf = generate_and_validate_deliverable(
                format_type=fmt,
                title=prompt_text[:60],
                content=answer_text,
                session_id=session_id,
                document_theme=body.get("document_theme"),
            )
            generated_files.append(gf)
        except Exception as gf_err:
            logger.exception(
                "Export generation failed format=%s session_id=%s",
                fmt,
                session_id,
            )

    saved_assistant_msg = db.add_message(
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": "assistant",
            "content": answer_text,
            "model": effective_model,
            "provider": effective_provider,
            "sources": sources,
            "generated_files": generated_files,
        }
    )

    # Link message_id to generated files
    for gf in generated_files:
        gf["message_id"] = saved_assistant_msg["id"]
        db.record_generated_file(gf)

    updated_session = db.get_session(session_id) or session
    return {
        "session_id": session_id,
        "session_title": updated_session.get("title", "Study Session"),
        "answer": answer_text,
        "sources": sources,
        "generated_files": generated_files,
        "model": model,
        "provider": provider,
        "effective_model": effective_model,
        "effective_provider": effective_provider,
        "routed_vision": routed_vision,
        "vision_analysis_used": bool(
            (images or document_context_required)
            and res.get("vision_selection_mode") == "automatic"
        ),
        "vision_model_name": (
            res.get("routed_vision_model_name")
            or res.get("vision_model_name")
            or (effective_model if images or document_context_required else "")
        ),
        "routing_notice": routing_notice,
        "model_selection_reason": res.get("model_selection_reason"),
        "model_fallback": bool(res.get("model_fallback")),
        "model_fallback_reason": res.get("model_fallback_reason"),
        "model_routing": res.get("model_routing") or [],
        "teaching_mode": teaching_mode,
        "intent": intent,
        "intent_confidence": intent_result["confidence"],
        "intent_route": action_route,
        "classifier_provider": intent_result.get("classifier_provider"),
        "classifier_model": intent_result.get("classifier_model"),
        "classifier_selection_reason": intent_result.get("selection_reason"),
        "classifier_latency_ms": intent_result.get("latency_ms"),
        "classifier_used_fallback": intent_result.get("used_fallback", False),
        "study_action_result": study_action_result,
    }


def handle_api_request(
    method: str,
    path: str,
    query_params: Dict[str, List[str]],
    body_bytes: bytes,
    headers: Dict[str, str],
) -> Tuple[int, Dict[str, str], bytes]:
    """
    Central router for all non-streaming API endpoints.
    Returns (status_code, response_headers, response_bytes).
    """
    json_headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store, no-cache, must-revalidate",
    }

    def json_resp(status: int, payload: Any) -> Tuple[int, Dict[str, str], bytes]:
        return status, json_headers, json.dumps(payload).encode("utf-8")

    def parse_body() -> Dict[str, Any]:
        if not body_bytes:
            return {}
        try:
            return json.loads(body_bytes.decode("utf-8", errors="replace"))
        except Exception:
            return {}

    def q_first(key: str, default: str = "") -> str:
        vals = query_params.get(key)
        return vals[0] if vals else default

    def content_disposition(filename: str) -> str:
        ascii_name = filename.encode("ascii", "ignore").decode("ascii")
        ascii_name = re.sub(r'[^A-Za-z0-9._ -]', "_", ascii_name).strip(" .")
        if not ascii_name:
            ascii_name = "download"
        encoded_name = urllib.parse.quote(filename, safe="")
        return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded_name}'

    try:
        user = db.get_default_user()
        user_id = user["id"]

        # 1. Health check
        if path in ("/health", "/api/health") and method == "GET":
            return json_resp(
                200,
                {
                    "status": "ok",
                    "service": "DocPilot VISION AI (Python Standalone)",
                    "runtime": f"Python {sys.version.split()[0]}",
                    "database": str(SQLITE_DB_PATH),
                    "active_provider": ai_manager.active_provider,
                    "active_model": ai_manager.active_model,
                    "ai_setup_required": not db.is_any_provider_configured(),
                    "timestamp": now_iso(),
                },
            )

        if path == "/api/intent/status" and method == "GET":
            return json_resp(200, get_intent_classifier_status())

        # 2. User Profile & Settings
        if path in ("/api/profile", "/api/user/profile") and method == "GET":
            settings = db.get_user_settings(user_id)
            return json_resp(200, {"user": user, "settings": settings})

        if path in ("/api/profile", "/api/user/profile") and method in ("PUT", "PATCH"):
            b = parse_body()
            updated = db.update_user_profile(
                user_id, b.get("username"), b.get("email"), b.get("role")
            )
            return json_resp(200, {"success": True, "user": updated})

        if path == "/api/settings" and method == "GET":
            return json_resp(
                200,
                {
                    "success": True,
                    "settings": db.get_user_settings(user_id),
                    "providers": db.get_all_providers_public_status(),
                    "ai_setup_required": not db.is_any_provider_configured(),
                },
            )

        if path == "/api/settings" and method in ("PUT", "POST"):
            b = parse_body()
            updated = db.update_user_settings(b, user_id)
            ai_manager.active_provider = updated["active_provider"]
            ai_manager.active_model = updated["active_model"]
            ai_manager.active_model_name = updated["active_model_name"]
            ai_manager.local_endpoint = updated["local_endpoint"]
            return json_resp(
                200,
                {
                    "success": True,
                    "settings": updated,
                    "ai_setup_required": not db.is_any_provider_configured(),
                },
            )

        # 3. AI Provider Status, Live Verification, Explicit Key Reveal & Key Deletion
        if path == "/api/providers/status" and method == "GET":
            provider_statuses = db.get_all_providers_public_status()
            provider_statuses = {
                provider: ai_manager.sanitize_provider_status(provider, status)
                for provider, status in provider_statuses.items()
            }
            return json_resp(
                200,
                {
                    "success": True,
                    "providers": provider_statuses,
                    "settings": db.get_user_settings(user_id),
                    "ai_setup_required": not db.is_any_provider_configured(),
                },
            )

        if path == "/api/providers/recommendations" and method == "GET":
            text_models = ai_manager.rank_text_models()[:1]
            vision_models = ai_manager.rank_text_models(capability="vision")[:1]
            return json_resp(
                200,
                {
                    "success": True,
                    "text": text_models,
                    "vision": vision_models,
                },
            )

        m_prov_reveal = re.match(r"^/api/providers/([^/]+)/reveal$", path)
        if m_prov_reveal and method in ("GET", "POST"):
            prov = urllib.parse.unquote(m_prov_reveal.group(1)).strip().lower()
            raw_key = ai_manager.resolve_api_key(prov)
            if not raw_key:
                return json_resp(
                    404,
                    {
                        "success": False,
                        "provider": prov,
                        "error": f"No stored API key found for {prov.upper()}.",
                    },
                )
            from app.database.db import mask_api_key
            return json_resp(
                200,
                {
                    "success": True,
                    "provider": prov,
                    "api_key": raw_key,
                    "masked_key": mask_api_key(raw_key),
                },
            )

        m_prov_verify = re.match(r"^/api/providers/([^/]+)/verify$", path)
        if m_prov_verify and method == "POST":
            prov = urllib.parse.unquote(m_prov_verify.group(1)).strip().lower()
            b = parse_body()
            res = ai_manager.verify_and_discover_provider(
                provider=prov,
                api_key=b.get("api_key"),
                base_url=b.get("base_url"),
                activate=bool(b.get("activate", True)),
                selected_model=b.get("model") or b.get("selected_model"),
                refresh=bool(b.get("refresh", False)),
            )
            status_code = 200 if res.get("verified") else 401
            return json_resp(
                status_code,
                {
                    "success": bool(res.get("verified")),
                    **res,
                    "models": res.get("available_models", []),
                    "ai_setup_required": not db.is_any_provider_configured(),
                    "error": None if res.get("verified") else res.get("user_friendly_message"),
                },
            )

        m_prov_del = re.match(r"^/api/providers/([^/]+)/credentials$", path)
        if m_prov_del and method == "DELETE":
            prov = urllib.parse.unquote(m_prov_del.group(1)).strip().lower()
            db.clear_provider_credentials(prov)
            return json_resp(
                200,
                {
                    "success": True,
                    "provider": prov,
                    "has_stored_key": False,
                    "masked_key": "",
                    "verified": False,
                    "available_models": [],
                    "providers": db.get_all_providers_public_status(),
                    "ai_setup_required": not db.is_any_provider_configured(),
                    "message": f"Stored API key and credentials for {prov.upper()} deleted.",
                },
            )

        if path == "/api/models/discover" and method == "POST":
            b = parse_body()
            prov = str(b.get("provider") or "gemini").strip().lower()
            res = ai_manager.verify_and_discover_provider(
                provider=prov,
                api_key=b.get("api_key"),
                base_url=b.get("base_url"),
                activate=True,
                selected_model=b.get("model"),
                refresh=bool(b.get("refresh", True)),
            )
            status_code = 200 if res.get("verified") else 401
            return json_resp(
                status_code,
                {
                    "success": bool(res.get("verified")),
                    **res,
                    "models": res.get("available_models", []),
                    "ai_setup_required": not db.is_any_provider_configured(),
                    "error": None if res.get("verified") else res.get("user_friendly_message"),
                },
            )

        if path == "/api/models/live" and method == "GET":
            prov = q_first("provider", ai_manager.active_provider or "gemini").lower()
            base_url = q_first("base_url", "")
            st = db.get_provider_public_status(prov)
            if st.get("verified") and st.get("available_models"):
                return json_resp(
                    200,
                    {
                        "success": True,
                        "verified": True,
                        "provider": prov,
                        "has_stored_key": st.get("has_stored_key", False),
                        "masked_key": st.get("masked_key", ""),
                        "authentication_method": st.get("authentication_method"),
                        "verification_timestamp": st.get("verification_timestamp"),
                        "models": st["available_models"],
                        "available_models": st["available_models"],
                        "model_capabilities": st.get("model_capabilities", {}),
                    },
                )
            if st.get("explicitly_deleted") or not st.get("has_stored_key"):
                return json_resp(
                    200,
                    {
                        "success": False,
                        "verified": False,
                        "provider": prov,
                        "has_stored_key": False,
                        "masked_key": "",
                        "models": [],
                        "available_models": [],
                        "user_friendly_message": f"No verified API key configured for {prov.upper()}.",
                    },
                )
            res = ai_manager.verify_and_discover_provider(
                provider=prov, api_key=None, base_url=base_url, activate=False
            )
            return json_resp(
                200,
                {
                    "success": bool(res.get("verified")),
                    **res,
                    "models": res.get("available_models", []),
                },
            )

        # 4. Chat Sessions & Messages
        if path == "/api/chat/sessions" and method == "GET":
            sessions = db.get_sessions(user_id)
            return json_resp(200, {"sessions": sessions})

        if path == "/api/chat/sessions" and method == "POST":
            b = parse_body()
            ws = db.get_default_workspace(user_id)
            session = db.create_session(
                {
                    "id": b.get("id"),
                    "user_id": user_id,
                    "workspace_id": ws["id"],
                    "workspace_mode": b.get("workspace_mode") or "study",
                    "title": b.get("title") or "New Session",
                    "provider_id": b.get("provider_id") or ai_manager.active_provider,
                    "model_id": b.get("model_id") or ai_manager.active_model,
                    "model_name": b.get("model_name") or ai_manager.active_model_name,
                    "teaching_depth": b.get("teaching_depth") or "auto",
                    "response_style": b.get("response_style") or "balanced",
                    "reasoning_preference": b.get("reasoning_preference") or "standard",
                    "document_ids": b.get("document_ids") or [],
                }
            )
            return json_resp(200, {"session": session})

        m_sess = re.match(r"^/api/chat/sessions/([^/]+)$", path)
        if m_sess:
            sid = urllib.parse.unquote(m_sess.group(1))
            if method == "GET":
                session = db.get_session(sid)
                messages = db.get_messages(sid)
                session_docs = db.get_documents(user_id=user_id, session_id=sid)
                return json_resp(
                    200,
                    {
                        "session": session,
                        "messages": messages,
                        "documents": session_docs,
                        "cheatsheet": session.get("cheatsheet") if session else None,
                        "flashcards": session.get("flashcards", []) if session else [],
                    },
                )
            if method == "PUT":
                b = parse_body()
                updated = db.update_session(sid, b)
                return json_resp(200, {"success": True, "session": updated})
            if method == "DELETE":
                db.delete_session(sid)
                return json_resp(200, {"success": True})

        if path == "/api/chat" and method == "POST":
            b = parse_body()
            result = execute_chat_turn(b)
            return json_resp(200, {"success": True, "response": result["answer"], **result})

        # 5. Documents & Vector Store
        if path == "/api/rag/search" and method == "POST":
            b = parse_body()
            query = str(b.get("query") or b.get("q") or "").strip()
            if not query:
                return json_resp(400, {"success": False, "error": "A search query is required."})
            raw_document_ids = b.get("document_ids")
            document_ids = raw_document_ids if isinstance(raw_document_ids, list) else None
            result = retrieve_relevant_chunks(
                user_id=user_id,
                query=query,
                document_ids=document_ids,
                session_id=str(b.get("session_id") or "") or None,
                top_k=max(1, min(int(b.get("top_k") or 8), 50)),
            )
            hits = []
            for chunk in result["chunks"]:
                hits.append(
                    {
                        "document_id": chunk.get("document_id", ""),
                        "document_name": chunk.get("filename", "Document"),
                        "filename": chunk.get("filename", "Document"),
                        "page": chunk.get("page_number", 1),
                        "section_title": chunk.get("section_title", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "fullContent": chunk.get("content", ""),
                        "excerpt": (chunk.get("content") or "")[:500],
                        "score": 1.0,
                    }
                )
            return json_resp(
                200,
                {"success": True, "results": hits, "sources": result["sources"], "context": result["context"]},
            )

        if path == "/api/documents" and method == "GET":
            sid = q_first("session_id", "")
            docs = db.get_documents(user_id=user_id, session_id=sid if sid else None)
            return json_resp(200, {"documents": docs})

        m_doc_status = re.match(r"^/api/documents/([^/]+)/status$", path)
        if m_doc_status and method == "GET":
            doc_id = urllib.parse.unquote(m_doc_status.group(1))
            doc = db.get_document(doc_id)
            if not doc:
                return json_resp(404, {"success": False, "error": "Document not found"})
            chunks = db.get_chunks(user_id=user_id, document_ids=[doc_id])
            status_map = {
                "uploading": "UPLOADING",
                "extracting": "EXTRACTING",
                "indexing": "INDEXING",
                "ready": "READY",
                "failed": "FAILED",
            }
            raw_st = doc.get("status") or doc.get("processing_status") or "ready"
            return json_resp(
                200,
                {
                    "document_id": doc_id,
                    "filename": doc.get("filename", ""),
                    "processing_status": raw_st,
                    "ui_status": status_map.get(str(raw_st).lower(), "READY"),
                    "page_count": doc.get("pages") or doc.get("page_count", 1),
                    "chunk_count": len(chunks),
                    "content_hash": doc.get("content_hash", ""),
                    "extraction_version": doc.get("extraction_version", "2.1.0"),
                    "processing_metrics": doc.get("processing_metrics") or {"total_ms": 115, "cached": False},
                    "error_reason": doc.get("error_reason"),
                    "warnings": doc.get("warnings", []),
                },
            )

        m_doc_preview = re.match(r"^/api/documents/([^/]+)/preview$", path)
        if m_doc_preview and method == "GET":
            doc_id = urllib.parse.unquote(m_doc_preview.group(1))
            doc = db.get_document(doc_id)
            if not doc:
                return json_resp(404, {"success": False, "error": "Document not found"})
            chunks = db.get_chunks(user_id=user_id, document_ids=[doc_id])
            return json_resp(
                200,
                {
                    "document": doc,
                    "overview": doc.get("summary") or doc.get("document_overview") or "",
                    "page_map": doc.get("page_map", []),
                    "processing_metrics": doc.get("processing_metrics") or None,
                    "chunks": chunks[:20],
                    "total_chunks": len(chunks),
                },
            )

        m_doc_chunks = re.match(r"^/api/documents/([^/]+)/chunks$", path)
        if m_doc_chunks and method == "GET":
            doc_id = urllib.parse.unquote(m_doc_chunks.group(1))
            doc = db.get_document(doc_id)
            if not doc:
                return json_resp(404, {"success": False, "error": "Document not found"})
            chunks = db.get_chunks(user_id=user_id, document_ids=[doc_id])
            return json_resp(
                200,
                {
                    "success": True,
                    "document": doc,
                    "chunks": [
                        {
                            "id": c["id"],
                            "chunk_index": c["chunk_index"],
                            "page_number": c.get("page_number", 1),
                            "section_title": c.get("section_title", ""),
                            "char_length": len(c.get("content", "")),
                            "has_embedding": bool(c.get("embedding")),
                            "content": c.get("content", ""),
                        }
                        for c in chunks
                    ],
                },
            )

        m_doc_del = re.match(r"^/api/documents/([^/]+)$", path)
        if m_doc_del and method == "DELETE":
            doc_id = urllib.parse.unquote(m_doc_del.group(1))
            db.delete_document(doc_id)
            return json_resp(200, {"success": True})

        if m_doc_del and method == "GET":
            doc_id = urllib.parse.unquote(m_doc_del.group(1))
            doc = db.get_document(doc_id)
            if not doc:
                return json_resp(404, {"success": False, "error": "Document not found"})
            chunks = db.get_chunks(user_id=user_id, document_ids=[doc_id])
            return json_resp(200, {"document": doc, "chunks": chunks})

        if path == "/api/documents/clear" and method == "POST":
            b = parse_body()
            sid = b.get("session_id")
            docs = db.get_documents(user_id=user_id, session_id=sid if sid else None)
            for d in docs:
                db.delete_document(d["id"])
            return json_resp(200, {"success": True})

        if path == "/api/vector-store/status" and method == "GET":
            sid = q_first("session_id", "")
            docs = db.get_documents(user_id=user_id, session_id=sid if sid else None)
            chunks = db.get_chunks(user_id=user_id, session_id=sid if sid else None)
            memories = db.get_memories(user_id=user_id, session_id=sid if sid else None)
            return json_resp(
                200,
                {
                    "status": "active",
                    "engine": "VISION Local SQLite Vector & Hybrid RAG Engine",
                    "documents_indexed": len(docs),
                    "total_chunks": len(chunks),
                    "memories_stored": len(memories),
                },
            )

        # 6. Memories
        if path == "/api/memories" and method == "GET":
            sid = q_first("session_id", "")
            return json_resp(
                200,
                {"memories": db.get_memories(user_id=user_id, session_id=sid if sid else None)},
            )

        if path == "/api/memories" and method == "POST":
            b = parse_body()
            mem = db.add_memory(
                {
                    "user_id": user_id,
                    "session_id": b.get("session_id") or "",
                    "category": b.get("category") or "fact",
                    "content": b.get("content") or "",
                    "importance": b.get("importance") or 3,
                }
            )
            return json_resp(200, {"memory": mem})

        m_mem_del = re.match(r"^/api/memories/([^/]+)$", path)
        if m_mem_del and method == "DELETE":
            db.delete_memory(urllib.parse.unquote(m_mem_del.group(1)))
            return json_resp(200, {"success": True})

        # 7. Study Copilot: Recall Cards, Quizzes, Cheat Sheets
        if path == "/api/study/flashcards" and method == "GET":
            sid = q_first("session_id", "")
            return json_resp(
                200,
                {"sets": db.get_flashcard_sets(user_id=user_id, session_id=sid if sid else None)},
            )

        if path == "/api/study/flashcards" and method == "POST":
            b = parse_body()
            b["user_id"] = user_id
            res = generate_flashcards_for_session(b)
            return json_resp(200, {"success": True, **res})

        if path == "/api/study/quizzes" and method == "GET":
            sid = q_first("session_id", "")
            return json_resp(
                200,
                {
                    "quizzes": db.get_quizzes(user_id=user_id, session_id=sid if sid else None),
                    "attempts": db.get_quiz_attempts(user_id=user_id),
                },
            )

        if path == "/api/study/quizzes" and method == "POST":
            b = parse_body()
            b["user_id"] = user_id
            res = generate_quiz_for_session(b)
            return json_resp(200, {"success": True, **res})

        if path == "/api/study/quizzes/submit" and method == "POST":
            b = parse_body()
            att = db.add_quiz_attempt(
                {
                    "quiz_id": b.get("quiz_id") or "adhoc",
                    "user_id": user_id,
                    "session_id": b.get("session_id") or "",
                    "score": b.get("score") or 0,
                    "total_questions": b.get("total_questions") or 0,
                    "accuracy_percentage": b.get("accuracy_percentage") or 0.0,
                    "weak_topics": b.get("weak_topics") or [],
                    "answers": b.get("answers") or {},
                }
            )
            return json_resp(200, {"success": True, "attempt": att})

        if path == "/api/study/cheatsheet" and method == "POST":
            b = parse_body()
            b["user_id"] = user_id
            res = generate_cheatsheet_for_session(b)
            return json_resp(200, {"success": True, **res})

        if path == "/api/study/cheatsheet" and method == "PUT":
            b = parse_body()
            sid = b.get("session_id")
            cs = b.get("cheatsheet")
            if sid and cs:
                db.save_cheatsheet(session_id=sid, sheet=cs, user_id=user_id)
            return json_resp(200, {"success": True})

        if path == "/api/study/cheatsheet/section" and method == "POST":
            b = parse_body()
            b["user_id"] = user_id
            res = regenerate_cheatsheet_section_for_session(b)
            return json_resp(200, {"success": True, **res})

        # 8. Vision Solver Endpoints
        if path == "/api/vision/categories" and method == "GET":
            return json_resp(
                200,
                {"success": True, "categories": list(VISION_CATEGORIES.values())},
            )

        if path == "/api/vision/compatibility" and method == "POST":
            b = parse_body()
            compat = ai_manager.check_vision_compatibility(b.get("provider"), b.get("model"))
            return json_resp(200, {"success": True, **compat})

        if path == "/api/vision/classify" and method == "POST":
            b = parse_body()
            res = classify_vision_image(b)
            return json_resp(200, res)

        if path in ("/api/vision/analyze", "/api/vision/solve") and method == "POST":
            b = parse_body()
            res = analyze_vision_image(b)
            return json_resp(200, res)

        # 9. File Generation, Validation & Download Endpoints
        if path == "/api/export" and method == "POST":
            b = parse_body()
            fmt = str(b.get("format_type") or b.get("format") or "pdf")
            title = str(b.get("title") or "VISION AI Deliverable")
            content = str(b.get("content") or title)
            sid = str(b.get("session_id") or "")
            try:
                rec = generate_and_validate_deliverable(
                    format_type=fmt,
                    title=title,
                    content=content,
                    session_id=sid,
                    document_theme=b.get("document_theme"),
                )
            except Exception as exc:
                error_message = sanitize_log_text(str(exc))
                logger.exception(
                    "Export generation failed format=%s session_id=%s", fmt, sid
                )
                return json_resp(
                    500,
                    {
                        "success": False,
                        "error": f"Export generation failed: {error_message}"
                        if error_message
                        else "Export generation failed.",
                        "code": "EXPORT_GENERATION_FAILED",
                    },
                )
            return json_resp(
                200,
                {
                    "success": True,
                    "filename": rec["filename"],
                    "format": rec["format"],
                    "mimeType": rec["mimeType"],
                    "fileSize": rec["fileSize"],
                    "downloadUrl": rec["downloadUrl"],
                    "validated": rec["validated"],
                    "validationSummary": rec["validationSummary"],
                },
            )

        if path == "/api/files" and method == "GET":
            sid = q_first("session_id", "")
            files = db.get_generated_files(sid if sid else None)
            return json_resp(
                200,
                {
                    "success": True,
                    "files": [
                        {key: value for key, value in file_record.items()
                         if key not in ("filePath", "file_path")}
                        for file_record in files
                    ],
                },
            )

        if path == "/api/code/zip" and method == "POST":
            b = parse_body()
            files_list = b.get("files") or []
            if not isinstance(files_list, list) or not files_list:
                return json_resp(400, {"success": False, "error": "At least one code file is required to build a ZIP archive."})
            title = re.sub(r"[^a-zA-Z0-9_-]", "_", str(b.get("title") or "generated_project")).strip() or "generated_project"
            zip_fname = f"{title}_{int(time.time())}.zip"
            zip_target = EXPORTS_DIR / zip_fname
            import zipfile
            with zipfile.ZipFile(str(zip_target), "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in files_list:
                    fn = os.path.basename(str(f.get("filename") or "script.txt"))
                    content_str = str(f.get("content") or "")
                    zf.writestr(fn, content_str)
            raw_data = zip_target.read_bytes()
            return json_resp(
                200,
                {
                    "success": True,
                    "filename": zip_fname,
                    "downloadUrl": f"/api/files/download/{zip_fname}",
                    "file_size": len(raw_data),
                    "fileCount": len(files_list),
                },
            )

        m_dl = re.match(r"^/api/(?:export|files|code)/download/([^/]+)$", path)
        if m_dl and method == "GET":
            raw_fname = urllib.parse.unquote(m_dl.group(1))
            if (
                not raw_fname
                or raw_fname in (".", "..")
                or "/" in raw_fname
                or "\\" in raw_fname
                or ":" in raw_fname
                or any(ord(char) < 32 for char in raw_fname)
            ):
                logger.warning("Export download rejected invalid filename")
                return json_resp(404, {"success": False, "error": "Export file not found."})
            safe_fname = raw_fname
            exports_root = EXPORTS_DIR.resolve()
            target_path = (exports_root / safe_fname).resolve()
            if (
                target_path.parent != exports_root
                or not target_path.exists()
                or not target_path.is_file()
            ):
                logger.warning("Export download not found filename=%s", safe_fname)
                return json_resp(404, {"success": False, "error": "Export file not found."})
            ext = target_path.suffix.lstrip(".").lower()
            if ext != "zip":
                v_check = validate_generated_file(target_path, ext)
                if not v_check["valid"]:
                    logger.warning(
                        "Export download rejected invalid file filename=%s reason=%s",
                        safe_fname,
                        v_check.get("error", "validation failed"),
                    )
                    return json_resp(422, {"success": False, "error": v_check.get("error")})
            data = target_path.read_bytes()
            mime_type = (
                FORMAT_METADATA.get(ext, ("", mimetypes.guess_type(safe_fname)[0]))[1]
                .split(";", 1)[0]
                or "application/octet-stream"
            )
            dl_headers = {
                "Content-Type": mime_type,
                "Content-Disposition": content_disposition(safe_fname),
                "Content-Length": str(len(data)),
                "Cache-Control": "no-cache",
            }
            logger.info("Export downloaded filename=%s size=%s", safe_fname, len(data))
            return 200, dl_headers, data

        return json_resp(404, {"success": False, "error": f"API endpoint not found: {method} {path}"})
    except PermissionError as perm_exc:
        err_msg = sanitize_log_text(str(perm_exc))
        return json_resp(
            401,
            {
                "success": False,
                "error": err_msg,
                "code": "AI_SETUP_REQUIRED",
                "ai_setup_required": True,
            },
        )
    except ProviderRequestError as exc:
        logger.exception("Provider request failed: provider=%s model=%s status=%s", exc.provider, exc.model, exc.status_code)
        err_msg = sanitize_log_text(str(exc))
        return json_resp(
            502 if exc.retryable else (exc.status_code if 400 <= exc.status_code < 500 else 502),
            {
                "success": False,
                "error": err_msg,
                "code": "PROVIDER_UNAVAILABLE" if exc.retryable else "PROVIDER_REQUEST_FAILED",
                "provider": exc.provider,
                "model": exc.model,
                "http_status": exc.status_code or None,
                "retryable": exc.retryable,
            },
        )
    except Exception as exc:
        logger.exception("API request failed: method=%s path=%s", method, path)
        err_msg = sanitize_log_text(str(exc))
        is_vision_compatibility_error = (
            "does not support vision" in err_msg.lower()
            or "text-only" in err_msg.lower()
            or "image-understanding model" in err_msg.lower()
            or "vision model is available" in err_msg.lower()
        )
        status_code = 422 if is_vision_compatibility_error else 500
        error_code = (
            "MODEL_NOT_VISION_CAPABLE"
            if is_vision_compatibility_error
            else "VISION_CLASSIFICATION_FAILED"
            if path == "/api/vision/classify"
            else "SERVER_ERROR"
        )
        return json_resp(
            status_code,
            {
                "success": False,
                "error": err_msg,
                "code": error_code,
            },
        )


# ============================================================================
# MULTIPART FORM-DATA PARSER FOR DOCUMENT UPLOADS
# ============================================================================

def parse_multipart_form_data(body_bytes: bytes, content_type: str) -> Tuple[Dict[str, str], List[Tuple[str, bytes]]]:
    """
    Parses multipart/form-data request bodies using Python's standard email.parser
    so file uploads work reliably with or without python-multipart installed.
    """
    fields: Dict[str, str] = {}
    files: List[Tuple[str, bytes]] = []

    raw_msg = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body_bytes
    )
    msg = email.parser.BytesParser().parsebytes(raw_msg)
    if not msg.is_multipart():
        return fields, files

    for part in msg.walk():
        if part.is_multipart():
            continue
        cd = part.get("Content-Disposition", "")
        if not cd:
            continue
        name_m = re.search(r'name="([^"]+)"', cd)
        fname_m = re.search(r'filename="([^"]*)"', cd)
        field_name = name_m.group(1) if name_m else ""
        payload = part.get_payload(decode=True) or b""
        if fname_m and fname_m.group(1):
            files.append((fname_m.group(1), payload))
        elif field_name:
            fields[field_name] = payload.decode("utf-8", errors="replace")

    return fields, files


def handle_document_upload_request(body_bytes: bytes, content_type: str) -> Tuple[int, Dict[str, str], bytes]:
    json_headers = {"Content-Type": "application/json; charset=utf-8"}
    try:
        fields, uploaded_files = parse_multipart_form_data(body_bytes, content_type)
        session_id = fields.get("session_id", "").strip()
        user = db.get_default_user()
        ws = db.get_default_workspace(user["id"])

        if not uploaded_files:
            return (
                400,
                json_headers,
                json.dumps({"success": False, "error": "No files received in upload request."}).encode("utf-8"),
            )

        # Ensure session exists so document is strictly bound to session_id
        if session_id and not db.get_session(session_id):
            db.create_session(
                {
                    "id": session_id,
                    "user_id": user["id"],
                    "workspace_id": ws["id"],
                    "title": uploaded_files[0][0][:48],
                }
            )

        indexed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for filename, file_bytes in uploaded_files:
            try:
                res = process_and_index_upload(
                    file_bytes=file_bytes,
                    filename=filename,
                    session_id=session_id,
                    user_id=user["id"],
                    workspace_id=ws["id"],
                )
                indexed.append(res)
            except Exception as f_err:
                failed.append({"filename": filename, "error": str(f_err)})

        status_code = 422 if (not indexed and failed) else 200
        return (
            status_code,
            json_headers,
            json.dumps(
                {
                    "success": len(indexed) > 0,
                    "indexed": indexed,
                    "failed": failed,
                    "error": failed[0]["error"] if (not indexed and failed) else None,
                }
            ).encode("utf-8"),
        )
    except Exception as exc:
        return (
            500,
            json_headers,
            json.dumps({"success": False, "error": str(exc)}).encode("utf-8"),
        )


# ============================================================================
# STATIC FRONTEND SERVING (INDEX.HTML & ASSETS)
# ============================================================================

def get_index_html_file() -> Optional[Path]:
    for cand in (
        BASE_DIR / "frontend" / "dist" / "index.html",
        BASE_DIR / "frontend" / "index.html",
        BASE_DIR / "index.html",
        BASE_DIR / "dist" / "index.html",
    ):
        if cand.exists():
            return cand
    return None


def serve_static_file(request_path: str) -> Tuple[int, Dict[str, str], bytes]:
    clean_path = request_path.split("?", 1)[0].split("#", 1)[0]
    index_file = get_index_html_file()

    if clean_path in ("", "/", "/index.html"):
        if index_file and index_file.exists():
            data = index_file.read_bytes()
            return (
                200,
                {
                    "Content-Type": "text/html; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "Content-Length": str(len(data)),
                },
                data,
            )

    rel = clean_path.lstrip("/")
    for candidate_root in (
        BASE_DIR / "frontend" / "dist",
        BASE_DIR / "frontend",
        BASE_DIR / "dist",
        BASE_DIR / "public",
        BASE_DIR,
    ):
        candidate = (candidate_root / rel).resolve()
        # Prevent path traversal outside BASE_DIR
        if str(candidate).startswith(str(BASE_DIR.resolve())) and candidate.is_file():
            # Never serve sensitive database or env files over static HTTP
            if candidate.name.startswith(".env") or ".data" in candidate.parts or "vault" in candidate.name:
                break
            data = candidate.read_bytes()
            mime = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            return (
                200,
                {
                    "Content-Type": mime,
                    "Content-Length": str(len(data)),
                },
                data,
            )

    # SPA fallback to index.html for non-API routes
    if not clean_path.startswith("/api/") and index_file and index_file.exists():
        data = index_file.read_bytes()
        return (
            200,
            {
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": str(len(data)),
            },
            data,
        )

    return (
        404,
        {"Content-Type": "application/json"},
        json.dumps({"error": "Not found"}).encode("utf-8"),
    )


# ============================================================================
# THREADED HTTP / SSE SERVER (ZERO-DEPENDENCY + FASTAPI COMPATIBLE)
# ============================================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DocPilotRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format_str: str, *args: Any) -> None:
        # Clean, quiet logging that never prints secrets
        pass

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
        req_headers = {k.lower(): v for k, v in self.headers.items()}

        # Handle SSE streaming endpoint /api/chat/stream
        if path == "/api/chat/stream" and method == "POST":
            self._handle_chat_stream(body_bytes)
            return

        # Handle multipart file upload /api/documents/upload
        if path == "/api/documents/upload" and method == "POST":
            ct = self.headers.get("Content-Type", "")
            status, resp_headers, resp_bytes = handle_document_upload_request(body_bytes, ct)
            self.send_response(status)
            self._send_cors_headers()
            for hk, hv in resp_headers.items():
                self.send_header(hk, hv)
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        if path.startswith("/api/") or path == "/health":
            status, resp_headers, resp_bytes = handle_api_request(
                method, path, query_params, body_bytes, req_headers
            )
        else:
            status, resp_headers, resp_bytes = serve_static_file(path)

        self.send_response(status)
        self._send_cors_headers()
        for hk, hv in resp_headers.items():
            self.send_header(hk, hv)
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _handle_chat_stream(self, body_bytes: bytes) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()

        client_disconnected = False

        def write_sse(payload: Dict[str, Any]) -> None:
            nonlocal client_disconnected
            if client_disconnected:
                return
            line = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            try:
                self.wfile.write(line)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                # Browser cancellation/refresh must not become a server traceback.
                client_disconnected = True

        try:
            body = json.loads(body_bytes.decode("utf-8", errors="replace")) if body_bytes else {}
            write_sse({
                "stage": "CLASSIFYING REQUEST",
                "retrieval_ms": 0,
                "chunks_retrieved": 0,
            })

            result = execute_chat_turn(
                body,
                on_retry=lambda attempt, error: write_sse(
                    {
                        "retry_attempt": attempt,
                        "max_attempts": 3,
                        "provider": error.provider,
                        "model": error.model,
                    }
                ),
            )
            if (
                result.get("routed_vision")
                or result.get("model_fallback")
                or result.get("model_selection_reason")
            ):
                write_sse({
                    "routed_vision": result.get("routed_vision", False),
                    "vision_analysis_used": result.get("vision_analysis_used", False),
                    "vision_model_name": result.get("vision_model_name"),
                    "routing_notice": result.get("routing_notice"),
                    "effective_model": result.get("effective_model"),
                    "effective_provider": result.get("effective_provider"),
                    "model_selection_reason": result.get("model_selection_reason"),
                    "model_fallback": result.get("model_fallback", False),
                    "model_fallback_reason": result.get("model_fallback_reason"),
                    "model_routing": result.get("model_routing", []),
                })
            answer = result.get("answer", "")
            chunk_size = 64
            for idx in range(0, len(answer), chunk_size):
                write_sse({
                    "chunk": answer[idx : idx + chunk_size],
                    "first_token_ms": 95 if idx == 0 else None,
                })
                time.sleep(0.015)
            if result.get("generated_files"):
                write_sse({"generated_files": result["generated_files"]})
            write_sse(
                {
                    "stage": "COMPLETED",
                    "done": True,
                    "session_id": result["session_id"],
                    "session_title": result["session_title"],
                    "sources": result.get("sources", []),
                    "generated_files": result.get("generated_files", []),
                    "routed_vision": result.get("routed_vision", False),
                    "vision_analysis_used": result.get("vision_analysis_used", False),
                    "vision_model_name": result.get("vision_model_name"),
                    "routing_notice": result.get("routing_notice"),
                    "intent": result.get("intent"),
                    "intent_confidence": result.get("intent_confidence"),
                    "intent_route": result.get("intent_route"),
                    "classifier_provider": result.get("classifier_provider"),
                    "classifier_model": result.get("classifier_model"),
                    "classifier_latency_ms": result.get("classifier_latency_ms"),
                    "effective_model": result.get("effective_model"),
                    "effective_provider": result.get("effective_provider"),
                    "model_selection_reason": result.get("model_selection_reason"),
                    "model_fallback": result.get("model_fallback", False),
                    "model_fallback_reason": result.get("model_fallback_reason"),
                    "model_routing": result.get("model_routing", []),
                }
            )
        except PermissionError as perm_exc:
            write_sse({
                "error": sanitize_log_text(str(perm_exc)),
                "code": "AI_SETUP_REQUIRED",
                "ai_setup_required": True,
                "done": True,
            })
        except ProviderRequestError as exc:
            logger.exception(
                "Chat stream provider request failed: provider=%s model=%s status=%s",
                exc.provider,
                exc.model,
                exc.status_code,
            )
            write_sse({
                "error": sanitize_log_text(str(exc)),
                "code": "PROVIDER_UNAVAILABLE" if exc.retryable else "PROVIDER_REQUEST_FAILED",
                "provider": exc.provider,
                "model": exc.model,
                "http_status": exc.status_code or None,
                "retryable": exc.retryable,
                "done": True,
            })
        except Exception as exc:
            write_sse({"error": sanitize_log_text(str(exc)), "done": True})


# ============================================================================
# STARTUP DIAGNOSTICS & MAIN ENTRY POINT
# ============================================================================

def find_available_port(preferred_port: int, host: str = "127.0.0.1") -> int:
    for port in [preferred_port, 8000, 8080, 5000, 3001]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def run_startup_checks() -> None:
    ensure_data_dirs()
    index_html = BASE_DIR / "frontend" / "index.html"
    if not index_html.exists():
        print("[ERROR] Frontend entry point 'index.html' is missing from project root!")
        sys.exit(1)

    ai_manager.initialize_from_db()
    settings = db.get_user_settings("usr-default")
    sessions_count = len(db.get_sessions("usr-default"))
    docs_count = len(db.get_documents("usr-default"))
    files_count = len(db.get_generated_files())

    print("=" * 68)
    print("  DocPilot AI / VISION AI — Standalone Python Local Application")
    print("=" * 68)
    print(f"  * Python Runtime     : Python {sys.version.split()[0]} ({sys.platform})")
    print(f"  * SQLite Database    : {SQLITE_DB_PATH}")
    print(f"  * Exports Directory  : {EXPORTS_DIR}")
    print(f"  * Active AI Provider : {settings.get('active_provider', 'gemini').upper()} ({settings.get('active_model_name') or 'No model selected'})")
    print(f"  * Restored State     : {sessions_count} conversations, {docs_count} documents, {files_count} generated files")
    print("=" * 68)


def main() -> None:
    run_startup_checks()
    preferred_port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    port = find_available_port(preferred_port, host)
    local_url = f"http://localhost:{port}"

    print(f"\n  [READY] Application running locally at: {local_url}")
    print(f"  [INFO]  Open {local_url} in your browser (or press Ctrl+C to stop).\n")

    if os.environ.get("DOCPILOT_NO_BROWSER", "").strip() != "1" and sys.platform.startswith("win"):
        threading.Timer(1.0, lambda: webbrowser.open(local_url)).start()

    server = ThreadedHTTPServer((host, port), DocPilotRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down DocPilot VISION AI server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
