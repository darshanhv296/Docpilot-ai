"""
DocPilot AI / VISION AI — Retrieval-Augmented Generation Service (rag_service.py)
Provides hybrid vector + keyword + phrase + section-heading retrieval with strict
conversation and document isolation. Never mixes chunks across unrelated sessions.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.services.ai_provider import ai_manager
from app.database.db import db


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "how", "why",
    "are", "was", "were", "have", "has", "had", "can", "could", "would", "should",
    "about", "into", "over", "under", "between", "Please", "please", "explain",
    "tell", "give", "summarize", "document", "uploaded", "file",
}


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    return dot / denom if denom > 0 else 0.0


def extract_query_terms(query: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9_]{2,}", (query or "").lower())
    filtered = [t for t in tokens if t not in STOPWORDS]
    return filtered if filtered else tokens


def retrieve_relevant_chunks(
    user_id: str,
    query: str,
    document_ids: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Retrieves relevant chunks strictly scoped to the active conversation (`session_id`)
    and explicitly selected `document_ids`.
    Never mixes documents from unrelated conversations.
    """
    if document_ids is not None and len(document_ids) == 0:
        return {"context": "", "sources": [], "chunks": []}

    session_docs = db.get_documents(user_id=user_id, session_id=session_id)
    if document_ids:
        # Filter to requested document_ids that belong to this session (or explicit user doc id)
        session_doc_ids = {d["id"] for d in session_docs}
        valid_ids = [did for did in document_ids if did in session_doc_ids]
        if not valid_ids:
            # Check if the explicit document_id exists for this user
            for did in document_ids:
                doc_obj = db.get_document(did)
                if doc_obj and doc_obj.get("user_id") == user_id:
                    if not session_id or not doc_obj.get("session_id") or doc_obj.get("session_id") == session_id:
                        valid_ids.append(did)
        if not valid_ids:
            return {"context": "", "sources": [], "chunks": []}
        target_doc_ids = valid_ids
    else:
        target_doc_ids = [d["id"] for d in session_docs]

    if not target_doc_ids:
        return {"context": "", "sources": [], "chunks": []}

    all_chunks = db.get_chunks(user_id=user_id, document_ids=target_doc_ids, session_id=session_id)
    if not all_chunks:
        all_chunks = db.get_chunks(user_id=user_id, document_ids=target_doc_ids)
    if not all_chunks:
        return {"context": "", "sources": [], "chunks": []}

    q_vec = ai_manager.generate_embedding(query or "")
    q_terms = extract_query_terms(query or "")
    q_clean = re.sub(r"[^\w\s]", " ", (query or "").lower()).strip()

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in all_chunks:
        content_lower = (chunk.get("content") or "").lower()
        section_lower = (chunk.get("section_title") or "").lower()
        filename_lower = (chunk.get("filename") or "").lower()

        c_vec = chunk.get("embedding") or []
        vec_score = cosine_similarity(q_vec, c_vec) if c_vec else 0.0

        kw_hits = 0.0
        if q_terms:
            for term in q_terms:
                if term in content_lower:
                    kw_hits += 1.0
                if term in section_lower or term in filename_lower:
                    kw_hits += 0.5
            kw_score = kw_hits / max(len(q_terms), 1)
        else:
            kw_score = 0.0

        phrase_boost = 0.25 if (len(q_clean) > 4 and q_clean in content_lower) else 0.0
        intro_boost = 0.1 if int(chunk.get("chunk_index", 0)) == 0 else 0.0

        total_score = (vec_score * 0.45) + (kw_score * 0.45) + phrase_boost + intro_boost
        scored.append((total_score, chunk))

    scored.sort(key=lambda item: (item[0], -int(item[1].get("chunk_index", 0))), reverse=True)
    selected = [item[1] for item in scored[:top_k]]

    # Sort selected chunks in document & page order for coherent reading
    selected.sort(key=lambda c: (c.get("document_id", ""), int(c.get("chunk_index", 0))))

    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for c in selected:
        fname = c.get("filename", "Document")
        page = c.get("page_number", 1)
        sec = c.get("section_title") or "Section"
        c_idx = int(c.get("chunk_index", 0)) + 1
        context_blocks.append(
            f"[Source: {fname} | Page {page} | {sec} | Chunk #{c_idx}]\n{c.get('content', '')}"
        )
        sources.append(
            {
                "document_id": c.get("document_id", ""),
                "document_name": fname,
                "title": fname,
                "page_number": page,
                "section_title": sec,
                "chunk_index": c_idx,
                "snippet": (c.get("content") or "")[:200],
            }
        )

    return {
        "context": "\n\n---\n\n".join(context_blocks),
        "sources": sources,
        "chunks": selected,
    }
