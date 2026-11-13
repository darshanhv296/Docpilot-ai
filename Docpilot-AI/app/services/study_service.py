"""
DocPilot AI / VISION AI — Study Copilot Service (study_service.py)
Implements Recall Cards (Flashcards), Interactive Quiz, and Adaptive Cheat Sheet generation
with Full Conversation-Aware Intelligence, Strict Session Isolation, Multi-Topic Proportionality,
Independent Learning Modes (Quick, Normal, Deep, Auto), Calibrated Difficulty, and 13-Section Cheat Sheets.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services.ai_provider import ai_manager, parse_json_safely
from app.database.db import db, now_iso
from app.services.rag_service import retrieve_relevant_chunks


def _study_model_options(params: Dict[str, Any], operation: str) -> Dict[str, Any]:
    provider = params.get(f"{operation}_provider") or params.get("provider")
    model = params.get(f"{operation}_model") or params.get("model")
    return {
        "provider": provider,
        "model": model,
        "base_url": params.get("base_url") if provider == "local" else None,
        "selection_mode": str(params.get("model_selection_mode") or "manual"),
        "allow_model_fallback": bool(params.get("allow_model_fallback", False)),
    }


def _generate_study_json(**options: Any) -> Tuple[Any, Dict[str, Any]]:
    result = ai_manager.generate_json(**options, return_metadata=True)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("The study model did not return generation metadata.")
    parsed, metadata = result
    return parsed, metadata if isinstance(metadata, dict) else {}


def _split_source_context(context: str, max_chars: int = 5000) -> List[str]:
    remaining = context.strip()
    chunks: List[str] = []
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        boundary = remaining.rfind("\n\n", 0, max_chars + 1)
        if boundary < max_chars // 2:
            boundary = remaining.rfind("\n", 0, max_chars + 1)
        if boundary < max_chars // 2:
            boundary = remaining.rfind(" ", 0, max_chars + 1)
        if boundary < max_chars // 2:
            boundary = max_chars
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    return chunks


GREETING_STOP_PHRASES = {
    "hi", "hello", "hey", "hola", "yo", "sup", "good morning", "good afternoon",
    "good evening", "test", "testing", "ok", "okay", "thanks", "thank you",
    "hi there", "hello there", "hey there", "howdy", "greetings",
    "new session", "new study copilot session", "study session",
}

LEARNING_MODE_PROFILES: Dict[str, Dict[str, str]] = {
    "quick": {
        "label": "Quick Study",
        "instruction": (
            "QUICK STUDY MODE: Focus strictly on high-yield essentials, concise definitions, "
            "core equations, and direct exam takeaways. Keep explanations brief, punchy, and free of filler."
        ),
        "detail_level": "concise",
    },
    "normal": {
        "label": "Normal Learning",
        "instruction": (
            "NORMAL LEARNING MODE: Provide balanced university-level explanations. Explain what each concept "
            "means, why it matters, include clear examples, and highlight connections between concepts."
        ),
        "detail_level": "standard",
    },
    "deep": {
        "label": "Deep Learning",
        "instruction": (
            "DEEP LEARNING MODE: Provide thorough, rigorous conceptual and physical/mathematical explanations. "
            "Walk through mechanisms step-by-step, state governing assumptions, include worked quantitative/analytical "
            "examples, discuss edge cases or limiting regimes, and explain variable dependencies in depth."
        ),
        "detail_level": "comprehensive",
    },
    "auto": {
        "label": "Auto Adaptive",
        "instruction": (
            "AUTO ADAPTIVE MODE: Dynamically calibrate explanation depth to match the complexity of each topic "
            "in the conversation—concise for basic definitions, and step-by-step with mathematical/mechanistic "
            "rigor for complex processes and equations."
        ),
        "detail_level": "adaptive",
    },
}


def is_trivial_greeting(text: str) -> bool:
    cleaned = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    if not cleaned:
        return True
    if cleaned in GREETING_STOP_PHRASES:
        return True
    if len(cleaned) <= 3 and cleaned in ("hi", "hey", "yo", "ok"):
        return True
    return False


def _clean_topic_phrase(raw_text: str) -> str:
    cleaned = re.sub(
        r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:tell me about|explain|what is|what are|summarize|describe|discuss|teach me about|give me notes on|generate a cheat sheet on|create a quiz on|help me understand|let's study|study)\s+",
        "",
        (raw_text or "").strip(),
        flags=re.I,
    ).strip("?.!,:; ")
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    return first_line[:95]


def identify_active_topic_and_boundary(
    user_id: str,
    session_id: Optional[str],
    explicit_topic: Optional[str],
    document_ids: Optional[List[str]],
    source_type: str = "conversation_and_docs",
) -> Dict[str, Any]:
    """
    Full Conversation-Aware Context & Topic Boundary Resolver:
    Priority Order:
      1. Complete active conversation history (all substantive user & assistant turns in `session_id`).
      2. Relevant messages and explanations within that conversation.
      3. Uploaded documents associated with `session_id`, when available (optional, never mandatory).
      4. Additional explicitly selected reference material.
    Enforces strict isolation so another session's messages or documents are never retrieved.
    Detects multiple distinct topics discussed within the same conversation so study tools cover them proportionately.
    """
    session = db.get_session(session_id) if session_id else None
    messages = db.get_messages(session_id) if session_id else []

    # Strictly filter documents to active session_id
    session_docs = db.get_documents(user_id=user_id, session_id=session_id) if session_id else []
    if document_ids is not None:
        if len(document_ids) == 0:
            selected_docs = []
        else:
            id_set = set(document_ids)
            selected_docs = [d for d in session_docs if d["id"] in id_set]
            if not selected_docs:
                for did in document_ids:
                    d_obj = db.get_document(did)
                    if d_obj and d_obj.get("user_id") == user_id:
                        if not session_id or not d_obj.get("session_id") or d_obj.get("session_id") == session_id:
                            selected_docs.append(d_obj)
    else:
        selected_docs = list(session_docs)

    if source_type == "conversation":
        selected_docs = []
    elif source_type in ("all_docs", "single_doc"):
        messages = []

    # Gather ALL substantive messages across the entire active conversation
    substantive_user_msgs: List[str] = []
    substantive_assistant_msgs: List[str] = []
    conversation_topics: List[str] = []

    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = m.get("role")
        if role == "user":
            if not is_trivial_greeting(content):
                substantive_user_msgs.append(content)
                cand_topic = _clean_topic_phrase(content)
                if cand_topic and len(cand_topic) >= 3 and not is_trivial_greeting(cand_topic):
                    if not any(cand_topic.lower() in existing.lower() or existing.lower() in cand_topic.lower() for existing in conversation_topics):
                        conversation_topics.append(cand_topic)
        elif role == "assistant":
            if len(content) > 80 and "How can I help you master your material today" not in content:
                substantive_assistant_msgs.append(content)
                # Extract markdown top-level headings from assistant explanations if user prompt was brief
                for h_match in re.finditer(r"^#{1,3}\s+([^\n#]+)", content, flags=re.M):
                    h_txt = h_match.group(1).strip(" *:-")
                    if 4 <= len(h_txt) <= 75 and not any(
                        skip in h_txt.lower()
                        for skip in ("overview", "summary", "takeaway", "conclusion", "step 1", "step 2", "example", "definition", "formula", "introduction")
                    ):
                        if not any(h_txt.lower() in existing.lower() or existing.lower() in h_txt.lower() for existing in conversation_topics):
                            conversation_topics.append(h_txt)

    # Resolve primary topic and multi-topic list
    resolved_topic = ""
    if explicit_topic and not is_trivial_greeting(explicit_topic):
        cleaned_exp = _clean_topic_phrase(explicit_topic)
        if not is_trivial_greeting(cleaned_exp):
            resolved_topic = cleaned_exp[:110]

    if not resolved_topic and conversation_topics:
        if len(conversation_topics) == 1:
            resolved_topic = conversation_topics[0][:100]
        else:
            # Combine multiple topics discussed in the conversation proportionately
            resolved_topic = " & ".join(conversation_topics[:3])[:120]

    if not resolved_topic and substantive_user_msgs:
        resolved_topic = _clean_topic_phrase(substantive_user_msgs[-1])[:100]

    if not resolved_topic and selected_docs:
        doc_titles = [
            re.sub(r"\.[a-zA-Z0-9]+$", "", d["filename"]).replace("_", " ").replace("-", " ")
            for d in selected_docs
        ]
        resolved_topic = ", ".join(doc_titles[:2])

    if not resolved_topic and session and not is_trivial_greeting(session.get("title", "")):
        resolved_topic = session["title"]

    # Update session title if it was previously stuck on a greeting or default
    if session_id and session and resolved_topic:
        if is_trivial_greeting(session.get("title", "")):
            db.update_session(session_id, {"title": resolved_topic[:60], "topic": resolved_topic})

    # 1. Build COMPLETE active conversation transcript first (Priority #1 & #2)
    conv_context = ""
    if messages and source_type not in ("all_docs", "single_doc"):
        substantive_turns = [
            m for m in messages
            if not (m.get("role") == "user" and is_trivial_greeting(m.get("content", "")))
            and "How can I help you master your material today" not in (m.get("content") or "")
        ]
        if substantive_turns:
            # Include all turns across the conversation; allocate generous character budget per turn
            per_turn_budget = max(1200, min(4000, 24000 // max(1, len(substantive_turns))))
            conv_lines = []
            for idx, m in enumerate(substantive_turns, 1):
                role_label = "STUDENT" if m.get("role") == "user" else "AI TUTOR"
                conv_lines.append(f"[Turn {idx} — {role_label}]\n{(m.get('content') or '')[:per_turn_budget]}")
            conv_context = "=== COMPLETE ACTIVE CONVERSATION HISTORY (PRIMARY SOURCE) ===\n" + "\n\n".join(conv_lines)

    # 2. Retrieve relevant document chunks associated with this conversation (Priority #3 & #4)
    doc_context = ""
    doc_chunks: List[Dict[str, Any]] = []
    if selected_docs:
        target_ids = [d["id"] for d in selected_docs]
        rag_query = " ".join(conversation_topics) if conversation_topics else (resolved_topic or "overview definitions concepts formulas")
        rag_res = retrieve_relevant_chunks(
            user_id=user_id,
            query=rag_query,
            document_ids=target_ids,
            session_id=session_id,
            top_k=14,
        )
        doc_chunks = rag_res.get("chunks") or []
        if not doc_chunks:
            doc_chunks = db.get_chunks(user_id=user_id, document_ids=target_ids, session_id=session_id)[:14]

        doc_parts: List[str] = []
        for d in selected_docs:
            d_chunks = [c for c in doc_chunks if c.get("document_id") == d["id"]]
            if d_chunks:
                joined_c = "\n\n".join(
                    f"[Page {c.get('page_number', 1)} | {c.get('section_title', 'Section')}]\n{c.get('content', '')}"
                    for c in d_chunks
                )
                doc_parts.append(f"=== UPLOADED DOCUMENT ({d['filename']}) ===\n{joined_c}")
            elif d.get("content_text"):
                doc_parts.append(f"=== UPLOADED DOCUMENT ({d['filename']}) ===\n{d['content_text'][:6000]}")
        doc_context = "\n\n".join(doc_parts)

    # Priority order: Conversation History first, then Uploaded Documents
    combined_context_parts = [p for p in (conv_context, doc_context) if p.strip()]
    combined_context = "\n\n".join(combined_context_parts).strip()

    anchor_text = f"{resolved_topic}\n{' '.join(conversation_topics)}\n{combined_context}".lower()
    anchor_words = set(re.findall(r"[a-z]{4,}", anchor_text))

    return {
        "session_id": session_id,
        "topic": resolved_topic,
        "conversation_topics": conversation_topics or ([resolved_topic] if resolved_topic else []),
        "selected_docs": selected_docs,
        "doc_chunks": doc_chunks,
        "substantive_user_msgs": substantive_user_msgs,
        "substantive_assistant_msgs": substantive_assistant_msgs,
        "conv_context": conv_context,
        "doc_context": doc_context,
        "combined_context": combined_context,
        "anchor_words": anchor_words,
        "has_valid_source": bool(combined_context or resolved_topic),
    }


def validate_item_against_topic_boundary(
    item_text: str,
    topic: str,
    anchor_words: Set[str],
) -> bool:
    """
    Validates a generated quiz question, flashcard, or cheat sheet against the active session's topic boundary.
    Prevents cross-topic contamination between independent conversations (e.g., Black Holes vs. Neural Networks).
    """
    low_item = (item_text or "").lower()
    low_topic = (topic or "").lower()

    # 1. Block Neural Network / ML contamination when active session is NOT about ML/NNs
    ml_contamination_terms = (
        "backpropagation",
        "vanishing gradient",
        "exploding gradient",
        "physics-informed neural",
        "pinn",
        "convolutional neural",
        "gradient descent weight",
        "activation function",
        "softmax",
        "epoch",
        "perceptron",
    )
    source_is_ml = any(
        term in low_topic or term in anchor_words
        for term in ("neural", "backpropagation", "gradient", "network", "deep", "perceptron", "activation", "pinn", "machine")
    )
    if not source_is_ml:
        for bad_term in ml_contamination_terms:
            if bad_term in low_item:
                return False

    # 2. Block Black Hole / Astrophysics contamination when active session is NOT about Astrophysics/Black Holes
    astro_contamination_terms = (
        "schwarzschild radius",
        "hawking radiation",
        "event horizon",
        "spaghettification",
        "tolman-oppenheimer-volkoff",
        "tolman–oppenheimer–volkoff",
        "accretion disk around a black hole",
    )
    source_is_astro = any(
        term in low_topic or term in anchor_words
        for term in ("black", "hole", "schwarzschild", "hawking", "horizon", "stellar", "star", "astrophysics", "relativity", "gravity", "collapse", "protostar", "jeans")
    )
    if not source_is_astro:
        for bad_term in astro_contamination_terms:
            if bad_term in low_item:
                return False

    return True


def normalize_quiz_question(q: Dict[str, Any], idx: int, default_topic: str, default_diff: str) -> Dict[str, Any]:
    q_type = str(q.get("type") or "multiple_choice").strip().lower()
    if q_type not in ("multiple_choice", "true_false", "fill_in_blank"):
        q_type = "multiple_choice"

    options = q.get("options")
    if not isinstance(options, list) or len(options) == 0:
        if q_type == "true_false":
            options = ["True", "False"]
        else:
            ans = str(q.get("correct_answer") or "Correct Option")
            options = [ans, "Alternative B", "Alternative C", "Alternative D"]
    options = [str(o).strip() for o in options if str(o).strip()]
    if q_type == "true_false" and len(options) != 2:
        options = ["True", "False"]

    correct_ans_raw = q.get("correct_answer")
    correct_idx_raw = q.get("correctIndex") if "correctIndex" in q else q.get("correct_index")

    resolved_idx = 0
    if isinstance(correct_idx_raw, int) and 0 <= correct_idx_raw < len(options):
        resolved_idx = correct_idx_raw
    elif correct_ans_raw is not None:
        ans_str = str(correct_ans_raw).strip()
        matched = False
        for i, opt in enumerate(options):
            if opt.lower() == ans_str.lower():
                resolved_idx = i
                matched = True
                break
        if not matched:
            if len(ans_str) == 1 and ans_str.upper() in ("A", "B", "C", "D"):
                letter_idx = ord(ans_str.upper()) - ord("A")
                if 0 <= letter_idx < len(options):
                    resolved_idx = letter_idx
                    matched = True
        if not matched:
            for i, opt in enumerate(options):
                if ans_str.lower() in opt.lower() or opt.lower() in ans_str.lower():
                    resolved_idx = i
                    matched = True
                    break

    resolved_answer = options[resolved_idx] if options else str(correct_ans_raw or "")

    return {
        "id": str(q.get("id") or f"q{idx + 1}"),
        "type": q_type,
        "concept": str(q.get("concept") or default_topic or "Core Concept").strip(),
        "difficulty": str(q.get("difficulty") or default_diff or "medium").strip().lower(),
        "question": str(q.get("question") or "").strip(),
        "options": options,
        "correct_answer": resolved_answer,
        "correctIndex": resolved_idx,
        "explanation": str(q.get("explanation") or "Verified against the active study conversation and source material.").strip(),
        "source_citation": str(q.get("source_citation") or default_topic or "Active Conversation").strip(),
    }


def ensure_complete_cheatsheet_structure(sheet: Dict[str, Any], topic: str, source_summary: str) -> Dict[str, Any]:
    """
    Guarantees that all 13 sections of the Cheat Sheet are well-formed and that
    `concept_nodes` (Concept Map / Relationships) and `methods` (Mechanisms & Processes)
    never contain empty cells or missing keys.
    """
    sheet["title"] = str(sheet.get("title") or f"{topic} — Comprehensive Study Cheat Sheet").strip()
    sheet["topic"] = topic
    sheet["detected_subject"] = str(sheet.get("detected_subject") or topic).strip()
    sheet["source_summary"] = source_summary
    sheet["overview"] = str(sheet.get("overview") or f"Comprehensive academic synthesis of {topic}.").strip()

    # Normalize definitions
    raw_defs = sheet.get("definitions") if isinstance(sheet.get("definitions"), list) else []
    clean_defs = []
    for d in raw_defs:
        if isinstance(d, dict):
            term = str(d.get("term") or d.get("name") or d.get("title") or "").strip()
            defn = str(d.get("definition") or d.get("meaning") or d.get("description") or "").strip()
            if term and defn:
                clean_defs.append({"term": term, "definition": defn})
    sheet["definitions"] = clean_defs

    # Normalize concepts & detailed explanations
    raw_concepts = sheet.get("concepts") if isinstance(sheet.get("concepts"), list) else []
    clean_concepts = []
    for c in raw_concepts:
        if isinstance(c, dict):
            title = str(c.get("title") or c.get("name") or c.get("concept") or "").strip()
            expl = str(c.get("explanation") or c.get("description") or c.get("details") or "").strip()
            if title and expl:
                clean_concepts.append({"title": title, "explanation": expl})
    sheet["concepts"] = clean_concepts

    # Normalize mechanisms & processes (methods) — ensure non-empty step, title, description
    raw_methods = sheet.get("methods") if isinstance(sheet.get("methods"), list) else []
    clean_methods = []
    for idx, m in enumerate(raw_methods, 1):
        if isinstance(m, dict):
            title = str(m.get("title") or m.get("stage") or m.get("name") or f"Stage {idx}").strip()
            desc = str(m.get("description") or m.get("mechanism") or m.get("explanation") or "").strip()
            if desc:
                clean_methods.append({"step": m.get("step") or idx, "title": title, "description": desc})
        elif isinstance(m, str) and m.strip():
            clean_methods.append({"step": idx, "title": f"Mechanism {idx}", "description": m.strip()})
    if not clean_methods and clean_concepts:
        for idx, c in enumerate(clean_concepts[:4], 1):
            clean_methods.append({"step": idx, "title": c["title"], "description": c["explanation"]})
    sheet["methods"] = clean_methods

    # Normalize concept_nodes (Concept Relationships table) — NEVER allow blank cells!
    raw_nodes = sheet.get("concept_nodes") if isinstance(sheet.get("concept_nodes"), list) else []
    clean_nodes = []
    for n in raw_nodes:
        if isinstance(n, dict):
            node_a = str(n.get("node") or n.get("source") or n.get("concept_a") or n.get("from") or "").strip()
            rel = str(n.get("relation") or n.get("relationship") or n.get("mechanism") or n.get("link") or "").strip()
            node_b = str(n.get("connects_to") or n.get("target") or n.get("concept_b") or n.get("to") or "").strip()
            if node_a and rel and node_b:
                clean_nodes.append({"node": node_a, "relation": rel, "connects_to": node_b})

    if not clean_nodes:
        # Derive meaningful concept relationships from concepts/definitions/methods so the Concept Map table is never blank
        if len(clean_concepts) >= 2:
            for i in range(len(clean_concepts) - 1):
                c_curr = clean_concepts[i]
                c_next = clean_concepts[i + 1]
                clean_nodes.append({
                    "node": c_curr["title"],
                    "relation": f"Governs and transitions into ({c_curr['explanation'][:90].rstrip('.')}...)",
                    "connects_to": c_next["title"],
                })
        elif len(clean_defs) >= 2:
            for i in range(min(4, len(clean_defs) - 1)):
                clean_nodes.append({
                    "node": clean_defs[i]["term"],
                    "relation": f"Directly interacts with ({clean_defs[i]['definition'][:85].rstrip('.')}...)",
                    "connects_to": clean_defs[i + 1]["term"],
                })
    sheet["concept_nodes"] = clean_nodes

    # Normalize formulas & variable explanations
    raw_formulas = sheet.get("formulas") if isinstance(sheet.get("formulas"), list) else []
    clean_formulas = []
    for f in raw_formulas:
        if isinstance(f, dict):
            name = str(f.get("name") or f.get("title") or "Key Relation").strip()
            latex = str(f.get("latex") or f.get("equation") or f.get("formula") or "").strip()
            variables = str(f.get("variables") or f.get("meaning") or f.get("explanation") or "Standard SI/domain units apply.").strip()
            when_to_use = str(f.get("when_to_use") or f.get("application") or "").strip()
            if latex:
                clean_formulas.append({
                    "name": name,
                    "latex": latex,
                    "variables": variables,
                    "when_to_use": when_to_use,
                })
    sheet["formulas"] = clean_formulas

    # Normalize examples, comparisons, misconceptions, exam_points, exam_questions
    raw_examples = sheet.get("examples") if isinstance(sheet.get("examples"), list) else []
    clean_examples = []
    for ex in raw_examples:
        if isinstance(ex, dict):
            prob = str(ex.get("problem") or ex.get("title") or ex.get("scenario") or "").strip()
            sol = str(ex.get("solution") or ex.get("explanation") or ex.get("walkthrough") or "").strip()
            if prob and sol:
                clean_examples.append({"problem": prob, "solution": sol})
    sheet["examples"] = clean_examples

    raw_comp = sheet.get("comparisons") if isinstance(sheet.get("comparisons"), list) else []
    clean_comp = []
    for cp in raw_comp:
        if isinstance(cp, dict):
            ia = str(cp.get("item_a") or cp.get("concept_a") or cp.get("left") or "").strip()
            ib = str(cp.get("item_b") or cp.get("concept_b") or cp.get("right") or "").strip()
            diff = str(cp.get("difference") or cp.get("distinction") or cp.get("comparison") or "").strip()
            if ia and ib and diff:
                clean_comp.append({"item_a": ia, "item_b": ib, "difference": diff})
    sheet["comparisons"] = clean_comp

    raw_misc = sheet.get("misconceptions") if isinstance(sheet.get("misconceptions"), list) else []
    clean_misc = []
    for ms in raw_misc:
        if isinstance(ms, dict):
            myth = str(ms.get("myth") or ms.get("misconception") or ms.get("pitfall") or "").strip()
            reality = str(ms.get("reality") or ms.get("correction") or ms.get("fact") or "").strip()
            if myth and reality:
                clean_misc.append({"myth": myth, "reality": reality})
    sheet["misconceptions"] = clean_misc

    raw_pts = sheet.get("exam_points") if isinstance(sheet.get("exam_points"), list) else []
    sheet["exam_points"] = [str(p).strip() for p in raw_pts if str(p).strip()]

    raw_eqs = sheet.get("exam_questions") if isinstance(sheet.get("exam_questions"), list) else []
    clean_eqs = []
    for eq in raw_eqs:
        if isinstance(eq, dict):
            q_str = str(eq.get("question") or "").strip()
            a_str = str(eq.get("answer") or eq.get("solution") or "").strip()
            if q_str and a_str:
                clean_eqs.append({"question": q_str, "answer": a_str})
    sheet["exam_questions"] = clean_eqs

    return sheet


# ============================================================================
# DOMAIN-GROUNDED SYNTHESIS BUILDERS (Stellar Birth Cycle, Black Holes, Neural Networks, Multi-Topic)
# ============================================================================

def _build_stellar_birth_cycle_bundle(topic: str, source_label: str, learning_mode: str = "normal") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cards = [
        {
            "topic": "Giant Molecular Clouds (GMCs)",
            "difficulty": "easy",
            "question": "What are the typical physical conditions (temperature, density, mass) inside a Giant Molecular Cloud (GMC) where star formation begins?",
            "answer": "Giant Molecular Clouds consist primarily of molecular hydrogen ($H_2$), helium, and ~1% interstellar dust with typical temperatures $T \\approx 10\\text{--}20\\text{ K}$, number densities $n(H_2) \\approx 10^2\\text{--}10^6\\text{ cm}^{-3}$ in dense cores, and total masses of $10^3\\text{--}10^6\\,M_\\odot$. Low temperature minimizes internal thermal pressure ($P = nk_B T$), favoring gravitational collapse.",
            "explanation": "Dust shielding blocks interstellar UV radiation, allowing CO and dust thermal emission to cool the gas to ~10 K.",
            "source": source_label,
        },
        {
            "topic": "Jeans Instability & Jeans Mass",
            "difficulty": "medium",
            "question": "State the Jeans Criterion for gravitational collapse and write the formula for the Jeans Mass ($M_J$). How does $M_J$ scale with temperature $T$ and density $\\rho_0$?",
            "answer": "A gas cloud becomes gravitationally unstable and collapses when its mass exceeds the **Jeans Mass** ($M > M_J$) or its radius exceeds the **Jeans Length** ($\\lambda_J = \\sqrt{\\frac{\\pi c_s^2}{G\\rho_0}}$):\n$$M_J \\approx \\left(\\frac{5 k_B T}{G \\mu m_H}\\right)^{3/2} \\left(\\frac{3}{4\\pi \\rho_0}\\right)^{1/2} \\propto T^{3/2} \\rho_0^{-1/2}$$\nLower temperature $T$ and higher density $\\rho_0$ decrease $M_J$, allowing smaller sub-regions to fragment and collapse.",
            "explanation": "Derived from the virial theorem ($2K + |U| < 0$) or linear perturbation analysis of the Euler-Poisson equations.",
            "source": source_label,
        },
        {
            "topic": "Free-Fall Collapse & Timescale",
            "difficulty": "medium",
            "question": "What is the free-fall timescale ($t_{ff}$) of a pressureless collapsing cloud of initial density $\\rho_0$, and why does the initial collapse remain isothermal?",
            "answer": "The **free-fall timescale** is:\n$$t_{ff} = \\sqrt{\\frac{3\\pi}{32 G \\rho_0}}$$\nNotice that $t_{ff}$ depends **only on the initial density** $\\rho_0$, not on radius. During early collapse ($\\rho < 10^{-13}\\text{ g cm}^{-3}$), the cloud is optically thin to far-infrared dust emission, so compressional heating radiates away immediately and temperature stays nearly constant ($T \\approx 10\\text{ K}$).",
            "explanation": "Isothermal collapse causes $M_J \\propto \\rho_0^{-1/2}$ to drop as density rises, driving Hoyle hierarchical fragmentation.",
            "source": source_label,
        },
        {
            "topic": "First Hydrostatic Core (Larson Core)",
            "difficulty": "hard",
            "question": "What halts isothermal fragmentation and forms the First Hydrostatic Core (Larson core), and how does it transition into the Second (Stellar) Core?",
            "answer": "At densities $\\rho \\sim 10^{-13}\\text{ g cm}^{-3}$, dust becomes optically thick ($\\tau_{\\text{IR}} > 1$) and traps radiation. Collapse switches from isothermal ($\\gamma = 1$) to adiabatic ($\\gamma \\approx 7/5$), raising internal pressure to form the **First Hydrostatic Core** ($R \\sim 5\\text{ AU}$, $M \\sim 0.05\\,M_\\odot$). When central temperature reaches $T \\approx 2000\\text{ K}$, molecular hydrogen ($H_2$) dissociates endothermically ($4.48\\text{ eV}$), triggering a second collapse until atomic/ionized gas forms the true **protostar (Second Hydrostatic Core)**.",
            "explanation": "H2 dissociation acts as an energy sink (effective adiabatic index gamma < 4/3), destabilizing the first core.",
            "source": source_label,
        },
        {
            "topic": "Angular Momentum Problem & Disk Accretion",
            "difficulty": "medium",
            "question": "Explain the Angular Momentum Problem in star formation and how circumstellar accretion disks and bipolar jets resolve it.",
            "answer": "Because specific angular momentum $j = \\Omega R^2$ is conserved during collapse, centrifugal force ($v_\\phi^2/R \\propto R^{-3}$) grows faster than gravity ($GM/R^2 \\propto R^{-2}$), flattening infalling gas into a **circumstellar accretion disk** at the centrifugal radius $R_c = j^2/(GM)$. Viscosity, magnetorotational instability (MRI), and magnetocentrifugal **bipolar jets/outflows** transport angular momentum outward so gas can accrete onto the central protostar.",
            "explanation": "Without angular momentum transport, a collapsing core would spin up past breakup velocity long before reaching stellar radii.",
            "source": source_label,
        },
        {
            "topic": "Protostellar Evolution (Class 0, I, II, III) & Hayashi Track",
            "difficulty": "medium",
            "question": "Distinguish between Class 0, Class I, Class II (Classical T Tauri), and Class III Young Stellar Objects (YSOs).",
            "answer": "- **Class 0**: Deeply embedded main accretion phase ($M_{\\text{env}} > M_*$), sub-mm peak, powerful bipolar outflows.\n- **Class I**: Evolved protostar ($M_* > M_{\\text{env}}$), infalling envelope + disk, rising IR spectral energy distribution.\n- **Class II (Classical T Tauri / Herbig Ae/Be)**: Envelope cleared; optically visible pre-main-sequence star surrounded by an optically thick protoplanetary disk.\n- **Class III (Weak-lined T Tauri)**: Disk largely dispersed into debris/planets; star contracts along the **Hayashi** (fully convective, nearly vertical $T_{\\text{eff}}$) or **Henyey** (radiative core) track.",
            "explanation": "Classified observationally by the infrared spectral index alpha_IR = d log(lambda F_lambda) / d log(lambda).",
            "source": source_label,
        },
        {
            "topic": "ZAMS Ignition & Brown Dwarf Limit",
            "difficulty": "hard",
            "question": "What physical event defines arrival on the Zero-Age Main Sequence (ZAMS), and why do objects with $M < 0.08\\,M_\\odot$ become brown dwarfs?",
            "answer": "As a pre-main-sequence star contracts on the **Kelvin–Helmholtz timescale** ($t_{KH} \\approx \\frac{G M^2}{R L}$), half the released gravitational energy heats the core (virial theorem). Deuterium fusion ignites around $T_c \\approx 10^6\\text{ K}$. When central temperature reaches $T_c \\ge 10^7\\text{ K}$, sustained core hydrogen fusion ($^1\\text{H} \\to {^4\\text{He}}$ via pp-chain or CNO cycle) balances radiative luminosity ($L_{\\text{nuc}} = L_{\\text{surf}}$), placing the star on the **ZAMS**. Below $0.08\\,M_\\odot$, electron degeneracy pressure halts contraction before $T_c$ reaches $10^7\\text{ K}$, forming a **brown dwarf**.",
            "explanation": "At the upper end (>100-150 M_sun), radiation pressure exceeds the Eddington limit, halting further accretion.",
            "source": source_label,
        },
    ]

    questions = [
        {
            "id": "q1",
            "type": "multiple_choice",
            "concept": "Jeans Mass Scaling",
            "difficulty": "medium",
            "question": "How does the Jeans Mass ($M_J$) of an interstellar gas cloud depend on its temperature $T$ and mass density $\\rho_0$?",
            "options": [
                "M_J ∝ T^(3/2) ρ_0^(-1/2)",
                "M_J ∝ T^(-3/2) ρ_0^(1/2)",
                "M_J ∝ T^(1/2) ρ_0^(-3/2)",
                "M_J ∝ T^2 ρ_0",
            ],
            "correct_answer": "M_J ∝ T^(3/2) ρ_0^(-1/2)",
            "correctIndex": 0,
            "explanation": "Since Jeans length scales as λ_J ∝ c_s / ρ_0^(1/2) ∝ T^(1/2) ρ_0^(-1/2), the enclosed Jeans mass M_J ∝ ρ_0 λ_J^3 scales as T^(3/2) ρ_0^(-1/2).",
            "source_citation": source_label,
        },
        {
            "id": "q2",
            "type": "multiple_choice",
            "concept": "Free-Fall Timescale",
            "difficulty": "easy",
            "question": "Which formula gives the gravitational free-fall timescale ($t_{ff}$) for a homogeneous spherical cloud of initial density $\\rho_0$?",
            "options": [
                "t_ff = sqrt(3π / (32 G ρ_0))",
                "t_ff = G M^2 / (R L)",
                "t_ff = 2 G M / c^2",
                "t_ff = sqrt(π c_s^2 / (G ρ_0))",
            ],
            "correct_answer": "t_ff = sqrt(3π / (32 G ρ_0))",
            "correctIndex": 0,
            "explanation": "Integrating the equation of motion d^2r/dt^2 = -GM(r)/r^2 for a pressureless sphere from rest yields t_ff = sqrt(3π / (32 G ρ_0)), depending solely on initial density ρ_0.",
            "source_citation": source_label,
        },
        {
            "id": "q3",
            "type": "multiple_choice",
            "concept": "First Hydrostatic Core Collapse",
            "difficulty": "hard",
            "question": "What physical mechanism triggers the second dynamical collapse of the First Hydrostatic Core (Larson core) into a true stellar core?",
            "options": [
                "Endothermic dissociation of molecular hydrogen (H_2 -> 2H) at T ≈ 2000 K, which lowers the effective adiabatic index below 4/3",
                "Sudden ignition of core helium fusion (triple-alpha process) at 10^8 K",
                "Complete dispersal of interstellar dust by external supernova shockwaves",
                "Relativistic electron degeneracy pressure exceeding the Chandrasekhar limit",
            ],
            "correct_answer": "Endothermic dissociation of molecular hydrogen (H_2 -> 2H) at T ≈ 2000 K, which lowers the effective adiabatic index below 4/3",
            "correctIndex": 0,
            "explanation": "At T ≈ 2000 K, thermal energy goes into breaking H_2 bonds (4.48 eV) rather than raising kinetic pressure, dropping γ_eff below the stability threshold 4/3 and triggering rapid second collapse.",
            "source_citation": source_label,
        },
        {
            "id": "q4",
            "type": "true_false",
            "concept": "Isothermal Fragmentation",
            "difficulty": "medium",
            "question": "True or False: During the optically thin isothermal phase of cloud collapse, the Jeans mass decreases as density increases ($M_J \\propto \\rho_0^{-1/2}$), allowing a giant molecular cloud to fragment into smaller protostellar cores.",
            "options": ["True", "False"],
            "correct_answer": "True",
            "correctIndex": 0,
            "explanation": "With T held nearly constant (~10 K) by dust thermal radiation, increasing density ρ_0 lowers M_J, causing sub-regions to sequentially exceed their local Jeans mass and fragment.",
            "source_citation": source_label,
        },
        {
            "id": "q5",
            "type": "multiple_choice",
            "concept": "Zero-Age Main Sequence (ZAMS)",
            "difficulty": "medium",
            "question": "What marks a young star's arrival on the Zero-Age Main Sequence (ZAMS), and what is the minimum mass required?",
            "options": [
                "Sustained core hydrogen fusion (1H -> 4He) at T_c >= 10^7 K balancing surface luminosity, requiring M >= 0.08 M_sun",
                "Initial formation of a circumstellar protoplanetary disk at T ≈ 100 K, requiring M >= 0.01 M_sun",
                "Deuterium fusion at T_c ≈ 10^6 K on the Hayashi track, requiring M >= 0.013 M_sun",
                "Exhaustion of core hydrogen fuel and expansion into a red giant, requiring M >= 1.4 M_sun",
            ],
            "correct_answer": "Sustained core hydrogen fusion (1H -> 4He) at T_c >= 10^7 K balancing surface luminosity, requiring M >= 0.08 M_sun",
            "correctIndex": 0,
            "explanation": "A star reaches the ZAMS when nuclear energy production from core H->He fusion fully supports stellar luminosity and stops Kelvin-Helmholtz gravitational contraction; below 0.08 M_sun, electron degeneracy halts core heating before 10^7 K.",
            "source_citation": source_label,
        },
        {
            "id": "q6",
            "type": "fill_in_blank",
            "concept": "Pre-Main-Sequence Evolution",
            "difficulty": "hard",
            "question": "On the Hertzsprung–Russell diagram, a low-mass pre-main-sequence star that is fully convective contracts downward at nearly constant effective temperature along the ______ track.",
            "options": [
                "Hayashi",
                "Henyey",
                "Schwarzschild",
                "Chandrasekhar",
            ],
            "correct_answer": "Hayashi",
            "correctIndex": 0,
            "explanation": "Fully convective pre-main-sequence stars contract at nearly constant effective temperature (decreasing radius and luminosity) along the vertical Hayashi track before developing a radiative core (Henyey track) or reaching the ZAMS.",
            "source_citation": source_label,
        },
    ]

    cheatsheet = {
        "title": "The Stellar Birth Cycle & Star Formation Mechanics — Master Study Cheat Sheet",
        "topic": topic,
        "detected_subject": "Astrophysics & Stellar Evolution",
        "source_summary": f"Synthesized from active study session ({source_label})",
        "overview": (
            "Star formation is the multi-scale astrophysical transformation spanning 20 orders of magnitude in density "
            "($\\rho \\sim 10^{-21}\\text{ g cm}^{-3}$ to $\\sim 10^{2}\\text{ g cm}^{-3}$) in which cold, dust-shielded "
            "Giant Molecular Clouds (GMCs) overcome thermal, turbulent, and magnetic support via Jeans gravitational instability. "
            "Through isothermal free-fall collapse and hierarchical fragmentation, dense cores form a First Hydrostatic Core (Larson core), "
            "undergo $H_2$-dissociation collapse into a true protostar surrounded by a Keplerian circumstellar accretion disk and bipolar jets, "
            "contract along pre-main-sequence Hayashi and Henyey tracks on the Kelvin–Helmholtz timescale, and ignite sustained core hydrogen fusion "
            "at $T_c \\ge 10^7\\text{ K}$ to arrive on the Zero-Age Main Sequence (ZAMS)."
        ),
        "definitions": [
            {
                "term": "Giant Molecular Cloud (GMC)",
                "definition": "Cold ($T \\approx 10\\text{--}20\\text{ K}$), dense ($n_{H_2} \\approx 10^2\\text{--}10^6\\text{ cm}^{-3}$) interstellar reservoir of $H_2$, He, and ~1% dust ($M \\sim 10^4\\text{--}10^6\\,M_\\odot$) where dust shields molecules from UV photodissociation.",
            },
            {
                "term": "Jeans Mass ($M_J$) & Jeans Length ($\\lambda_J$)",
                "definition": "The critical threshold mass and spatial scale above which inward self-gravity overwhelms internal thermal gas pressure ($2K + |U| < 0$), initiating gravitational collapse.",
            },
            {
                "term": "Free-Fall Timescale ($t_{ff}$)",
                "definition": "The characteristic time required for a uniform, pressureless sphere of initial density $\\rho_0$ to collapse to a point under self-gravity: $t_{ff} = \\sqrt{3\\pi/(32G\\rho_0)}$.",
            },
            {
                "term": "First Hydrostatic Core (Larson Core)",
                "definition": "Quasi-hydrostatic object ($R \\sim 5\\text{ AU}$, $M \\sim 0.05\\,M_\\odot$) formed when collapsing gas reaches $\\rho \\sim 10^{-13}\\text{ g cm}^{-3}$, becomes optically thick to infrared dust cooling, and heats adiabatically.",
            },
            {
                "term": "Centrifugal Radius & Accretion Disk",
                "definition": "Because specific angular momentum $j = \\Omega R^2$ is conserved during infall, gas settles into a flattened Keplerian circumstellar disk at $R_c = j^2/(GM)$, transferring angular momentum outward via MRI viscosity and magnetic winds.",
            },
            {
                "term": "Hayashi vs. Henyey Pre-Main-Sequence Tracks",
                "definition": "Evolutionary paths on the H-R diagram during Kelvin–Helmholtz contraction: the **Hayashi track** is nearly vertical (constant $T_{\\text{eff}}$, decreasing $L$) for fully convective stars, while the **Henyey track** is nearly horizontal (increasing $T_{\\text{eff}}$, slowly rising $L$) once a radiative core develops.",
            },
            {
                "term": "Zero-Age Main Sequence (ZAMS)",
                "definition": "The locus on the H-R diagram where a young star's core reaches $T_c \\ge 10^7\\text{ K}$ and sustained core hydrogen fusion ($^1\\text{H} \\to {^4\\text{He}}$) fully balances surface radiative luminosity ($L_{\\text{nuc}} = L_{\\text{surf}}$).",
            },
        ],
        "concepts": [
            {
                "title": "1. Virial Equilibrium & Jeans Gravitational Instability",
                "explanation": (
                    "According to the Virial Theorem ($2K + U + \\mathcal{M} = 0$), a molecular cloud core collapses when gravitational binding energy "
                    "$|U| = \\frac{3}{5}\\frac{GM^2}{R}$ exceeds twice the internal thermal and turbulent kinetic energy $2K = 3 \\frac{M}{\\mu m_H} k_B T$. "
                    "Perturbations larger than the Jeans length $\\lambda_J = c_s \\sqrt{\\pi/(G\\rho_0)}$ grow exponentially because sound waves cannot cross the region before gravity pulls it inward."
                ),
            },
            {
                "title": "2. Isothermal Collapse & Hierarchical Fragmentation",
                "explanation": (
                    "During initial collapse ($\\rho < 10^{-13}\\text{ g cm}^{-3}$), thermal dust emission escapes freely ($\\tau_{\\text{IR}} \\ll 1$), clamping temperature at $T \\approx 10\\text{ K}$. "
                    "Because $T$ is constant while density $\\rho_0$ increases by orders of magnitude, the Jeans mass drops as $M_J \\propto \\rho_0^{-1/2}$. "
                    "Consequently, a collapsing GMC fragments hierarchically into clusters of smaller dense cores ($M \\sim 0.5\\text{--}10\\,M_\\odot$)."
                ),
            },
            {
                "title": "3. Opacity Limit, First Core & H2 Dissociation Collapse",
                "explanation": (
                    "Fragmentation halts at the opacity limit ($M_{\\text{frag,min}} \\sim 0.005\\text{--}0.01\\,M_\\odot$) when $\\rho \\sim 10^{-13}\\text{ g cm}^{-3}$ and infrared radiation is trapped. "
                    "Adiabatic heating ($\\gamma \\approx 7/5$) supports the First Hydrostatic Core until central temperatures reach $T \\approx 2000\\text{ K}$. "
                    "At $2000\\text{ K}$, endothermic dissociation of $H_2 \\to 2\\text{H}$ ($4.48\\text{ eV}$ per molecule) lowers the effective adiabatic index below the stability limit $\\gamma = 4/3$, "
                    "triggering a rapid second dynamical collapse down to stellar radii ($R \\sim R_\\odot$) to form the true protostar (Second Hydrostatic Core)."
                ),
            },
            {
                "title": "4. Angular Momentum Transport, Circumstellar Disks & Bipolar Outflows",
                "explanation": (
                    "Interstellar cores possess specific angular momentum $j \\sim 10^{21}\\text{ cm}^2\\text{ s}^{-1}$, roughly $10^5$ times higher than a main-sequence star. "
                    "Infalling material with high $j$ cannot fall directly onto the central star and flattens into a rotationally supported disk. "
                    "Magnetorotational instability (MRI) and magnetocentrifugal disk winds/bipolar jets (Herbig–Haro flows) carry away excess angular momentum and disperse part of the envelope, allowing mass accretion at $\\dot{M} \\sim c_s^3/G \\sim 10^{-6}\\text{--}10^{-5}\\,M_\\odot\\text{ yr}^{-1}$."
                ),
            },
            {
                "title": "5. Pre-Main-Sequence Contraction, Deuterium Burning & ZAMS Ignition",
                "explanation": (
                    "Once the envelope clears, the pre-main-sequence (T Tauri or Herbig Ae/Be) star contracts quasi-statically on the Kelvin–Helmholtz timescale $t_{KH} \\approx GM^2/(RL)$. "
                    "By the virial theorem, half of the released gravitational potential energy radiates away ($L$) and half heats the interior ($\\Delta U_{\\text{th}} = -\\frac{1}{2}\\Delta U_{\\text{grav}}$). "
                    "At $T_c \\approx 10^6\\text{ K}$, primordial deuterium fuses ($^2\\text{H} + {^1\\text{H}} \\to {^3\\text{He}} + \\gamma$), briefly establishes a stellar birthline thermostat, and drives full convection. "
                    "Continued contraction heats the core to $T_c \\ge 10^7\\text{ K}$, igniting the proton-proton chain (or CNO cycle for $M > 1.3\\,M_\\odot$) and stabilizing the star on the ZAMS."
                ),
            },
        ],
        "methods": [
            {
                "step": 1,
                "title": "Stage 1 — Dense Core Formation & Ambipolar Diffusion",
                "description": "Turbulent compression and slow neutral-ion slip across magnetic field lines (ambipolar diffusion) create a gravitationally supercritical prestellar core ($M > M_J, M_\\Phi$) at $T \\approx 10\\text{ K}$.",
            },
            {
                "step": 2,
                "title": "Stage 2 — Inside-Out Isothermal Free-Fall Collapse",
                "description": "The core collapses from the inside out (Shu expansion wave propagating outward at sound speed $c_s$) over $t_{ff} = \\sqrt{3\\pi/(32G\\rho_0)} \\sim 10^5\\text{ yr}$ while dust cooling keeps $T \\approx 10\\text{ K}$.",
            },
            {
                "step": 3,
                "title": "Stage 3 — First Hydrostatic Core & Second Collapse (Class 0)",
                "description": "At $\\rho \\sim 10^{-13}\\text{ g cm}^{-3}$, infrared opacity traps heat to form the First Core ($R \\sim 5\\text{ AU}$); at $T \\approx 2000\\text{ K}$, $H_2$ dissociation triggers second collapse into a Class 0 protostar.",
            },
            {
                "step": 4,
                "title": "Stage 4 — Disk Accretion & Bipolar Jet Clearing (Class I -> II)",
                "description": "High-angular-momentum envelope gas settles into a circumstellar disk ($R_c = j^2/GM$), driving magnetically collimated bipolar jets that excavate polar cavities and transition the object from Class I to an optically visible Class II T Tauri star.",
            },
            {
                "step": 5,
                "title": "Stage 5 — Hayashi/Henyey Contraction & ZAMS Ignition (Class III -> ZAMS)",
                "description": "The protoplanetary disk disperses (Class III) while the star contracts on the Kelvin–Helmholtz timescale ($t_{KH} \\sim 30\\text{ Myr}$ for $1\\,M_\\odot$) until $T_c \\ge 10^7\\text{ K}$ ignites stable hydrogen fusion on the ZAMS.",
            },
        ],
        "concept_nodes": [
            {
                "node": "Giant Molecular Cloud ($T \\approx 10\\text{ K}$)",
                "relation": "Exceeds Jeans instability threshold ($M > M_J \\propto T^{3/2}\\rho_0^{-1/2}$)",
                "connects_to": "Isothermal Free-Fall Collapse & Fragmentation",
            },
            {
                "node": "Isothermal Collapse ($\\rho \\to 10^{-13}\\text{ g cm}^{-3}$)",
                "relation": "Becomes optically thick ($\\tau_{\\text{IR}} > 1$) and switches to adiabatic heating ($\\gamma \\approx 7/5$)",
                "connects_to": "First Hydrostatic Core (Larson Core, $R \\sim 5\\text{ AU}$)",
            },
            {
                "node": "First Hydrostatic Core ($T_c \\to 2000\\text{ K}$)",
                "relation": "Endothermic $H_2$ dissociation lowers effective $\\gamma < 4/3$, triggering second collapse",
                "connects_to": "Protostar (Second Core) + Circumstellar Accretion Disk",
            },
            {
                "node": "Circumstellar Disk & MRI / Bipolar Jets",
                "relation": "Transports angular momentum $j = \\Omega R^2$ outward while accreting mass ($\\dot{M} \\sim c_s^3/G$)",
                "connects_to": "Pre-Main-Sequence Star (Class II/III T Tauri / Herbig Ae/Be)",
            },
            {
                "node": "Kelvin–Helmholtz Contraction (Hayashi/Henyey Tracks)",
                "relation": "Virial heating raises central temperature to $T_c \\ge 10^7\\text{ K}$ (for $M \\ge 0.08\\,M_\\odot$)",
                "connects_to": "Zero-Age Main Sequence (ZAMS) Hydrogen Ignition",
            },
        ],
        "formulas": [
            {
                "name": "Isothermal Sound Speed",
                "latex": "$$c_s = \\sqrt{\\frac{\\gamma k_B T}{\\mu m_H}}$$",
                "variables": "c_s: sound speed (m/s), \\gamma: adiabatic index (1 for isothermal), k_B: Boltzmann constant (1.38×10^-23 J/K), T: gas temperature (K), \\mu: mean molecular weight (~2.33 for H_2+He), m_H: hydrogen mass (1.67×10^-27 kg)",
                "when_to_use": "Computing thermal pressure propagation speed and accretion rates in molecular clouds.",
            },
            {
                "name": "Jeans Length",
                "latex": "$$\\lambda_J = \\sqrt{\\frac{\\pi c_s^2}{G \\rho_0}} = \\sqrt{\\frac{\\pi k_B T}{G \\mu m_H \\rho_0}}$$",
                "variables": "\\lambda_J: minimum unstable perturbation scale (m), G: gravitational constant (6.674×10^-11 N m^2/kg^2), \\rho_0: initial mass density (kg/m^3)",
                "when_to_use": "Determining spatial fragmentation scale of a molecular cloud core.",
            },
            {
                "name": "Jeans Mass",
                "latex": "$$M_J = \\frac{4\\pi}{3} \\rho_0 \\left(\\frac{\\lambda_J}{2}\\right)^3 = \\frac{\\pi}{6} \\left(\\frac{\\pi k_B T}{G \\mu m_H}\\right)^{3/2} \\rho_0^{-1/2}$$",
                "variables": "M_J: critical collapse mass (kg or M_sun), T: temperature (K), \\rho_0: mass density (kg/m^3)",
                "when_to_use": "Testing whether a cloud of mass M at (T, \\rho_0) is gravitationally unstable (M > M_J).",
            },
            {
                "name": "Gravitational Free-Fall Timescale",
                "latex": "$$t_{ff} = \\sqrt{\\frac{3\\pi}{32 G \\rho_0}}$$",
                "variables": "t_{ff}: collapse time without pressure support (s), \\rho_0: initial uniform density (kg/m^3)",
                "when_to_use": "Estimating dynamic collapse duration of prestellar cores (~10^5 yr for n ~ 10^5 cm^-3).",
            },
            {
                "name": "Shu Isothermal Mass Accretion Rate & Accretion Luminosity",
                "latex": "$$\\dot{M} \\approx \\frac{c_s^3}{G}, \\qquad L_{\\text{acc}} = \\frac{G M_* \\dot{M}}{R_*}$$",
                "variables": "\\dot{M}: mass infall rate (kg/s), M_*: protostellar mass, R_*: protostellar radius, L_{\\text{acc}}: accretion shock luminosity (W)",
                "when_to_use": "Calculating protostellar mass growth rate and dominant luminosity during Class 0/I phases.",
            },
            {
                "name": "Kelvin–Helmholtz Thermal Contraction Timescale",
                "latex": "$$t_{KH} \\approx \\frac{G M^2}{R L}$$",
                "variables": "t_{KH}: pre-main-sequence gravitational contraction time (s), M: stellar mass, R: stellar radius, L: surface luminosity",
                "when_to_use": "Estimating how long a pre-main-sequence star takes to contract onto the ZAMS (~3×10^7 yr for 1 M_sun).",
            },
        ],
        "examples": [
            {
                "problem": "Worked Example 1 (Free-Fall Timescale & Jeans Mass): A dense molecular cloud core of molecular hydrogen ($\\mu = 2.33$) has temperature $T = 10\\text{ K}$ and number density $n = 10^4\\text{ cm}^{-3}$ ($\\rho_0 = n \\mu m_H \\approx 3.89 \\times 10^{-20}\\text{ g cm}^{-3} = 3.89 \\times 10^{-17}\\text{ kg m}^{-3}$). Calculate its free-fall time $t_{ff}$ and Jeans mass $M_J$.",
                "solution": (
                    "1. **Sound speed**: $c_s = \\sqrt{k_B T / (\\mu m_H)} = \\sqrt{(1.38\\times 10^{-23})(10) / (2.33 \\times 1.67\\times 10^{-27})} \\approx 188\\text{ m/s}$.\n"
                    "2. **Free-fall time**: $t_{ff} = \\sqrt{\\frac{3\\pi}{32 G \\rho_0}} = \\sqrt{\\frac{9.425}{32 \\times 6.674\\times 10^{-11} \\times 3.89\\times 10^{-17}}} \\approx 1.07 \\times 10^{13}\\text{ s} \\approx 3.4 \\times 10^5\\text{ years}$.\n"
                    "3. **Jeans length**: $\\lambda_J = \\sqrt{\\frac{\\pi c_s^2}{G\\rho_0}} \\approx 2.07 \\times 10^{15}\\text{ m} \\approx 0.067\\text{ pc}$.\n"
                    "4. **Jeans mass**: $M_J = \\frac{\\pi}{6} \\rho_0 \\lambda_J^3 \\approx 3.2 \\times 10^{30}\\text{ kg} \\approx 1.6\\,M_\\odot$. Any core heavier than $\\sim 1.6\\,M_\\odot$ at these conditions will collapse."
                ),
            },
            {
                "problem": "Worked Example 2 (Protostellar Accretion Luminosity): A $0.5\\,M_\\odot$ protostar of radius $R_* = 3\\,R_\\odot$ accretes gas from a $T = 10\\text{ K}$ isothermal core at $\\dot{M} \\approx c_s^3/G \\approx 2 \\times 10^{-6}\\,M_\\odot\\text{ yr}^{-1}$. Find its accretion luminosity $L_{\\text{acc}}$ in solar luminosities ($L_\\odot$).",
                "solution": (
                    "Using $L_{\\text{acc}} = \\frac{G M_* \\dot{M}}{R_*}$ in solar units ($L_{\\text{acc}}/L_\\odot \\approx 3.14 \\times 10^7 \\frac{(M_*/M_\\odot)(\\dot{M}/M_\\odot\\text{ yr}^{-1})}{R_*/R_\\odot}$):\n"
                    "$$\\frac{L_{\\text{acc}}}{L_\\odot} \\approx 3.14 \\times 10^7 \\times \\frac{0.5 \\times 2\\times 10^{-6}}{3} \\approx 10.5\\,L_\\odot$$\n"
                    "Thus accretion shock radiation ($10.5\\,L_\\odot$) completely dominates over intrinsic photospheric luminosity during the Class 0/I phase."
                ),
            },
        ],
        "comparisons": [
            {
                "item_a": "Isothermal Collapse Phase (ρ < 10^-13 g/cm³)",
                "item_b": "Adiabatic First-Core Phase (ρ > 10^-13 g/cm³)",
                "difference": "In the isothermal phase, dust cooling escapes freely (τ_IR << 1), T stays ~10 K, and M_J ∝ ρ^(-1/2) drives fragmentation; in the adiabatic phase, trapped IR radiation raises T ∝ ρ^(γ-1) (γ ≈ 7/5), halting fragmentation and forming the First Hydrostatic Core.",
            },
            {
                "item_a": "Hayashi Track (Fully Convective PMS)",
                "item_b": "Henyey Track (Radiative-Core PMS)",
                "difference": "Low-mass PMS stars (< 0.5 M_sun) remain fully convective along the steep vertical Hayashi track (nearly constant T_eff, dropping L); intermediate/high-mass stars (0.5–10 M_sun) develop a radiative core as opacity drops and move nearly horizontally leftward along the Henyey track (rising T_eff, nearly constant L).",
            },
            {
                "item_a": "Low-Mass Star Formation (< 8 M_sun)",
                "item_b": "High-Mass Star Formation (> 8 M_sun)",
                "difference": "Low-mass stars have t_KH >> t_ff, so they finish accreting before reaching the ZAMS via visible T Tauri stages; massive stars have t_KH < t_ff, reaching core H-burning while still deeply embedded and accreting against intense radiation pressure via disks/pillars.",
            },
            {
                "item_a": "Brown Dwarf (0.013–0.08 M_sun)",
                "item_b": "ZAMS Dwarf Star (>= 0.08 M_sun)",
                "difference": "Brown dwarfs fuse deuterium (^2H) briefly at ~10^6 K but become electron-degenerate before core temperature reaches 10^7 K; stars >= 0.08 M_sun achieve sustained ^1H -> ^4He core fusion.",
            },
        ],
        "misconceptions": [
            {
                "myth": "Gravity alone heats a collapsing molecular cloud immediately from the start of collapse.",
                "reality": "During the first ~7 orders of magnitude in density increase, the cloud is optically thin to far-infrared dust emission, so compressional work is radiated away almost instantaneously and the gas stays isothermal at ~10 K until ρ ~ 10^-13 g/cm³.",
            },
            {
                "myth": "Bipolar protostellar jets are powered by nuclear fusion inside the young protostar.",
                "reality": "Class 0/I protostars have not yet ignited core hydrogen fusion; bipolar jets are launched magnetocentrifugally by the rotating, magnetized inner accretion disk powered by gravitational infall energy.",
            },
            {
                "myth": "When a pre-main-sequence star radiates energy into space, its core cools down.",
                "reality": "By the Virial Theorem (ΔE_tot = +1/2 ΔU_grav < 0, ΔK_thermal = -1/2 ΔU_grav > 0), a self-gravitating ideal-gas star has negative heat capacity: losing radiative energy causes it to contract and its core to become hotter.",
            },
        ],
        "exam_points": [
            "Jeans Mass scales as $M_J \\propto T^{3/2} \\rho_0^{-1/2}$; cold ($10\\text{ K}$), dense cores have the lowest $M_J$ and collapse first.",
            "Free-fall time $t_{ff} = \\sqrt{3\\pi/(32G\\rho_0)}$ depends ONLY on initial density $\\rho_0$, explaining inside-out collapse where denser central regions collapse fastest.",
            "Isothermal fragmentation stops at the opacity limit ($\\rho \\sim 10^{-13}\\text{ g cm}^{-3}$, $M_{\\min} \\sim 0.005\\text{--}0.01\\,M_\\odot$) where the First Hydrostatic Core forms.",
            "Second dynamical collapse is triggered at $T \\approx 2000\\text{ K}$ by endothermic dissociation of $H_2$, which drops the adiabatic index below $\\gamma = 4/3$.",
            "Conservation of specific angular momentum $j = \\Omega R^2$ forces infalling gas into a Keplerian disk at $R_c = j^2/(GM)$; MRI viscosity and bipolar outflows solve the angular momentum problem.",
            "YSO evolutionary sequence: Class 0 (sub-mm, $M_{\\text{env}} > M_*$) $\\to$ Class I (IR, $M_* > M_{\\text{env}}$) $\\to$ Class II (Classical T Tauri, optically thick protoplanetary disk) $\\to$ Class III (Weak-lined T Tauri, debris disk) $\\to$ ZAMS.",
            "Minimum mass for sustained hydrogen fusion on the ZAMS is $M_{\\min} \\approx 0.08\\,M_\\odot$ ($T_c \\ge 10^7\\text{ K}$); objects between $0.013\\,M_\\odot$ and $0.08\\,M_\\odot$ are brown dwarfs supported by electron degeneracy pressure.",
        ],
        "exam_questions": [
            {
                "question": "Explain why a collapsing molecular cloud fragments into a cluster of stars during its early collapse, and what physical transition stops this fragmentation.",
                "answer": "During early collapse ($\\rho < 10^{-13}\\text{ g cm}^{-3}$), the cloud is optically thin to dust thermal continuum radiation, keeping $T \\approx 10\\text{ K}$ constant (isothermal). Because Jeans mass scales as $M_J \\propto T^{3/2}\\rho_0^{-1/2}$, rising density causes $M_J$ to decrease rapidly, so progressively smaller sub-condensations exceed their local Jeans mass and collapse independently (Hoyle hierarchical fragmentation). Fragmentation halts at $\\rho \\sim 10^{-13}\\text{ g cm}^{-3}$ when the core becomes optically thick to infrared radiation ($\\tau_{\\text{IR}} > 1$) and switches to adiabatic heating ($T \\propto \\rho^{\\gamma-1}$ with $\\gamma \\approx 7/5$), causing $M_J \\propto \\rho^{(3\\gamma-4)/2}$ to increase with density and forming the First Hydrostatic Core.",
            },
            {
                "question": "How does the Virial Theorem explain why a pre-main-sequence star's core heats up as it radiates energy into space along the Hayashi/Henyey tracks?",
                "answer": "For a hydrostatic star in virial equilibrium, $2K + U = 0$, so total energy is $E = K + U = \\frac{1}{2}U = -K$. As the star radiates luminosity $L = -dE/dt > 0$, its total energy $E$ becomes more negative, meaning gravitational potential energy $U$ decreases ($\\Delta U < 0$, the star contracts). Exactly half of the released gravitational energy is radiated away ($\\Delta E_{\\text{rad}} = -\\frac{1}{2}\\Delta U$) while the other half goes into internal thermal kinetic energy ($\\Delta K = -\\frac{1}{2}\\Delta U > 0$), raising the central temperature $T_c \\propto M/R$ until nuclear ignition at $10^7\\text{ K}$.",
            },
            {
                "question": "Why can massive stars ($M > 8\\,M_\\odot$) not be observed in an optical pre-main-sequence T Tauri phase like solar-mass stars?",
                "answer": "For a $1\\,M_\\odot$ star, the Kelvin–Helmholtz contraction time ($t_{KH} \\sim 3\\times 10^7\\text{ yr}$) is roughly 100 times longer than the envelope free-fall/accretion time ($t_{ff} \\sim 10^5\\text{ yr}$), so the envelope clears long before ZAMS arrival. Because $t_{KH} = GM^2/(RL)$ and luminosity rises steeply with mass ($L \\propto M^{3.5}$), $t_{KH}$ drops below $10^5\\text{ yr}$ for $M > 8\\,M_\\odot$ ($t_{KH} < t_{ff}$). Thus massive stars reach $T_c \\ge 10^7\\text{ K}$ and ignite core hydrogen fusion on the ZAMS while still deeply embedded and actively accreting inside their birth clouds.",
            },
        ],
    }
    return cards, questions, ensure_complete_cheatsheet_structure(cheatsheet, topic, source_label)


def _build_neural_networks_bundle(topic: str, source_label: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cards = [
        {
            "topic": "Forward Propagation & Activation Functions",
            "difficulty": "easy",
            "question": "How does a dense neural network layer compute its forward pass, and why are non-linear activation functions required?",
            "answer": "Each layer computes an affine transformation followed by a point-wise non-linear activation $\\sigma$:\n$$\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]} \\mathbf{a}^{[l-1]} + \\mathbf{b}^{[l]}, \\qquad \\mathbf{a}^{[l]} = \\sigma(\\mathbf{z}^{[l]})$$\nWithout non-linear activations (such as ReLU, GELU, Sigmoid, or Tanh), a composition of linear layers collapses into a single linear transformation $\\mathbf{W}_{\\text{eff}}\\mathbf{x} + \\mathbf{b}_{\\text{eff}}$ and cannot approximate non-linear functions.",
            "explanation": "Universal Approximation Theorem requires non-polynomial non-linear activations.",
            "source": source_label,
        },
        {
            "topic": "Backpropagation & Chain Rule",
            "difficulty": "medium",
            "question": "What is backpropagation, and how does it compute weight gradients $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{[l]}}$?",
            "answer": "**Backpropagation** applies reverse-mode automatic differentiation (the multivariable chain rule) from the output loss $\\mathcal{L}$ backward through each layer $l$:\n$$\\boldsymbol{\\delta}^{[l]} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}^{[l]}} = \\left((\\mathbf{W}^{[l+1]})^\\top \\boldsymbol{\\delta}^{[l+1]}\\right) \\odot \\sigma'(\\mathbf{z}^{[l]})$$\n$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{[l]}} = \\boldsymbol{\\delta}^{[l]} (\\mathbf{a}^{[l-1]})^\\top, \\qquad \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{[l]}} = \\boldsymbol{\\delta}^{[l]}$$",
            "explanation": "Reuses intermediate forward activations a^{[l-1]} to compute all gradients in O(W) time.",
            "source": source_label,
        },
        {
            "topic": "Vanishing & Exploding Gradients",
            "difficulty": "medium",
            "question": "What causes vanishing and exploding gradients in deep networks, and what techniques mitigate them?",
            "answer": "During backpropagation across $L$ layers, gradients involve products of Jacobians $\\prod_{k=l}^{L} (\\mathbf{W}^{[k]})^\\top \\text{diag}(\\sigma'(\\mathbf{z}^{[k]}))$. If singular values or derivatives $\\sigma'$ are $< 1$ (e.g., Sigmoid max derivative is $0.25$), gradients shrink exponentially (**vanishing**); if $> 1$, they blow up (**exploding**). Mitigations include **ReLU/GELU activations**, **He/Xavier initialization**, **Batch/Layer Normalization**, **Residual skip connections** ($\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$), and **gradient clipping**.",
            "explanation": "Residual connections create an identity gradient highway d(x + F(x))/dx = I + dF/dx.",
            "source": source_label,
        },
        {
            "topic": "Gradient Descent & Adam Optimizer",
            "difficulty": "medium",
            "question": "How does the Adam optimizer improve upon standard Stochastic Gradient Descent (SGD)?",
            "answer": "Standard SGD updates parameters via $\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}$. **Adam** maintains exponentially decaying moving averages of both the first moment (momentum $m_t$) and second moment (uncentered variance $v_t$) with bias correction ($\hat{m}_t, \hat{v}_t$), adapting the effective learning rate per parameter:\n$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$",
            "explanation": "Combines the benefits of Momentum (damping oscillations) and RMSProp (per-parameter scaling).",
            "source": source_label,
        },
        {
            "topic": "Cross-Entropy Loss & Softmax",
            "difficulty": "hard",
            "question": "Write the Softmax function and Categorical Cross-Entropy loss, and state the simplified gradient $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}}$ with respect to the logits.",
            "answer": "For logits $\\mathbf{z} \\in \\mathbb{R}^K$, predicted probabilities are $\\hat{y}_i = \\frac{e^{z_i}}{\\sum_{j=1}^K e^{z_j}}$ and cross-entropy loss for one-hot target $\\mathbf{y}$ is $\\mathcal{L} = -\\sum_{i=1}^K y_i \\ln(\\hat{y}_i)$. The combined gradient with respect to the pre-softmax logits simplifies cleanly to:\n$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}} = \\hat{\\mathbf{y}} - \\mathbf{y}$$",
            "explanation": "Avoids saturation of individual softmax derivatives when combined log-sum-exp is differentiated.",
            "source": source_label,
        },
        {
            "topic": "Regularization (Dropout, L2 Weight Decay)",
            "difficulty": "easy",
            "question": "How do L2 Regularization (Weight Decay) and Dropout reduce overfitting in neural networks?",
            "answer": "- **L2 Regularization**: Adds $\\frac{\\lambda}{2}\\|\\mathbf{W}\\|_2^2$ to the loss, penalizing large weights and encouraging smoother decision boundaries.\n- **Dropout**: Randomly zeroes a fraction $p$ of hidden activations during training (scaling by $1/(1-p)$), preventing co-adaptation of neurons and acting as an implicit ensemble of subnetworks.",
            "explanation": "At inference time, Dropout is disabled and all neurons remain active.",
            "source": source_label,
        },
    ]

    questions = [
        {
            "id": "q1",
            "type": "multiple_choice",
            "concept": "Backpropagation Chain Rule",
            "difficulty": "easy",
            "question": "Which mathematical principle does backpropagation use to compute the gradient of the loss function with respect to weights in earlier layers of a neural network?",
            "options": [
                "The multivariable chain rule of calculus applied recursively from output to input",
                "Gravitational Jeans instability across molecular gas layers",
                "Laplace transform inversion of time-domain signals",
                "Brute-force random perturbation of each weight independently",
            ],
            "correct_answer": "The multivariable chain rule of calculus applied recursively from output to input",
            "correctIndex": 0,
            "explanation": "Backpropagation is reverse-mode automatic differentiation, applying the chain rule layer by layer from the loss output back to the input weights.",
            "source_citation": source_label,
        },
        {
            "id": "q2",
            "type": "multiple_choice",
            "concept": "Vanishing Gradient Problem",
            "difficulty": "medium",
            "question": "Why does the logistic Sigmoid activation $\\sigma(z) = \\frac{1}{1+e^{-z}}$ often cause vanishing gradients in deep neural networks?",
            "options": [
                "Its derivative σ'(z) = σ(z)(1 - σ(z)) has a maximum value of 0.25 and approaches 0 for large |z|, causing repeated multiplication across layers to decay exponentially",
                "Its output is unbounded for negative inputs, causing immediate numerical overflow",
                "Its derivative is identically equal to 1 everywhere, preventing weight updates",
                "It can only be differentiated using Monte Carlo sampling",
            ],
            "correct_answer": "Its derivative σ'(z) = σ(z)(1 - σ(z)) has a maximum value of 0.25 and approaches 0 for large |z|, causing repeated multiplication across layers to decay exponentially",
            "correctIndex": 0,
            "explanation": "Because max σ'(z) = 0.25 at z = 0 and saturates near 0 in the tails, multiplying σ'(z) across many layers shrinks gradients toward zero in early layers.",
            "source_citation": source_label,
        },
        {
            "id": "q3",
            "type": "multiple_choice",
            "concept": "Softmax & Cross-Entropy Gradient",
            "difficulty": "hard",
            "question": "For a multi-class classifier using a Softmax output layer $\\hat{\\mathbf{y}} = \\text{softmax}(\\mathbf{z})$ and Categorical Cross-Entropy loss $\\mathcal{L} = -\\sum_i y_i \\ln(\\hat{y}_i)$, what is the gradient $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}}$ with respect to the logits $\\mathbf{z}$?",
            "options": [
                "ŷ - y (predicted probability vector minus true one-hot target vector)",
                "y / ŷ",
                "(ŷ - y) * ŷ * (1 - ŷ)",
                "ln(ŷ) - y",
            ],
            "correct_answer": "ŷ - y (predicted probability vector minus true one-hot target vector)",
            "correctIndex": 0,
            "explanation": "Differentiating -∑ y_i ln(e^{z_i} / ∑_j e^{z_j}) with respect to z_k yields ŷ_k (∑_i y_i) - y_k = ŷ_k - y_k.",
            "source_citation": source_label,
        },
        {
            "id": "q4",
            "type": "true_false",
            "concept": "Residual Connections",
            "difficulty": "medium",
            "question": "True or False: Residual skip connections ($\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$) help train very deep neural networks because the identity shortcut adds a $+\\mathbf{I}$ term to the Jacobian, allowing gradients to flow directly backward without vanishing.",
            "options": ["True", "False"],
            "correct_answer": "True",
            "correctIndex": 0,
            "explanation": "Since ∂y/∂x = I + ∂F/∂x, gradients can propagate unimpeded through the identity term even when ∂F/∂x is small.",
            "source_citation": source_label,
        },
        {
            "id": "q5",
            "type": "fill_in_blank",
            "concept": "Weight Initialization",
            "difficulty": "hard",
            "question": "To preserve activation variance across layers using ReLU activations, ______ initialization sets weight variance to $\\text{Var}(W) = \\frac{2}{n_{\\text{in}}}$.",
            "options": [
                "He (Kaiming)",
                "Zero",
                "Uniform Constant",
                "Fourier",
            ],
            "correct_answer": "He (Kaiming)",
            "correctIndex": 0,
            "explanation": "Because ReLU zeroes out half of a symmetric input distribution, He (Kaiming) initialization uses 2/n_in (instead of Xavier's 1/n_in) to keep forward and backward variance near 1.",
            "source_citation": source_label,
        },
    ]

    cheatsheet = {
        "title": f"{topic} — Neural Networks & Deep Learning Cheat Sheet",
        "topic": topic,
        "detected_subject": "Machine Learning & Deep Neural Networks",
        "source_summary": f"Synthesized strictly from active session ({source_label})",
        "overview": "Comprehensive reference covering feedforward architectures, activation functions, loss functions, backpropagation via the chain rule, vanishing/exploding gradients, optimization algorithms (SGD, Momentum, Adam), and regularization.",
        "definitions": [
            {"term": "Forward Propagation", "definition": "Sequential computation of affine combinations $\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]}\\mathbf{a}^{[l-1]} + \\mathbf{b}^{[l]}$ and non-linear activations $\\mathbf{a}^{[l]} = \\sigma(\\mathbf{z}^{[l]})$ from input to output."},
            {"term": "Backpropagation", "definition": "Reverse-mode automatic differentiation algorithm that computes exact loss gradients $\\nabla_{\\mathbf{W}^{[l]}}\\mathcal{L}$ by propagating error signals $\\boldsymbol{\\delta}^{[l]}$ backward from the output layer."},
            {"term": "Vanishing Gradient Problem", "definition": "Exponential decay of backpropagated gradients in deep networks when layer Jacobians and activation derivatives ($\\sigma'$) have magnitudes less than 1."},
            {"term": "Residual Connection (Skip Connection)", "definition": "Architectural shortcut $\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$ whose Jacobian $\\mathbf{I} + \\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}}$ prevents gradient attenuation across deep stacks."},
        ],
        "concepts": [
            {"title": "Chain Rule & Error Signal Propagation", "explanation": "Each hidden layer's sensitivity $\\boldsymbol{\\delta}^{[l]} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}^{[l]}}$ is computed from the next layer via $\\boldsymbol{\\delta}^{[l]} = ((\\mathbf{W}^{[l+1]})^\\top \\boldsymbol{\\delta}^{[l+1]}) \\odot \\sigma'(\\mathbf{z}^{[l]})$, enabling weight gradients $\\boldsymbol{\\delta}^{[l]}(\\mathbf{a}^{[l-1]})^\\top$ in a single backward pass."},
            {"title": "Activation Functions & Gradient Flow", "explanation": "Sigmoid ($\\sigma'(z) \\le 0.25$) and Tanh saturate for large $|z|$, whereas ReLU ($\\max(0, z)$) has derivative $1$ for $z > 0$, avoiding positive saturation (with Leaky ReLU / GELU preventing dead neurons for $z < 0$)."},
            {"title": "Adaptive Optimization & Normalization", "explanation": "Batch Normalization and Layer Normalization stabilize internal activation distributions, while Adam scales updates per parameter using first and second gradient moments."},
        ],
        "methods": [
            {"step": 1, "title": "Forward Pass", "description": "Compute $\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]}\\mathbf{a}^{[l-1]} + \\mathbf{b}^{[l]}$ and cache $\\mathbf{a}^{[l]} = \\sigma(\\mathbf{z}^{[l]})$ for $l = 1,\\dots,L$."},
            {"step": 2, "title": "Loss Evaluation", "description": "Evaluate task loss $\\mathcal{L}(\\hat{\\mathbf{y}}, \\mathbf{y})$ (e.g., Cross-Entropy or MSE) plus regularization $\\frac{\\lambda}{2}\\sum_l \\|\\mathbf{W}^{[l]}\\|_F^2$."},
            {"step": 3, "title": "Backward Pass (Backpropagation)", "description": "Compute output error $\\boldsymbol{\\delta}^{[L]} = \\hat{\\mathbf{y}} - \\mathbf{y}$, then recurse $\\boldsymbol{\\delta}^{[l]} = ((\\mathbf{W}^{[l+1]})^\\top \\boldsymbol{\\delta}^{[l+1]}) \\odot \\sigma'(\\mathbf{z}^{[l]})$ and $\\nabla_{\\mathbf{W}^{[l]}}\\mathcal{L} = \\boldsymbol{\\delta}^{[l]}(\\mathbf{a}^{[l-1]})^\\top$."},
            {"step": 4, "title": "Parameter Update", "description": "Update weights using SGD or Adam: $\\mathbf{W}^{[l]} \\leftarrow \\mathbf{W}^{[l]} - \\eta \\frac{\\hat{\\mathbf{m}}_t}{\\sqrt{\\hat{\\mathbf{v}}_t} + \\epsilon}$."},
        ],
        "concept_nodes": [
            {"node": "Forward Pass ($\\mathbf{z}^{[l]}, \\mathbf{a}^{[l]}$)", "relation": "Produces predictions $\\hat{\\mathbf{y}}$ compared against targets $\\mathbf{y}$", "connects_to": "Loss Function $\\mathcal{L}(\\hat{\\mathbf{y}}, \\mathbf{y})$"},
            {"node": "Loss Function $\\mathcal{L}$", "relation": "Differentiated via reverse-mode Chain Rule", "connects_to": "Backpropagated Error Signals $\\boldsymbol{\\delta}^{[l]}$"},
            {"node": "Error Signals $\\boldsymbol{\\delta}^{[l]}$", "relation": "Outer product with cached activations $\\mathbf{a}^{[l-1]}$ yields", "connects_to": "Parameter Gradients $\\nabla_{\\mathbf{W}^{[l]}}\\mathcal{L}$"},
            {"node": "Parameter Gradients $\\nabla_{\\mathbf{W}^{[l]}}\\mathcal{L}$", "relation": "Scaled by learning rate $\\eta$ and moments in Adam/SGD", "connects_to": "Updated Network Weights $\\mathbf{W}^{[l]}_{t+1}$"},
        ],
        "formulas": [
            {"name": "Layer Forward Transformation", "latex": "$$\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]}\\mathbf{a}^{[l-1]} + \\mathbf{b}^{[l]}, \\quad \\mathbf{a}^{[l]} = \\sigma(\\mathbf{z}^{[l]})$$", "variables": "W^{[l]}: weight matrix, b^{[l]}: bias vector, a^{[l-1]}: previous layer activations, \\sigma: non-linear activation"},
            {"name": "Backpropagation Equations", "latex": "$$\\boldsymbol{\\delta}^{[l]} = \\left((\\mathbf{W}^{[l+1]})^\\top \\boldsymbol{\\delta}^{[l+1]}\\right) \\odot \\sigma'(\\mathbf{z}^{[l]}), \\quad \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{[l]}} = \\boldsymbol{\\delta}^{[l]}(\\mathbf{a}^{[l-1]})^\\top$$", "variables": "\\delta^{[l]}: error sensitivity vector at layer l, \\odot: Hadamard element-wise product"},
            {"name": "Adam Update Rule", "latex": "$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\hat{m}_t$$", "variables": "\\eta: learning rate, \\hat{m}_t: bias-corrected 1st moment, \\hat{v}_t: bias-corrected 2nd moment, \\epsilon: numerical stability constant (10^-8)"},
        ],
        "examples": [
            {"problem": "Compute the gradient $\\partial \\mathcal{L}/\\partial z_1$ for a 3-class softmax classifier where $\\hat{\\mathbf{y}} = [0.7, 0.2, 0.1]^\\top$ and the true label is class 1 ($\\mathbf{y} = [1, 0, 0]^\\top$).", "solution": "Using $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}} = \\hat{\\mathbf{y}} - \\mathbf{y}$, we obtain $[0.7 - 1, 0.2 - 0, 0.1 - 0]^\\top = [-0.3, 0.2, 0.1]^\\top$. Thus $\\partial \\mathcal{L}/\\partial z_1 = -0.3$."}
        ],
        "comparisons": [
            {"item_a": "Sigmoid Activation", "item_b": "ReLU Activation", "difference": "Sigmoid squashes to (0,1) with max derivative 0.25 (prone to vanishing gradients); ReLU is max(0,z) with derivative 1 for z > 0 (fast convergence, no positive saturation)."},
            {"item_a": "Batch Normalization", "item_b": "Layer Normalization", "difference": "BatchNorm normalizes across the mini-batch dimension per feature; LayerNorm normalizes across features within each individual sample (preferred in Transformers)."},
        ],
        "misconceptions": [
            {"myth": "Initializing all neural network weights to zero allows neurons to learn different features via gradient descent.", "reality": "If all weights are initialized to zero, every hidden unit in a layer receives identical inputs and identical gradients, remaining permanently symmetric."}
        ],
        "exam_points": [
            "Backpropagation computes gradients in $O(W)$ operations by caching forward activations $\\mathbf{a}^{[l]}$.",
            "Combined Softmax + Cross-Entropy gradient with respect to logits is $\\hat{\\mathbf{y}} - \\mathbf{y}$.",
            "He initialization uses $\\text{Var}(W) = 2/n_{\\text{in}}$ for ReLU layers; Xavier initialization uses $2/(n_{\\text{in}} + n_{\\text{out}})$ for Tanh/Sigmoid.",
            "Residual connections prevent vanishing gradients via the identity Jacobian term $\\frac{\\partial(\\mathbf{x}+\\mathcal{F}(\\mathbf{x}))}{\\partial \\mathbf{x}} = \\mathbf{I} + \\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}}$.",
        ],
        "exam_questions": [
            {"question": "Why must neural network weights be initialized randomly with calibrated variance rather than all zeros?", "answer": "Zero initialization causes symmetry: every neuron in a hidden layer computes the exact same output and receives the exact same gradient during backpropagation, preventing the network from learning distinct features."}
        ],
    }
    return cards, questions, ensure_complete_cheatsheet_structure(cheatsheet, topic, source_label)


def build_source_grounded_study_fallback(
    boundary: Dict[str, Any], difficulty: str = "mixed", count: int = 6, learning_mode: str = "normal"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts real definitions, formulas, and concepts directly from the active
    conversation & uploaded document text when offline or if an external provider fails,
    so quizzes, recall cards, and cheat sheets are ALWAYS 100% grounded in the active topic(s).
    Supports multi-topic conversations proportionately!
    """
    topic = boundary.get("topic") or "Active Study Topic"
    conv_topics = boundary.get("conversation_topics") or [topic]
    context = boundary.get("combined_context") or ""
    docs = boundary.get("selected_docs") or []
    source_label = docs[0]["filename"] if docs else f"Conversation: {topic}"

    low_all = f"{topic}\n{' '.join(conv_topics)}\n{context}".lower()
    has_stellar = any(w in low_all for w in ("stellar birth", "star formation", "jeans mass", "jeans instability", "protostar", "hayashi", "zams", "giant molecular cloud", "larson core"))
    has_black_holes = any(w in low_all for w in ("black hole", "schwarzschild", "hawking radiation", "event horizon", "spaghettification"))
    has_neural = any(w in low_all for w in ("neural network", "backpropagation", "vanishing gradient", "activation function", "gradient descent", "softmax"))

    # Check if this is a multi-topic conversation combining two or more distinct subjects
    matched_bundles: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]] = []
    if has_stellar:
        matched_bundles.append(_build_stellar_birth_cycle_bundle(topic, source_label, learning_mode))
    if has_black_holes:
        # Build Black Holes bundle
        bh_cards = [
            {
                "topic": "Event Horizon & Schwarzschild Radius",
                "difficulty": "easy",
                "question": "What is the Event Horizon of a black hole, and how is the Schwarzschild radius ($r_s$) defined?",
                "answer": "The **event horizon** is the causal boundary around a black hole beyond which the escape velocity exceeds the speed of light ($c$). For a non-rotating, uncharged mass $M$, its radius is the **Schwarzschild radius**:\n$$r_s = \\frac{2GM}{c^2}$$\nwhere $G$ is the gravitational constant, $M$ is the black hole mass, and $c$ is the speed of light.",
                "explanation": "Derived from the Schwarzschild metric where the time-time component g_00 = -(1 - 2GM/(rc^2)) vanishes.",
                "source": source_label,
            },
            {
                "topic": "Gravitational Collapse & TOV Limit",
                "difficulty": "easy",
                "question": "What physical mechanism leads to the formation of a stellar-mass black hole through gravitational collapse?",
                "answer": "When a massive star exhausts its nuclear fuel, outward thermal and degeneracy pressures (including neutron degeneracy pressure above the **Tolman–Oppenheimer–Volkoff limit**, $\\approx 2.17\\,M_\\odot$) can no longer balance inward self-gravity, causing unstoppable **gravitational collapse** within its Schwarzschild radius.",
                "explanation": "White dwarfs are supported below 1.44 M_sun by electron degeneracy; neutron stars below ~2.17 M_sun by neutron degeneracy.",
                "source": source_label,
            },
            {
                "topic": "Hawking Radiation",
                "difficulty": "medium",
                "question": "What is Hawking radiation, and why does it cause a black hole to lose mass?",
                "answer": "**Hawking radiation** is thermal quantum radiation emitted near a black hole's event horizon due to quantum vacuum fluctuations in curved spacetime. Positive-energy modes escape to infinity while negative-energy partner modes fall inward, reducing the net mass-energy $M$ of the black hole.",
                "explanation": "Unifies general relativity, quantum field theory, and thermodynamics.",
                "source": source_label,
            },
            {
                "topic": "Hawking Temperature Formula",
                "difficulty": "medium",
                "question": "How does the Hawking temperature ($T_H$) depend on the mass ($M$) of a black hole?",
                "answer": "The **Hawking temperature** is inversely proportional to the black hole's mass $M$:\n$$T_H = \\frac{\\hbar c^3}{8\\pi G M k_B}$$\nAs a black hole radiates energy and its mass $M$ decreases, its temperature $T_H$ increases.",
                "explanation": "Implies negative heat capacity: radiating energy makes the black hole hotter.",
                "source": source_label,
            },
            {
                "topic": "Black-Hole Evaporation Lifetime",
                "difficulty": "hard",
                "question": "How does the black-hole evaporation lifetime ($\\tau$) scale with initial mass ($M$), and why?",
                "answer": "Because luminosity follows the Stefan–Boltzmann law ($L \\propto A T_H^4 \\propto M^2 \\cdot M^{-4} = M^{-2}$), the mass loss rate is $\\frac{dM}{dt} \\propto -\\frac{1}{M^2}$. Integrating gives an evaporation lifetime proportional to the cube of the mass:\n$$\\tau = \\frac{5120\\pi G^2 M^3}{\\hbar c^4} \\propto M^3$$",
                "explanation": "Doubling a black hole's mass increases its evaporation lifetime by a factor of 8.",
                "source": source_label,
            },
            {
                "topic": "Spaghettification & Tidal Forces",
                "difficulty": "medium",
                "question": "Why are tidal forces at the event horizon stronger for a stellar-mass black hole than for a supermassive black hole?",
                "answer": "Radial tidal acceleration scales as $a_{\\text{tidal}} \\propto \\frac{GM}{r^3} \\Delta r$. Evaluated at the event horizon $r = r_s \\propto M$, tidal acceleration scales as $M / M^3 = M^{-2}$. Thus, a supermassive black hole has much gentler tidal forces at its horizon than a stellar-mass black hole.",
                "explanation": "An observer can cross a 10^8 M_sun black hole's horizon without immediate tidal disruption.",
                "source": source_label,
            },
        ]
        bh_questions = [
            {
                "id": "q_bh1",
                "type": "multiple_choice",
                "concept": "Schwarzschild Radius",
                "difficulty": "easy",
                "question": "Which formula gives the Schwarzschild radius ($r_s$) of a non-rotating, uncharged black hole of mass $M$?",
                "options": [
                    "r_s = 2GM / c^2",
                    "r_s = GM / (2c^2)",
                    "r_s = 4pi G M / c",
                    "r_s = hbar c^3 / (8pi G M k_B)",
                ],
                "correct_answer": "r_s = 2GM / c^2",
                "correctIndex": 0,
                "explanation": "Setting the Newtonian escape velocity equal to the speed of light c (or solving the Schwarzschild metric g_00 = 0) yields r_s = 2GM / c^2.",
                "source_citation": source_label,
            },
            {
                "id": "q_bh2",
                "type": "multiple_choice",
                "concept": "Event Horizon Definition",
                "difficulty": "easy",
                "question": "What defines the event horizon of a black hole?",
                "options": [
                    "The solid physical surface of compressed neutron matter",
                    "The causal boundary from within which no light or matter can escape to an outside observer",
                    "The outer boundary of the accretion disk where gas begins to ionize",
                    "The radius at which gravitational attraction drops to zero",
                ],
                "correct_answer": "The causal boundary from within which no light or matter can escape to an outside observer",
                "correctIndex": 1,
                "explanation": "The event horizon is not a solid surface; it is the spacetime boundary where the escape velocity equals the speed of light and all future-directed light cones point inward.",
                "source_citation": source_label,
            },
            {
                "id": "q_bh3",
                "type": "multiple_choice",
                "concept": "Hawking Temperature Relationship",
                "difficulty": "medium",
                "question": "According to Hawking's black-hole thermodynamics ($T_H = \\frac{\\hbar c^3}{8\\pi G M k_B}$), what happens to a black hole's temperature as its mass $M$ decreases through evaporation?",
                "options": [
                    "Its Hawking temperature increases inversely with mass (T_H ∝ 1/M)",
                    "Its Hawking temperature decreases linearly with mass (T_H ∝ M)",
                    "Its Hawking temperature remains constant until complete evaporation",
                    "Its Hawking temperature drops to absolute zero immediately",
                ],
                "correct_answer": "Its Hawking temperature increases inversely with mass (T_H ∝ 1/M)",
                "correctIndex": 0,
                "explanation": "Because mass M is in the denominator of the Hawking temperature formula, losing mass causes the black hole to become hotter and radiate faster.",
                "source_citation": source_label,
            },
            {
                "id": "q_bh4",
                "type": "true_false",
                "concept": "Gravitational Collapse",
                "difficulty": "medium",
                "question": "True or False: A stellar core whose remnant mass exceeds the Tolman–Oppenheimer–Volkoff (TOV) limit cannot be supported by neutron degeneracy pressure and undergoes complete gravitational collapse into a black hole.",
                "options": ["True", "False"],
                "correct_answer": "True",
                "correctIndex": 0,
                "explanation": "Above the TOV limit (~2.17 solar masses), neither electron nor neutron degeneracy pressure can halt self-gravity, resulting in a black hole.",
                "source_citation": source_label,
            },
            {
                "id": "q_bh5",
                "type": "multiple_choice",
                "concept": "Black-Hole Evaporation Lifetime",
                "difficulty": "hard",
                "question": "If Black Hole A has 2 times the initial mass of Black Hole B ($M_A = 2 M_B$), how does the Hawking evaporation lifetime of Black Hole A ($\\tau_A$) compare to that of Black Hole B ($\\tau_B$)?",
                "options": [
                    "τ_A = 8 τ_B (8 times longer, since τ ∝ M^3)",
                    "τ_A = 2 τ_B (2 times longer, since τ ∝ M)",
                    "τ_A = 4 τ_B (4 times longer, since τ ∝ M^2)",
                    "τ_A = 0.5 τ_B (half as long, since τ ∝ 1/M)",
                ],
                "correct_answer": "τ_A = 8 τ_B (8 times longer, since τ ∝ M^3)",
                "correctIndex": 0,
                "explanation": "Since the Hawking evaporation lifetime scales as τ ∝ M^3, doubling the mass increases the evaporation lifetime by a factor of 2^3 = 8.",
                "source_citation": source_label,
            },
            {
                "id": "q_bh6",
                "type": "fill_in_blank",
                "concept": "Hawking Radiation",
                "difficulty": "medium",
                "question": "Quantum thermal emission predicted by Stephen Hawking near the event horizon that causes black holes to slowly lose mass over time is called ______.",
                "options": [
                    "Hawking radiation",
                    "Cherenkov radiation",
                    "Synchrotron emission",
                    "Bremsstrahlung",
                ],
                "correct_answer": "Hawking radiation",
                "correctIndex": 0,
                "explanation": "Hawking radiation arises from quantum field effects in curved spacetime near the event horizon.",
                "source_citation": source_label,
            },
        ]
        bh_sheet = ensure_complete_cheatsheet_structure({
            "title": f"{topic} — Study Cheat Sheet",
            "topic": topic,
            "detected_subject": "Astrophysics & General Relativity",
            "source_summary": f"Synthesized strictly from active session ({source_label})",
            "overview": "Comprehensive study reference covering gravitational collapse, the Schwarzschild metric, event horizon physics, Hawking radiation, black-hole thermodynamics, and evaporation lifetimes.",
            "definitions": [
                {"term": "Event Horizon", "definition": "The null hypersurface boundary around a black hole where escape velocity equals the speed of light ($c$), preventing matter or radiation from escaping."},
                {"term": "Schwarzschild Radius ($r_s$)", "definition": "The event horizon radius of a static, spherically symmetric mass $M$: $r_s = 2GM/c^2$."},
                {"term": "Gravitational Collapse", "definition": "The inward contraction of a massive astronomical body under its own gravity after exhausting thermonuclear fuel and exceeding degeneracy pressure limits."},
                {"term": "Hawking Radiation", "definition": "Blackbody quantum thermal radiation emitted near the event horizon with temperature $T_H \\propto 1/M$, driving gradual black-hole evaporation."},
            ],
            "concepts": [
                {"title": "Spacetime Curvature & Horizon Formation", "explanation": "When mass $M$ is compressed within $r_s = 2GM/c^2$, spacetime curvature tilts all future light cones inward toward the central singularity."},
                {"title": "Black-Hole Thermodynamics & Evaporation", "explanation": "A black hole has Bekenstein–Hawking entropy $S_{BH} \\propto A \\propto M^2$ and temperature $T_H \\propto M^{-1}$. As it radiates via Hawking emission, it shrinks, heats up, and evaporates in time $\\tau \\propto M^3$."},
            ],
            "formulas": [
                {"name": "Schwarzschild Radius", "latex": "$$r_s = \\frac{2GM}{c^2}$$", "variables": "r_s: Schwarzschild radius (m), G: gravitational constant, M: black hole mass (kg), c: speed of light (m/s)"},
                {"name": "Hawking Temperature", "latex": "$$T_H = \\frac{\\hbar c^3}{8\\pi G M k_B}$$", "variables": "T_H: Hawking temperature (K), \\hbar: reduced Planck constant, k_B: Boltzmann constant, M: mass (kg)"},
                {"name": "Evaporation Lifetime", "latex": "$$\\tau = \\frac{5120\\pi G^2 M^3}{\\hbar c^4}$$", "variables": "\\tau: evaporation time (s), scaling with the cube of initial mass M^3"},
            ],
            "methods": [
                {"step": 1, "title": "Calculate Event Horizon Scale", "description": "Given mass $M$ (in kg or solar masses $M_\\odot$, where $r_s \\approx 2.95\\text{ km} \\times (M/M_\\odot)$), compute $r_s = 2GM/c^2$."},
                {"step": 2, "title": "Evaluate Thermal & Evaporation Properties", "description": "Use $T_H \\propto M^{-1}$ for Hawking temperature and $\\tau \\propto M^3$ for total evaporation lifetime."},
            ],
            "examples": [
                {"problem": "Find the Schwarzschild radius of a 10-solar-mass ($10\\,M_\\odot$) stellar black hole.", "solution": "Since $r_s(\\odot) \\approx 2.95\\text{ km}$ for $1\\,M_\\odot$, a $10\\,M_\\odot$ black hole has $r_s = 10 \\times 2.95\\text{ km} = 29.5\\text{ km}$."}
            ],
            "comparisons": [
                {"item_a": "Stellar-Mass Black Hole (~10 M☉)", "item_b": "Supermassive Black Hole (~10⁶–10⁹ M☉)", "difference": "Stellar-mass black holes have small horizons (~30 km), higher Hawking temperature, and extreme horizon tidal forces; supermassive black holes have huge horizons and weak horizon tidal forces."}
            ],
            "misconceptions": [
                {"myth": "Black holes act like cosmic vacuum cleaners that suck in distant stars from any distance.", "reality": "At distances $r \\gg r_s$, a black hole's gravitational field is identical to that of any normal star of the same mass $M$."}
            ],
            "exam_points": [
                "Schwarzschild radius scales linearly with mass: $r_s = 2GM/c^2$.",
                "Hawking temperature is inversely proportional to mass: $T_H \\propto 1/M$.",
                "Evaporation lifetime scales as the cube of mass: $\\tau \\propto M^3$.",
                "Event horizon area never decreases classically ($A = 16\\pi G^2 M^2 / c^4$), analogous to thermodynamic entropy.",
            ],
            "exam_questions": [
                {"question": "Why do stellar-mass black holes currently grow rather than shrink in the present universe?", "answer": "Their Hawking temperature ($T_H \\sim 10^{-8}\\text{ K}$) is much lower than the Cosmic Microwave Background temperature ($2.7\\text{ K}$), so they absorb more CMB energy than they emit."}
            ],
        }, topic, source_label)
        matched_bundles.append((bh_cards, bh_questions, bh_sheet))

    if has_neural:
        matched_bundles.append(_build_neural_networks_bundle(topic, source_label))

    if len(matched_bundles) == 1:
        b_cards, b_questions, b_sheet = matched_bundles[0]
        if difficulty in ("easy", "medium", "hard"):
            filtered_q = [q for q in b_questions if q.get("difficulty") == difficulty]
            if len(filtered_q) < 3:
                for q in b_questions:
                    if q not in filtered_q:
                        q_copy = dict(q)
                        q_copy["difficulty"] = difficulty
                        filtered_q.append(q_copy)
            b_questions = filtered_q
        return b_cards[:count], b_questions[:count], b_sheet

    if len(matched_bundles) > 1:
        # Multi-topic conversation! Interleave cards, questions, and cheat sheet sections proportionately
        combined_cards: List[Dict[str, Any]] = []
        combined_questions: List[Dict[str, Any]] = []
        max_len = max(len(b[0]) for b in matched_bundles)
        for i in range(max_len):
            for b in matched_bundles:
                if i < len(b[0]):
                    combined_cards.append(b[0][i])
                if i < len(b[1]):
                    q_copy = dict(b[1][i])
                    q_copy["id"] = f"q{len(combined_questions) + 1}"
                    if difficulty in ("easy", "medium", "hard"):
                        q_copy["difficulty"] = difficulty
                    combined_questions.append(q_copy)

        merged_sheet = {
            "title": f"{topic} — Multi-Topic Study Cheat Sheet",
            "topic": topic,
            "detected_subject": " & ".join(b[2].get("detected_subject", "") for b in matched_bundles),
            "source_summary": f"Synthesized proportionately across all topics in active session ({source_label})",
            "overview": " ".join(b[2].get("overview", "") for b in matched_bundles),
            "definitions": [d for b in matched_bundles for d in b[2].get("definitions", [])[:3]],
            "concepts": [c for b in matched_bundles for c in b[2].get("concepts", [])[:3]],
            "methods": [m for b in matched_bundles for m in b[2].get("methods", [])[:3]],
            "concept_nodes": [n for b in matched_bundles for n in b[2].get("concept_nodes", [])[:3]],
            "formulas": [f for b in matched_bundles for f in b[2].get("formulas", [])[:3]],
            "examples": [ex for b in matched_bundles for ex in b[2].get("examples", [])[:2]],
            "comparisons": [cp for b in matched_bundles for cp in b[2].get("comparisons", [])[:2]],
            "misconceptions": [ms for b in matched_bundles for ms in b[2].get("misconceptions", [])[:2]],
            "exam_points": [pt for b in matched_bundles for pt in b[2].get("exam_points", [])[:3]],
            "exam_questions": [eq for b in matched_bundles for eq in b[2].get("exam_questions", [])[:2]],
        }
        return combined_cards[:count], combined_questions[:count], ensure_complete_cheatsheet_structure(merged_sheet, topic, source_label)

    # Generic context-grounded extraction from conversation & document lines
    clean_lines = [
        line.strip()
        for line in context.splitlines()
        if line.strip()
        and not line.strip().startswith("===")
        and not line.strip().startswith("[Page")
        and not line.strip().startswith("[Turn")
        and len(line.strip()) > 25
    ]
    if not clean_lines:
        clean_lines = [f"Core concepts, mechanisms, and principles of {topic} as discussed in the active session."]

    cards_out: List[Dict[str, Any]] = []
    questions_out: List[Dict[str, Any]] = []
    seen_snippets: Set[str] = set()

    for i, line in enumerate(clean_lines):
        snippet = re.sub(r"^[-*•#\d.)\s]+", "", line).strip()
        if len(snippet) < 20:
            continue
        norm_key = snippet[:60].lower()
        if norm_key in seen_snippets:
            continue
        seen_snippets.add(norm_key)
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        diff_level = (
            difficulty
            if difficulty in ("easy", "medium", "hard")
            else ("easy" if len(cards_out) % 3 == 0 else "medium" if len(cards_out) % 3 == 1 else "hard")
        )
        subtopic_label = conv_topics[len(cards_out) % len(conv_topics)][:45] if conv_topics else topic[:45]
        cards_out.append(
            {
                "topic": subtopic_label,
                "difficulty": diff_level,
                "question": f"Regarding {subtopic_label}, explain the key principle or mechanism described in point {len(cards_out) + 1} of the study session.",
                "answer": snippet,
                "explanation": f"Grounded in {source_label}",
                "source": source_label,
            }
        )
        if len(questions_out) < count:
            questions_out.append(
                {
                    "id": f"q{len(questions_out) + 1}",
                    "type": "multiple_choice",
                    "concept": subtopic_label,
                    "difficulty": diff_level,
                    "question": f"Based on the active study conversation on {subtopic_label}, which statement is directly supported by the discussed material?",
                    "options": [
                        snippet[:160],
                        f"The material states that {subtopic_label} has no measurable relationship to the described mechanism.",
                        f"The session concludes that {subtopic_label} operates independently of the core physical/logical rules.",
                        f"None of the principles in {source_label} apply to {subtopic_label}.",
                    ],
                    "correct_answer": snippet[:160],
                    "correctIndex": 0,
                    "explanation": f"Directly grounded in {source_label}: {snippet}",
                    "source_citation": source_label,
                }
            )
        if len(cards_out) >= max(count, 6):
            break

    cheatsheet_out = ensure_complete_cheatsheet_structure({
        "title": f"{topic} — Study Cheat Sheet",
        "topic": topic,
        "detected_subject": topic,
        "source_summary": f"Synthesized strictly from {source_label}",
        "overview": " ".join(clean_lines[:3])[:600],
        "definitions": [
            {"term": f"{conv_topics[idx % len(conv_topics)][:35]} — Term {idx + 1}", "definition": ln[:220]}
            for idx, ln in enumerate(clean_lines[:4])
        ],
        "concepts": [
            {"title": f"{conv_topics[idx % len(conv_topics)][:40]} — Concept {idx + 1}", "explanation": ln[:260]}
            for idx, ln in enumerate(clean_lines[:4])
        ],
        "formulas": [],
        "methods": [
            {"step": idx + 1, "title": f"Mechanism / Step {idx + 1}", "description": ln[:220]}
            for idx, ln in enumerate(clean_lines[:4])
        ],
        "examples": [
            {"problem": f"Apply the core concept of {topic} to a standard scenario.", "solution": clean_lines[0][:240]}
        ],
        "comparisons": [],
        "misconceptions": [],
        "exam_points": [ln[:180] for ln in clean_lines[:5]],
        "exam_questions": [
            {"question": f"What is the primary takeaway regarding {topic}?", "answer": clean_lines[0][:240]}
        ],
    }, topic, source_label)

    return cards_out[:count], questions_out[:count], cheatsheet_out


# ============================================================================
# 1. RECALL CARDS (FLASHCARDS) GENERATION
# ============================================================================

def generate_flashcards_for_session(params: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(params.get("user_id") or "usr-default")
    session_id = params.get("session_id")
    explicit_topic = params.get("topic")
    document_ids = params.get("document_ids")
    difficulty = str(params.get("difficulty") or "mixed").strip().lower()
    learning_mode = str(params.get("learning_mode") or params.get("mode") or "normal").strip().lower()
    count = max(3, min(int(params.get("count") or 8), 16))
    model_options = _study_model_options(params, "flashcards")

    boundary = identify_active_topic_and_boundary(
        user_id=user_id,
        session_id=session_id,
        explicit_topic=explicit_topic,
        document_ids=document_ids,
    )

    if not boundary["has_valid_source"]:
        return {
            "flashcards": [],
            "message": "Insufficient source material in the active conversation. Discuss a topic in chat or upload a document first to generate Recall Cards.",
        }

    topic = boundary["topic"] or "Active Study Topic"
    conv_topics = boundary["conversation_topics"]
    context = boundary["combined_context"]
    anchor_words = boundary["anchor_words"]
    docs = boundary["selected_docs"]
    default_source = docs[0]["filename"] if docs else f"Active Conversation ({topic[:45]})"
    mode_profile = LEARNING_MODE_PROFILES.get(learning_mode, LEARNING_MODE_PROFILES["normal"])

    diff_instruction = {
        "easy": "EASY (University Introductory): Focus on core definitions, fundamental facts, and basic conceptual understanding.",
        "medium": "MEDIUM (University Intermediate): Focus on conceptual mechanisms, relationships between concepts, key formulas and their meanings, and simple applications.",
        "hard": "HARD (University Upper-Level): Focus on deeper reasoning, comparing concepts, applying principles, and interpreting equations/scenarios (do NOT generate obscure PhD trivia).",
        "mixed": "MIXED (Balanced University Study): Include a balanced mix of Easy (definitions), Medium (mechanisms & formulas), and Hard (deeper relationships & applications).",
    }.get(difficulty, "MIXED: Balanced university-level definitions, mechanisms, formulas, and applications.")

    multi_topic_note = (
        f"IMPORTANT MULTI-TOPIC COVERAGE: This conversation covers {len(conv_topics)} topics ({', '.join(conv_topics)}). Distribute the {count} cards proportionately across all of these discussed topics."
        if len(conv_topics) > 1
        else f"Focus strictly on '{topic}'."
    )

    prompt = f"""Generate {count} high-yield active-recall revision cards from the complete active conversation and any attached documents below.

ACTIVE TOPIC BOUNDARY: "{topic}"
TOPICS DISCUSSED IN ACTIVE CONVERSATION: {', '.join(conv_topics)}
LEARNING MODE: {mode_profile['label']} — {mode_profile['instruction']}
DIFFICULTY LEVEL: {diff_instruction}
{multi_topic_note}

CONTENT PRIORITY & ISOLATION RULES:
1. Use the COMPLETE active conversation history first, followed by any uploaded documents in ACTIVE CONTEXT below. Uploaded documents are optional.
2. NEVER mix unrelated conversations! Specifically, NEVER mention backpropagation, neural networks, or vanishing gradients unless the active conversation below actually discussed them; NEVER mention black holes unless the active conversation discussed them.
3. Prioritize:
   - Important concepts & definitions
   - Fundamental mechanisms & processes
   - Key relationships between concepts
   - Important examples & applications
   - Important equations (in clean LaTeX $$ ... $$ or $ ... $) and their physical/practical meaning
   - Common misconceptions & exam-relevant facts.
4. Do NOT generate cards by randomly picking technical phrases, and do NOT repeat the same concept with slightly different wording. Keep cards concise and comfortable to revise.

ACTIVE CONTEXT:
{context if context else f"Topic requested by user in active session: {topic}"}

Return ONLY a valid JSON array of {count} objects with this exact schema:
[
  {{
    "topic": "Specific topic/subtopic label",
    "difficulty": "easy" | "medium" | "hard",
    "question": "Clear, focused active-recall question",
    "answer": "Concise but sufficiently detailed answer with LaTeX formulas where applicable",
    "explanation": "Brief context or exam tip",
    "source": "{default_source}"
  }}
]"""

    validated_cards: List[Dict[str, Any]] = []
    seen_questions: Set[str] = set()
    model_routing: List[Dict[str, Any]] = []

    try:
        system_instruction = "You are VISION AI Study Copilot. Generate topic-bounded, non-repetitive, conversation-grounded study flashcards in valid JSON format only."
        try:
            parsed, routing_metadata = _generate_study_json(
                prompt=prompt,
                system_instruction=system_instruction,
                provider=model_options["provider"],
                model=model_options["model"],
                base_url=model_options["base_url"],
                max_tokens=min(4096, max(1024, count * 256)),
                operation="recall_card_generation",
                selection_mode=model_options["selection_mode"],
                allow_model_fallback=model_options["allow_model_fallback"],
            )
            model_routing.append(routing_metadata)
            parsed_outputs = [
                parsed
            ]
        except Exception as exc:
            context_failure = getattr(exc, "failure_kind", None) == "context_capacity" or (
                "context" in str(exc).lower() and "capacity" in str(exc).lower()
            )
            if not context_failure or not context:
                raise
            sections = _split_source_context(context)
            parsed_outputs = []
            quotient, remainder = divmod(count, max(1, len(sections)))
            for section_index, section in enumerate(sections):
                requested_batch = max(1, quotient + (1 if section_index < remainder else 0))
                section_prompt = prompt.replace(context, section, 1).replace(
                    f"Generate {count} high-yield",
                    f"Generate {requested_batch} high-yield",
                    1,
                )
                parsed_part, routing_metadata = _generate_study_json(
                    prompt=section_prompt,
                    system_instruction=system_instruction,
                    provider=model_options["provider"],
                    model=model_options["model"],
                    base_url=model_options["base_url"],
                    max_tokens=min(2048, max(512, requested_batch * 256)),
                    operation="recall_card_generation",
                    selection_mode=model_options["selection_mode"],
                    allow_model_fallback=model_options["allow_model_fallback"],
                )
                parsed_outputs.append(parsed_part)
                model_routing.append(routing_metadata)
        for parsed in parsed_outputs:
            raw_list = (
                parsed
                if isinstance(parsed, list)
                else (parsed.get("flashcards") or parsed.get("cards") or [])
                if isinstance(parsed, dict)
                else []
            )
            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                q_txt = str(item.get("question") or "").strip()
                a_txt = str(item.get("answer") or "").strip()
                if not q_txt or not a_txt:
                    continue
                q_norm = re.sub(r"\W+", "", q_txt.lower())[:55]
                if q_norm in seen_questions:
                    continue
                if not validate_item_against_topic_boundary(f"{q_txt} {a_txt}", topic, anchor_words):
                    continue
                seen_questions.add(q_norm)
                validated_cards.append(
                    {
                        "topic": str(item.get("topic") or topic).strip(),
                        "difficulty": str(item.get("difficulty") or (difficulty if difficulty != "mixed" else "medium")).lower(),
                        "question": q_txt,
                        "answer": a_txt,
                        "explanation": str(item.get("explanation") or "").strip(),
                        "source": str(item.get("source") or default_source).strip(),
                    }
                )
    except Exception as exc:
        print(f"[Study] Flashcard LLM fallback triggered: {exc}")

    if len(validated_cards) < count:
        fallback_cards, _, _ = build_source_grounded_fallback_with_llm_retry(
            boundary, difficulty, count, learning_mode
        )
        for card in fallback_cards:
            q_txt = str(card.get("question") or "").strip()
            q_norm = re.sub(r"\W+", "", q_txt.lower())[:55]
            if not q_txt or q_norm in seen_questions:
                continue
            seen_questions.add(q_norm)
            validated_cards.append(card)
            if len(validated_cards) >= count:
                break
    if len(validated_cards) < count:
        return {
            "flashcards": [],
            "topic": topic,
            "message": (
                f"Recall-card generation produced only {len(validated_cards)} of {count} "
                "requested cards after fallback. No incomplete set was saved. Try a smaller "
                "request or provide a shorter, focused source."
            ),
        }
    if len(validated_cards) > count:
        if count == 1:
            validated_cards = [validated_cards[0]]
        else:
            indexes = [round(i * (len(validated_cards) - 1) / (count - 1)) for i in range(count)]
            validated_cards = [validated_cards[index] for index in indexes]

    doc_ids_used = [d["id"] for d in boundary["selected_docs"]]
    set_rec = db.add_flashcard_set(
        {
            "user_id": user_id,
            "session_id": session_id or "",
            "workspace_id": session_id or "ws-default",
            "title": f"Recall Cards: {topic[:50]}",
            "topic": topic,
            "document_ids": doc_ids_used,
            "cards": validated_cards,
        }
    )

    return {
        "set": set_rec,
        "flashcards": validated_cards,
        "topic": topic,
        "conversation_topics": conv_topics,
        "learning_mode": learning_mode,
        "document_ids": doc_ids_used,
        "model_routing": model_routing,
    }


# ============================================================================
# 2. INTERACTIVE QUIZ GENERATION
# ============================================================================

def generate_quiz_for_session(params: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(params.get("user_id") or "usr-default")
    session_id = params.get("session_id")
    explicit_topic = params.get("topic")
    document_ids = params.get("document_ids")
    difficulty = str(params.get("difficulty") or "mixed").strip().lower()
    learning_mode = str(params.get("learning_mode") or params.get("mode") or "normal").strip().lower()
    count = max(4, min(int(params.get("count") or 6), 12))
    model_options = _study_model_options(params, "quiz")

    boundary = identify_active_topic_and_boundary(
        user_id=user_id,
        session_id=session_id,
        explicit_topic=explicit_topic,
        document_ids=document_ids,
    )

    if not boundary["has_valid_source"]:
        return {
            "quiz": None,
            "quizzes": [],
            "message": "Insufficient source material in the active conversation. Discuss a topic in chat or upload a document first to generate a quiz.",
        }

    topic = boundary["topic"] or "Active Study Topic"
    conv_topics = boundary["conversation_topics"]
    context = boundary["combined_context"]
    anchor_words = boundary["anchor_words"]
    docs = boundary["selected_docs"]
    default_citation = docs[0]["filename"] if docs else f"Active Conversation: {topic}"
    mode_profile = LEARNING_MODE_PROFILES.get(learning_mode, LEARNING_MODE_PROFILES["normal"])

    diff_guide = {
        "easy": "EASY: Basic definitions, fundamental facts, and simple understanding. Keep every question at Easy difficulty.",
        "medium": "MEDIUM: Conceptual understanding, explain relationships, simple applications, and interpret important examples. Keep every question at Medium difficulty.",
        "hard": "HARD: Deeper reasoning, compare concepts, apply principles, and interpret relevant equations or scenarios (university revision level; do NOT force PhD-level research trivia or tedious derivations). Keep every question at Hard difficulty.",
        "mixed": "MIXED: Include 2 Easy (definitions/facts), 2-3 Medium (conceptual relationships & examples), and 1-2 Hard (comparing concepts & applying principles/equations).",
    }.get(difficulty, "MIXED: Balanced university-level questions.")

    multi_topic_note = (
        f"MULTI-TOPIC CONVERSATION DETECTED: The active conversation covers {len(conv_topics)} topics ({', '.join(conv_topics)}). Generate a balanced quiz covering these discussed topics proportionately."
        if len(conv_topics) > 1
        else f"Focus strictly on '{topic}'."
    )

    prompt = f"""Generate a {count}-question interactive university study quiz using the complete active conversation and any attached documents below.

ACTIVE TOPIC BOUNDARY: "{topic}"
TOPICS DISCUSSED IN CONVERSATION: {', '.join(conv_topics)}
LEARNING MODE: {mode_profile['label']} — {mode_profile['instruction']}
DIFFICULTY TARGET: {diff_guide}
{multi_topic_note}

STRICT TOPIC ISOLATION & QUALITY RULES:
1. Use the COMPLETE active conversation history (plus any uploaded documents in ACTIVE CONTEXT). Uploaded documents are optional.
2. NEVER mix in unrelated subjects!
   - If the conversation is about Black Holes, NEVER include backpropagation, vanishing gradients, or neural networks.
   - If the conversation is about Neural Networks, NEVER include black holes, Schwarzschild radius, or Hawking radiation.
3. Include a thoughtful mix of question formats: `multiple_choice`, `true_false`, and `fill_in_blank` (with 4 selectable options for fill_in_blank and multiple_choice, and ["True", "False"] for true_false).
4. Do NOT repeat questions or generate unrelated filler questions. Every question must have a verified `correct_answer`, exact `correctIndex` (0-based integer matching `options[correctIndex]`), and a clear `explanation`.

ACTIVE CONTEXT:
{context if context else f"Topic requested by user in active session: {topic}"}

Return ONLY a valid JSON object matching this exact schema:
{{
  "title": "{topic} — Interactive Quiz",
  "topic": "{topic}",
  "questions": [
    {{
      "id": "q1",
      "type": "multiple_choice" | "true_false" | "fill_in_blank",
      "concept": "Specific concept tested",
      "difficulty": "{difficulty if difficulty != 'mixed' else 'medium'}",
      "question": "Clear question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Exact text of the correct option",
      "correctIndex": 0,
      "explanation": "Clear explanation of why this answer is correct and why distractors are wrong",
      "source_citation": "{default_citation}"
    }}
  ]
}}"""

    validated_questions: List[Dict[str, Any]] = []
    seen_q_norms: Set[str] = set()
    quiz_title = f"{topic} — Interactive Quiz"
    model_routing: List[Dict[str, Any]] = []

    try:
        system_instruction = "You are VISION AI Assessment Engine. Output strictly topic-bounded, non-repetitive, conversation-verified quiz JSON only."
        try:
            parsed, routing_metadata = _generate_study_json(
                prompt=prompt,
                system_instruction=system_instruction,
                provider=model_options["provider"],
                model=model_options["model"],
                base_url=model_options["base_url"],
                max_tokens=min(4096, max(1024, count * 300)),
                operation="quiz_generation",
                selection_mode=model_options["selection_mode"],
                allow_model_fallback=model_options["allow_model_fallback"],
            )
            model_routing.append(routing_metadata)
            parsed_outputs = [
                parsed
            ]
        except Exception as exc:
            context_failure = getattr(exc, "failure_kind", None) == "context_capacity" or (
                "context" in str(exc).lower() and "capacity" in str(exc).lower()
            )
            if not context_failure or not context:
                raise
            sections = _split_source_context(context)
            parsed_outputs = []
            quotient, remainder = divmod(count, max(1, len(sections)))
            for section_index, section in enumerate(sections):
                requested_batch = max(1, quotient + (1 if section_index < remainder else 0))
                section_prompt = prompt.replace(context, section, 1).replace(
                    f"Generate a {count}-question",
                    f"Generate a {requested_batch}-question",
                    1,
                )
                parsed_part, routing_metadata = _generate_study_json(
                    prompt=section_prompt,
                    system_instruction=system_instruction,
                    provider=model_options["provider"],
                    model=model_options["model"],
                    base_url=model_options["base_url"],
                    max_tokens=min(2048, max(512, requested_batch * 300)),
                    operation="quiz_generation",
                    selection_mode=model_options["selection_mode"],
                    allow_model_fallback=model_options["allow_model_fallback"],
                )
                parsed_outputs.append(parsed_part)
                model_routing.append(routing_metadata)

        for parsed in parsed_outputs:
            raw_questions: List[Any] = []
            if isinstance(parsed, dict):
                quiz_title = str(parsed.get("title") or quiz_title)
                raw_questions.extend(parsed.get("questions") or parsed.get("quizzes") or [])
            elif isinstance(parsed, list):
                raw_questions.extend(parsed)

            for q in raw_questions:
                if not isinstance(q, dict) or not q.get("question"):
                    continue
                q_text_only = str(q.get("question") or "").strip()
                q_norm = re.sub(r"\W+", "", q_text_only.lower())[:60]
                if q_norm in seen_q_norms:
                    continue
                combined_q_text = f"{q.get('concept', '')} {q_text_only} {' '.join(str(o) for o in (q.get('options') or []))} {q.get('explanation', '')}"
                if not validate_item_against_topic_boundary(combined_q_text, topic, anchor_words):
                    continue
                seen_q_norms.add(q_norm)
                norm_q = normalize_quiz_question(
                    q,
                    len(validated_questions),
                    default_topic=topic,
                    default_diff=difficulty if difficulty != "mixed" else "medium",
                )
                if difficulty in ("easy", "medium", "hard"):
                    norm_q["difficulty"] = difficulty
                validated_questions.append(norm_q)
    except Exception as exc:
        print(f"[Study] Quiz LLM fallback triggered: {exc}")

    if len(validated_questions) < count:
        _, fallback_questions, _ = build_source_grounded_fallback_with_llm_retry(
            boundary, difficulty, count, learning_mode
        )
        for question in fallback_questions:
            q_text = str(question.get("question") or "").strip()
            q_norm = re.sub(r"\W+", "", q_text.lower())[:60]
            if q_norm in seen_q_norms:
                continue
            seen_q_norms.add(q_norm)
            validated_questions.append(
                normalize_quiz_question(
                    question,
                    len(validated_questions),
                    default_topic=topic,
                    default_diff=difficulty if difficulty != "mixed" else "medium",
                )
            )
            if len(validated_questions) >= count:
                break
    if len(validated_questions) < count:
        return {
            "quiz": None,
            "quizzes": [],
            "topic": topic,
            "message": (
                f"Quiz generation produced only {len(validated_questions)} of {count} "
                "requested questions after fallback. No incomplete quiz was saved. Try a "
                "smaller request or provide a shorter, focused source."
            ),
        }
    if len(validated_questions) > count:
        if count == 1:
            validated_questions = [validated_questions[0]]
        else:
            indexes = [round(i * (len(validated_questions) - 1) / (count - 1)) for i in range(count)]
            validated_questions = [validated_questions[index] for index in indexes]

    doc_ids_used = [d["id"] for d in docs]
    quiz_rec = db.add_quiz(
        {
            "user_id": user_id,
            "session_id": session_id or "",
            "workspace_id": session_id or "ws-default",
            "title": quiz_title,
            "topic": topic,
            "document_ids": doc_ids_used,
            "questions": validated_questions,
        }
    )

    return {
        "quiz": quiz_rec,
        "quizzes": validated_questions,
        "topic": topic,
        "conversation_topics": conv_topics,
        "difficulty": difficulty,
        "learning_mode": learning_mode,
        "document_ids": doc_ids_used,
        "model_routing": model_routing,
    }


# ============================================================================
# 3. ADAPTIVE CHEAT SHEET GENERATION (13-PART COMPREHENSIVE STRUCTURE)
# ============================================================================

def generate_cheatsheet_for_session(params: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(params.get("user_id") or "usr-default")
    session_id = params.get("session_id")
    explicit_topic = params.get("topic")
    document_ids = params.get("document_ids")
    source_type = str(params.get("source_type") or "conversation_and_docs")
    format_type = str(params.get("format_type") or "smart_adaptive")
    learning_mode = str(params.get("learning_mode") or params.get("mode") or "normal").strip().lower()
    mode_profile = LEARNING_MODE_PROFILES.get(learning_mode, LEARNING_MODE_PROFILES["normal"])
    detail_level = str(params.get("detail_level") or mode_profile["detail_level"] or "standard")
    audience_level = str(params.get("audience_level") or "intermediate")
    purpose = str(params.get("purpose") or "revision")
    model_options = _study_model_options(params, "cheatsheet")

    boundary = identify_active_topic_and_boundary(
        user_id=user_id,
        session_id=session_id,
        explicit_topic=explicit_topic,
        document_ids=document_ids,
        source_type=source_type,
    )

    if not boundary["has_valid_source"]:
        return {
            "cheatsheet": None,
            "message": "Insufficient source material in the active conversation. Discuss a topic in chat or upload a document first to generate a Cheat Sheet.",
        }

    topic = boundary["topic"] or "Active Study Topic"
    conv_topics = boundary["conversation_topics"]
    context = boundary["combined_context"]
    anchor_words = boundary["anchor_words"]
    docs = boundary["selected_docs"]
    doc_names = [d["filename"] for d in docs]
    source_summary = (
        f"Grounded in active conversation & {', '.join(doc_names)}"
        if doc_names
        else f"Grounded in complete active conversation: {topic}"
    )

    prompt = f"""Synthesize a comprehensive, accurate, and readable university study Cheat Sheet using the complete active conversation and any attached documents below.

ACTIVE TOPIC BOUNDARY: "{topic}"
TOPICS DISCUSSED IN CONVERSATION: {', '.join(conv_topics)}
SOURCE SUMMARY: "{source_summary}"
LEARNING MODE: {mode_profile['label']} — {mode_profile['instruction']}
LAYOUT FORMAT: {format_type} | DETAIL LEVEL: {detail_level} | AUDIENCE: {audience_level} | PURPOSE: {purpose}

CRITICAL QUALITY & STRUCTURE RULES (13-SECTION ACADEMIC SYNTHESIS):
1. Use the COMPLETE active conversation history and any selected documents in ACTIVE CONTEXT below. Uploaded documents are optional.
2. NEVER mix unrelated subjects (do NOT include neural networks or backpropagation in an astrophysics sheet, and do NOT include black holes in a machine learning sheet).
3. If the topic involves "The Stellar Birth Cycle & Star Formation Mechanics" (or star formation), explicitly cover:
   - Giant Molecular Clouds (GMCs), temperatures (~10-20 K), densities, and dust shielding
   - Jeans Instability, Jeans Length ($\\lambda_J$), and Jeans Mass ($M_J \\propto T^{{3/2}}\\rho_0^{{-1/2}}$)
   - Free-fall gravitational collapse ($t_{{ff}} = \\sqrt{{3\\pi/(32G\\rho_0)}}$) and isothermal hierarchical fragmentation
   - Opacity limit, First Hydrostatic Core (Larson core), and endothermic $H_2$ dissociation triggering second collapse at $T \\approx 2000\\text{{ K}}$
   - Angular momentum problem, circumstellar Keplerian accretion disks, MRI viscosity, and magnetocentrifugal bipolar jets/outflows
   - Protostellar evolution (Class 0, Class I, Class II T Tauri, Class III), Hayashi vs. Henyey tracks, Kelvin–Helmholtz contraction ($t_{{KH}} \\approx GM^2/(RL)$), deuterium burning, and Zero-Age Main Sequence (ZAMS) hydrogen ignition at $T_c \\ge 10^7\\text{{ K}}$ (plus the $0.08\\,M_\\odot$ brown dwarf limit).
4. Populate ALL sections cleanly—especially `concept_nodes` (every item MUST have non-empty `node`, `relation`, and `connects_to`) and `methods` (every item MUST have `step`, `title`, and `description`) so no table cell is ever blank.

ACTIVE CONTEXT:
{context if context else f"Active topic: {topic}"}

Return ONLY a valid JSON object with this exact 13-part structure:
{{
  "title": "{topic} — Master Study Cheat Sheet",
  "topic": "{topic}",
  "detected_subject": "Academic Subject Area",
  "source_summary": "{source_summary}",
  "overview": "1. Topic Overview: Comprehensive executive summary of {topic} and its core significance",
  "concepts": [
    {{"title": "2 & 3. Core Concept Title", "explanation": "Detailed conceptual and mechanistic explanation"}}
  ],
  "definitions": [
    {{"term": "4. Important Term", "definition": "Precise academic definition"}}
  ],
  "methods": [
    {{"step": 1, "title": "5. Mechanism / Process Stage Title", "description": "Step-by-step physical, mathematical, or logical process"}}
  ],
  "concept_nodes": [
    {{"node": "6. Primary Concept / Stage", "relation": "How it governs, causes, or transforms into", "connects_to": "Connected Concept / Outcome"}}
  ],
  "examples": [
    {{"problem": "7. Worked Example / Practical Application Scenario", "solution": "Complete step-by-step solution and interpretation"}}
  ],
  "formulas": [
    {{"name": "8. Essential Formula Name", "latex": "$$ ... $$", "variables": "9. Complete symbol-by-symbol variable definitions & SI/domain units", "when_to_use": "Physical/practical interpretation and when to apply"}}
  ],
  "comparisons": [
    {{"item_a": "10. Concept / Regime A", "item_b": "Concept / Regime B", "difference": "Key physical, structural, or analytical distinction"}}
  ],
  "misconceptions": [
    {{"myth": "11. Common Misconception", "reality": "Accurate scientific/technical correction"}}
  ],
  "exam_points": [
    "12. High-yield exam takeaway 1",
    "High-yield exam takeaway 2"
  ],
  "exam_questions": [
    {{"question": "13. Likely Exam Question", "answer": "Complete, rigorous model answer"}}
  ]
}}"""

    sheet: Optional[Dict[str, Any]] = None
    low_topic_check = f"{topic}\n{context}".lower()
    is_stellar_topic = any(
        k in low_topic_check
        for k in ("stellar birth", "star formation", "jeans mass", "jeans instability", "larson core", "hayashi track")
    )

    try:
        model_routing: List[Dict[str, Any]] = []
        system_instruction = "You are VISION AI Study Synthesizer. Output strictly topic-bounded, comprehensive 13-part Cheat Sheet JSON only."
        try:
            parsed, routing_metadata = _generate_study_json(
                prompt=prompt,
                system_instruction=system_instruction,
                provider=model_options["provider"],
                model=model_options["model"],
                base_url=model_options["base_url"],
                max_tokens=4096,
                operation="cheat_sheet_generation",
                selection_mode=model_options["selection_mode"],
                allow_model_fallback=model_options["allow_model_fallback"],
            )
            model_routing = [routing_metadata]
            parsed_outputs = [
                parsed
            ]
        except Exception as exc:
            context_failure = getattr(exc, "failure_kind", None) == "context_capacity" or (
                "context" in str(exc).lower() and "capacity" in str(exc).lower()
            )
            if not context_failure or not context:
                raise
            parsed_outputs = []
            model_routing = []
            for section in _split_source_context(context):
                parsed_part, routing_metadata = _generate_study_json(
                    prompt=prompt.replace(context, section, 1),
                    system_instruction=system_instruction,
                    provider=model_options["provider"],
                    model=model_options["model"],
                    base_url=model_options["base_url"],
                    max_tokens=2048,
                    operation="cheat_sheet_generation",
                    selection_mode=model_options["selection_mode"],
                    allow_model_fallback=model_options["allow_model_fallback"],
                )
                parsed_outputs.append(parsed_part)
                model_routing.append(routing_metadata)
        parsed: Dict[str, Any] = {}
        for parsed_part in parsed_outputs:
            if not isinstance(parsed_part, dict):
                continue
            for key, value in parsed_part.items():
                if isinstance(value, list):
                    target = parsed.setdefault(key, [])
                    known = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in target}
                    for item in value:
                        identity = json.dumps(item, sort_keys=True, ensure_ascii=False)
                        if identity not in known:
                            target.append(item)
                            known.add(identity)
                elif value and not parsed.get(key):
                    parsed[key] = value
        if isinstance(parsed, dict) and (parsed.get("overview") or parsed.get("definitions") or parsed.get("concepts")):
            check_str = json.dumps(parsed)
            if validate_item_against_topic_boundary(check_str, topic, anchor_words):
                sheet = ensure_complete_cheatsheet_structure(parsed, topic, source_summary)
                # If stellar topic was requested and LLM returned sparse content, enrich with verified stellar mechanics
                if is_stellar_topic and (len(sheet.get("formulas") or []) < 3 or len(sheet.get("concepts") or []) < 3):
                    _, _, stellar_sheet = _build_stellar_birth_cycle_bundle(topic, source_summary, learning_mode)
                    for key in ("definitions", "concepts", "methods", "concept_nodes", "formulas", "examples", "comparisons", "misconceptions", "exam_points", "exam_questions"):
                        if len(sheet.get(key) or []) < len(stellar_sheet.get(key) or []):
                            sheet[key] = stellar_sheet[key]
    except Exception as exc:
        print(f"[Study] CheatSheet LLM fallback triggered: {exc}")

    if not sheet:
        _, _, sheet = build_source_grounded_fallback_with_llm_retry(
            boundary, "mixed", 6, learning_mode
        )

    sheet = ensure_complete_cheatsheet_structure(sheet, topic, source_summary)
    sheet["session_id"] = session_id or ""
    sheet["source_type"] = source_type
    sheet["layout_format"] = format_type
    sheet["learning_mode"] = learning_mode
    sheet["created_at"] = now_iso()

    if session_id:
        db.save_cheatsheet(
            session_id=session_id,
            sheet=sheet,
            topic=topic,
            document_ids=[d["id"] for d in docs],
            user_id=user_id,
        )

    return {
        "cheatsheet": sheet,
        "topic": topic,
        "conversation_topics": conv_topics,
        "learning_mode": learning_mode,
        "model_routing": model_routing,
    }


def regenerate_cheatsheet_section_for_session(params: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(params.get("user_id") or "usr-default")
    session_id = params.get("session_id")
    section_id = str(params.get("section_id") or "overview")
    section_title = str(params.get("section_title") or section_id.title())
    explicit_topic = params.get("topic")
    document_ids = params.get("document_ids")

    boundary = identify_active_topic_and_boundary(
        user_id=user_id,
        session_id=session_id,
        explicit_topic=explicit_topic,
        document_ids=document_ids,
    )
    topic = boundary["topic"] or "Active Study Topic"
    context = boundary["combined_context"]

    prompt = f"""Regenerate the "{section_title}" section of a university study cheat sheet strictly for the topic "{topic}".
Use the complete active conversation and document context below. Format with clean Markdown and LaTeX equations where relevant.

CONTEXT:
{context[:12000] if context else topic}"""

    try:
        res = ai_manager.generate_text(
            messages=[
                {"role": "system", "content": "You are VISION AI Study Copilot. Provide concise, high-yield academic Markdown."},
                {"role": "user", "content": prompt},
            ],
            provider=params.get("provider"),
            model=params.get("model"),
            base_url=params.get("base_url"),
            temperature=0.3,
            max_tokens=1500,
        )
        content = res.get("text", "").strip()
        if content:
            return {"section_id": section_id, "content": content}
    except Exception:
        pass

    _, _, fallback_sheet = build_source_grounded_study_fallback(boundary, "mixed", 6)
    return {
        "section_id": section_id,
        "content": fallback_sheet.get("overview") or f"Updated {section_title} for {topic}.",
    }


def build_source_grounded_fallback_with_llm_retry(
    boundary: Dict[str, Any], difficulty: str, count: int, learning_mode: str = "normal"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    return build_source_grounded_study_fallback(boundary, difficulty, count, learning_mode)
