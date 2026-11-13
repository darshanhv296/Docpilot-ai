"""
DocPilot AI / VISION AI — Multimodal Vision Solver Service (vision_service.py)
Provides binary image payload validation, 14-category visual classification,
dynamic domain-specific action routing, and grounded multimodal analysis.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

from app.services.ai_provider import (
    ProviderRequestError,
    ai_manager,
    parse_json_safely,
    sanitize_log_text,
)

logger = logging.getLogger(__name__)


VISION_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "math_equation": {
        "id": "math_equation",
        "name": "Math / Equation",
        "actions": [
            {"id": "transcribe_latex", "label": "Transcribe to LaTeX", "description": "Convert visible mathematical notation into clean LaTeX", "icon": "fa-superscript text-brand-400"},
            {"id": "solve_step_by_step", "label": "Solve Step-by-Step", "description": "Complete mathematical derivation & final answer", "icon": "fa-wand-magic-sparkles text-emerald-400"},
            {"id": "check_mistakes", "label": "Check for Calculation Mistakes", "description": "Verify each step for sign, algebra, or calculus errors", "icon": "fa-magnifying-glass-chart text-rose-400"},
            {"id": "explain_concepts", "label": "Explain Concepts Used", "description": "Explain theorems, identities & mathematical intuition", "icon": "fa-lightbulb text-amber-400"},
            {"id": "similar_problems", "label": "Generate Similar Practice Problems", "description": "Create 3 practice problems with worked solutions", "icon": "fa-list-ol text-indigo-400"},
        ],
    },
    "physics_problem": {
        "id": "physics_problem",
        "name": "Physics Problem",
        "actions": [
            {"id": "identify_given", "label": "Identify Given Values & Units", "description": "Extract physical quantities, vectors & boundary conditions", "icon": "fa-compass-drafting text-brand-400"},
            {"id": "identify_formula", "label": "Identify the Applicable Formula", "description": "State governing physical laws and equations", "icon": "fa-square-root-variable text-indigo-400"},
            {"id": "solve_step_by_step", "label": "Solve Step-by-Step", "description": "Full physical derivation & numerical evaluation", "icon": "fa-wand-magic-sparkles text-emerald-400"},
            {"id": "explain_principles", "label": "Explain Physical Principles", "description": "Explain forces, conservation laws & intuition", "icon": "fa-atom text-amber-400"},
            {"id": "check_units", "label": "Check Units and Dimensions", "description": "Verify SI unit consistency & dimensional analysis", "icon": "fa-scale-balanced text-rose-400"},
        ],
    },
    "chemistry_reaction": {
        "id": "chemistry_reaction",
        "name": "Chemistry / Reaction",
        "actions": [
            {"id": "identify_compounds", "label": "Identify Compounds and Structures", "description": "Name reactants, products & functional groups", "icon": "fa-flask text-brand-400"},
            {"id": "balance_equation", "label": "Balance the Equation", "description": "Balance stoichiometry, atoms & ionic charges", "icon": "fa-scale-balanced text-emerald-400"},
            {"id": "explain_mechanism", "label": "Explain Reaction Mechanism", "description": "Detail electron movement, intermediates & bonds", "icon": "fa-diagram-project text-indigo-400"},
            {"id": "calculate_quantities", "label": "Calculate Quantities", "description": "Compute molar masses, limiting reagents & yields", "icon": "fa-calculator text-amber-400"},
            {"id": "explain_concepts", "label": "Explain Chemical Concepts", "description": "Explain bonding, equilibrium or thermodynamics", "icon": "fa-lightbulb text-rose-400"},
        ],
    },
    "circuit_schematic": {
        "id": "circuit_schematic",
        "name": "Circuit / Schematic",
        "actions": [
            {"id": "identify_components", "label": "Identify Components", "description": "List resistors, capacitors, ICs, sources & values", "icon": "fa-microchip text-brand-400"},
            {"id": "explain_operation", "label": "Explain Circuit Operation", "description": "Explain how the circuit works and its purpose", "icon": "fa-bolt text-emerald-400"},
            {"id": "trace_current", "label": "Trace Current Flow", "description": "Follow current paths, node voltages & loops", "icon": "fa-wave-square text-indigo-400"},
            {"id": "analyze_connections", "label": "Analyze Pin / Node Connections", "description": "Map series/parallel topology and pin wiring", "icon": "fa-network-wired text-amber-400"},
            {"id": "identify_issues", "label": "Identify Possible Circuit Issues", "description": "Check for shorts, open loops or polarity errors", "icon": "fa-bug text-rose-400"},
        ],
    },
    "engineering_diagram": {
        "id": "engineering_diagram",
        "name": "Engineering Diagram",
        "actions": [
            {"id": "identify_components", "label": "Identify Components and Structure", "description": "Catalog subsystems, members, nodes & blocks", "icon": "fa-sitemap text-brand-400"},
            {"id": "explain_design", "label": "Explain Engineering Design", "description": "Explain system architecture & working principles", "icon": "fa-gears text-emerald-400"},
            {"id": "extract_dimensions", "label": "Extract Dimensions and Labels", "description": "Extract all visible measurements, units & labels", "icon": "fa-ruler-combined text-indigo-400"},
            {"id": "explain_load_paths", "label": "Explain Load Paths / Functional Relationships", "description": "Analyze force flow, signal flow or process dependencies", "icon": "fa-arrows-split-up-and-left text-amber-400"},
            {"id": "identify_design_issues", "label": "Identify Potential Design Issues", "description": "Spot bottlenecks, missing constraints or ambiguities", "icon": "fa-shield-halved text-rose-400"},
        ],
    },
    "architecture_structural": {
        "id": "architecture_structural",
        "name": "Architecture / Structural Design",
        "actions": [
            {"id": "identify_structural_elements", "label": "Identify Structural Elements", "description": "Catalog trusses, chords, webs, beams, columns & joints", "icon": "fa-building-columns text-brand-400"},
            {"id": "extract_visible_dimensions", "label": "Extract Dimensions and Spans", "description": "Extract visible spans, heights, pitches, angles & labels", "icon": "fa-ruler-combined text-emerald-400"},
            {"id": "explain_structural_layout", "label": "Explain Structural Layout", "description": "Explain geometry, framing system & support conditions", "icon": "fa-compass-drafting text-indigo-400"},
            {"id": "identify_load_transfer", "label": "Explain Load Paths (Tension / Compression)", "description": "Analyze how loads transfer through members to supports", "icon": "fa-arrows-down-to-line text-amber-400"},
            {"id": "review_design_anomalies", "label": "Identify Potential Design Issues", "description": "Check for missing dimensions, bracing or ambiguities", "icon": "fa-shield-halved text-rose-400"},
        ],
    },
    "mechanical_drawing": {
        "id": "mechanical_drawing",
        "name": "Mechanical / Technical Drawing",
        "actions": [
            {"id": "identify_mechanical_components", "label": "Identify Mechanical Components", "description": "Identify parts, features, views & fasteners", "icon": "fa-gears text-brand-400"},
            {"id": "extract_dimensions", "label": "Extract Dimensions and Tolerances", "description": "Extract visible diameters, lengths, angles & tolerances", "icon": "fa-ruler-combined text-emerald-400"},
            {"id": "explain_assembly", "label": "Explain Assembly / Part Geometry", "description": "Explain orthographic views, sections & kinematic fit", "icon": "fa-screwdriver-wrench text-indigo-400"},
            {"id": "interpret_symbols", "label": "Interpret Technical Symbols & GD&T", "description": "Explain surface finish, weld, thread & GD&T callouts", "icon": "fa-compass-drafting text-amber-400"},
            {"id": "identify_manufacturing_features", "label": "Identify Manufacturing Features", "description": "Note holes, fillets, chamfers, slots & machining needs", "icon": "fa-industry text-rose-400"},
        ],
    },
    "graph_chart": {
        "id": "graph_chart",
        "name": "Graph / Chart / Plot",
        "actions": [
            {"id": "interpret_graph", "label": "Interpret the Graph", "description": "Identify axes, units, variables & core message", "icon": "fa-chart-line text-brand-400"},
            {"id": "extract_data_points", "label": "Extract Data Points", "description": "Tabulate visible coordinates, peaks & intercepts", "icon": "fa-table text-emerald-400"},
            {"id": "explain_trends", "label": "Explain Trends and Patterns", "description": "Analyze growth, decay, anomalies & correlations", "icon": "fa-arrow-trend-up text-indigo-400"},
            {"id": "calculate_slope", "label": "Calculate Slope / Area if Applicable", "description": "Estimate rate of change or area under the curve", "icon": "fa-superscript text-amber-400"},
            {"id": "compare_values", "label": "Compare Values", "description": "Compare series, categories, maxima & minima", "icon": "fa-chart-bar text-rose-400"},
        ],
    },
    "handwritten_notes": {
        "id": "handwritten_notes",
        "name": "Handwritten Notes",
        "actions": [
            {"id": "transcribe_handwriting", "label": "Transcribe Handwriting", "description": "Convert handwritten text into clean digital Markdown", "icon": "fa-file-signature text-brand-400"},
            {"id": "summarize_notes", "label": "Summarize Notes", "description": "Condense into a high-yield executive summary", "icon": "fa-compress text-emerald-400"},
            {"id": "explain_key_concepts", "label": "Explain Key Concepts", "description": "Clarify and expand on topics mentioned in notes", "icon": "fa-lightbulb text-indigo-400"},
            {"id": "generate_study_questions", "label": "Generate Study Questions", "description": "Create active recall questions from these notes", "icon": "fa-circle-question text-amber-400"},
            {"id": "convert_structured_notes", "label": "Convert into Structured Notes", "description": "Organize into headings, bullet points & definitions", "icon": "fa-book-open text-rose-400"},
        ],
    },
    "document_page": {
        "id": "document_page",
        "name": "Document / Page",
        "actions": [
            {"id": "extract_visible_text", "label": "Extract Visible Text", "description": "Perform clean OCR of all printed text on the page", "icon": "fa-file-lines text-brand-400"},
            {"id": "summarize_document", "label": "Summarize Document", "description": "Summarize main points, headings & conclusions", "icon": "fa-align-left text-emerald-400"},
            {"id": "explain_important_sections", "label": "Explain Important Sections", "description": "Break down complex paragraphs, rules or clauses", "icon": "fa-graduation-cap text-indigo-400"},
            {"id": "extract_tables", "label": "Extract Tables or Key Data", "description": "Convert visible tables & figures into Markdown", "icon": "fa-table-cells text-amber-400"},
            {"id": "generate_questions", "label": "Generate Questions", "description": "Create review questions based on this page", "icon": "fa-list-check text-rose-400"},
        ],
    },
    "code_terminal": {
        "id": "code_terminal",
        "name": "Code / Terminal",
        "actions": [
            {"id": "explain_code", "label": "Explain Code", "description": "Transcribe and explain logic, flow & functions", "icon": "fa-code text-brand-400"},
            {"id": "identify_errors", "label": "Identify Errors", "description": "Locate syntax errors, exceptions or stack trace causes", "icon": "fa-bug text-rose-400"},
            {"id": "debug_code", "label": "Debug the Code", "description": "Diagnose root cause and explain how to fix it", "icon": "fa-screwdriver-wrench text-amber-400"},
            {"id": "suggest_improvements", "label": "Suggest Improvements", "description": "Recommend cleaner patterns, edge cases & optimizations", "icon": "fa-gauge-high text-indigo-400"},
            {"id": "generate_corrected_code", "label": "Generate Corrected Code", "description": "Output clean, copyable, working source code", "icon": "fa-terminal text-emerald-400"},
        ],
    },
    "biology_diagram": {
        "id": "biology_diagram",
        "name": "Biology / Scientific Diagram",
        "actions": [
            {"id": "identify_structures", "label": "Identify Biological / Scientific Structures", "description": "Identify organelles, anatomical parts or pathways", "icon": "fa-dna text-brand-400"},
            {"id": "explain_processes", "label": "Explain Biological / Scientific Processes", "description": "Walk through the physiological or scientific cycle", "icon": "fa-arrows-spin text-emerald-400"},
            {"id": "label_components", "label": "Extract and Explain Labels", "description": "List all visible labels and define their functions", "icon": "fa-tags text-indigo-400"},
            {"id": "explain_relationships", "label": "Explain Functional Relationships", "description": "Explain how the structures interact as a system", "icon": "fa-diagram-project text-amber-400"},
            {"id": "generate_study_questions", "label": "Generate Study Questions", "description": "Create exam-style review questions from this diagram", "icon": "fa-circle-question text-rose-400"},
        ],
    },
    "general_object": {
        "id": "general_object",
        "name": "General Object / Scene",
        "actions": [
            {"id": "describe_scene", "label": "Describe the Image", "description": "Detailed visual description of objects & context", "icon": "fa-eye text-brand-400"},
            {"id": "identify_objects", "label": "Identify Objects or Elements", "description": "Catalog distinct items, equipment or landmarks", "icon": "fa-cubes text-emerald-400"},
            {"id": "component_breakdown", "label": "Break Down Visible Components", "description": "Analyze physical parts, materials & arrangement", "icon": "fa-magnifying-glass-location text-indigo-400"},
            {"id": "read_visible_text", "label": "Extract Visible Labels or Text", "description": "Read signs, serial numbers, markings or captions", "icon": "fa-font text-amber-400"},
            {"id": "explain_practical_context", "label": "Answer a Question About the Image", "description": "Explain real-world function, context & key observations", "icon": "fa-lightbulb text-rose-400"},
        ],
    },
    "other": {
        "id": "other",
        "name": "Other / Unknown",
        "actions": [
            {"id": "describe_scene", "label": "Describe Visual Content", "description": "Comprehensive overview of everything visible", "icon": "fa-eye text-brand-400"},
            {"id": "read_visible_text", "label": "Extract All Visible Text & Labels", "description": "Transcribe any text, numbers or symbols", "icon": "fa-font text-emerald-400"},
            {"id": "component_breakdown", "label": "Analyze Structure & Elements", "description": "Break down the key visual elements and layout", "icon": "fa-sitemap text-indigo-400"},
            {"id": "explain_practical_context", "label": "Explain Purpose & Context", "description": "Interpret the meaning and context of the visual", "icon": "fa-lightbulb text-amber-400"},
        ],
    },
}


def validate_and_parse_image_payload(
    image_input: str, explicit_mime: Optional[str] = None
) -> Dict[str, Any]:
    if not image_input or not isinstance(image_input, str) or not image_input.strip():
        raise ValueError("No image payload provided. Upload or capture an image first.")

    trimmed = image_input.strip()
    mime_type = explicit_mime or "image/png"
    b64_data = trimmed

    if trimmed.startswith("data:"):
        m = re.match(r"^data:([^;]+);base64,(.+)$", trimmed, re.S)
        if not m:
            raise ValueError("Invalid Data URL image format.")
        mime_type = m.group(1).strip().lower()
        b64_data = m.group(2).strip()

    b64_clean = re.sub(r"\s+", "", b64_data)
    try:
        raw_bytes = base64.b64decode(b64_clean)
    except Exception:
        raise ValueError("Image base64 payload could not be decoded.")

    if len(raw_bytes) < 16:
        raise ValueError("Captured image is empty or corrupted (< 16 bytes).")

    width: Optional[int] = None
    height: Optional[int] = None

    # Inspect PNG header
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n" and len(raw_bytes) >= 24:
        mime_type = "image/png"
        width, height = struct.unpack(">II", raw_bytes[16:24])
    # Inspect JPEG header
    elif raw_bytes[:2] == b"\xff\xd8":
        mime_type = "image/jpeg"
    # Inspect GIF header
    elif raw_bytes[:6] in (b"GIF87a", b"GIF89a") and len(raw_bytes) >= 10:
        mime_type = "image/gif"
        width, height = struct.unpack("<HH", raw_bytes[6:10])
    # Inspect WEBP header
    elif raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"

    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
    data_url = f"data:{mime_type};base64,{b64_clean}"

    return {
        "data_url": data_url,
        "base64_data": b64_clean,
        "mime_type": mime_type,
        "byte_length": len(raw_bytes),
        "width": width,
        "height": height,
        "sha256": sha256_hash,
    }


def classify_vision_image(params: Dict[str, Any]) -> Dict[str, Any]:
    validated = validate_and_parse_image_payload(
        params.get("image") or "", params.get("mime_type")
    )
    provider = params.get("provider") or ai_manager.active_provider
    model = params.get("model") or ai_manager.active_model
    base_url = params.get("base_url") or ai_manager.local_endpoint
    manual_cat = (params.get("manual_category") or "").strip()

    if manual_cat and manual_cat != "auto" and manual_cat in VISION_CATEGORIES:
        spec = VISION_CATEGORIES[manual_cat]
        return {
            "success": True,
            "classification": {
                "category_id": spec["id"],
                "category": spec["name"],
                "subcategory": "Manual Selection",
                "confidence": 1.0,
                "confidence_label": "High",
                "is_ambiguous": False,
                "alternative_categories": [],
                "description": f"Visual category manually set to {spec['name']}.",
            },
            "detected_content": [spec["name"]],
            "recommended_actions": spec["actions"],
            "image_metadata": {
                "mime_type": validated["mime_type"],
                "byte_length": validated["byte_length"],
                "width": validated["width"],
                "height": validated["height"],
                "sha256": validated["sha256"],
            },
            "provider": provider,
            "model": model,
        }

    allowed_ids = ", ".join(VISION_CATEGORIES.keys())
    prompt = f"""Inspect this image carefully and classify its primary domain into ONE of these exact category_id values:
{allowed_ids}

Return ONLY a valid JSON object with this structure:
{{
  "category_id": "one of the allowed category_id values",
  "subcategory": "Specific subject (e.g. Calculus Integral, RC Filter Schematic, Roof Truss, Python Traceback, Handwritten Physics Notes)",
  "confidence": 0.92,
  "description": "1-2 sentence factual summary of what is visibly present in the image",
  "detected_content": ["visible item 1", "visible label/formula 2", "visible element 3"],
  "alternative_category_ids": []
}}"""

    cat_id = ""
    subcat = ""
    confidence = 0.0
    description = ""
    detected_content: List[str] = []
    alt_ids: List[str] = []

    try:
        res = ai_manager.generate_text(
            messages=[
                {
                    "role": "system",
                    "content": "You are VISION AI Multimodal Classifier. Output valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        {
                            "dataUrl": validated["data_url"],
                            "mimeType": validated["mime_type"],
                        }
                    ],
                },
            ],
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=0.1,
            max_tokens=1024,
            json_mode=True,
            operation="vision_classification",
            allow_vision_fallback=True,
            allow_model_fallback=True,
        )
        effective_provider = res.get("provider") or provider
        effective_model = res.get("model") or model
        routed_vision = bool(res.get("routed_vision"))
        routed_model_name = (
            res.get("routed_vision_model_name")
            or res.get("vision_model_name")
            or effective_model
        )
        routing_notice = res.get("vision_routing_notice") or ""
        parsed = parse_json_safely(res.get("text", ""))
        if not isinstance(parsed, dict):
            raise RuntimeError("The vision provider returned an invalid classification response.")
        raw_cat = str(parsed.get("category_id") or "").strip().lower()
        if raw_cat not in VISION_CATEGORIES:
            raise RuntimeError("The vision provider returned an unsupported image category.")
        cat_id = raw_cat
        subcat = str(parsed.get("subcategory") or VISION_CATEGORIES[cat_id]["name"]).strip()
        try:
            confidence = max(0.1, min(1.0, float(parsed.get("confidence", 0.9))))
        except (TypeError, ValueError):
            confidence = 0.9
        description = str(parsed.get("description") or "").strip()
        if isinstance(parsed.get("detected_content"), list):
            detected_content = [
                str(x).strip() for x in parsed["detected_content"] if str(x).strip()
            ][:8]
        if isinstance(parsed.get("alternative_category_ids"), list):
            alt_ids = [
                str(x).strip()
                for x in parsed["alternative_category_ids"]
                if str(x).strip() in VISION_CATEGORIES and str(x).strip() != cat_id
            ][:3]
    except ProviderRequestError:
        logger.exception("Vision classification provider request failed")
        raise
    except Exception as exc:
        logger.exception("Vision classification failed")
        raise RuntimeError(f"Image classification failed: {sanitize_log_text(str(exc))}") from exc

    spec = VISION_CATEGORIES.get(cat_id, VISION_CATEGORIES["other"])
    conf_label = "High" if confidence >= 0.8 else ("Medium" if confidence >= 0.55 else "Low")

    return {
        "success": True,
        "classification": {
            "category_id": cat_id,
            "category": spec["name"],
            "subcategory": subcat,
            "confidence": confidence,
            "confidence_label": conf_label,
            "is_ambiguous": confidence < 0.7,
            "alternative_categories": [
                {"category_id": aid, "category": VISION_CATEGORIES[aid]["name"]}
                for aid in alt_ids
            ],
            "description": description,
        },
        "detected_content": detected_content,
        "recommended_actions": spec["actions"],
        "image_metadata": {
            "mime_type": validated["mime_type"],
            "byte_length": validated["byte_length"],
            "width": validated["width"],
            "height": validated["height"],
            "sha256": validated["sha256"],
        },
        "provider": effective_provider,
        "model": effective_model,
        "original_provider": provider,
        "original_model": model,
        "routed_vision": routed_vision,
        "routed_model": effective_model,
        "routed_model_name": routed_model_name,
        "routing_notice": routing_notice,
        "selection_mode": res.get("vision_selection_mode", "automatic"),
        "selection_reason": res.get("vision_selection_reason"),
        "model_fallback": bool(res.get("model_fallback")),
        "model_fallback_reason": res.get("model_fallback_reason"),
    }


def analyze_vision_image(params: Dict[str, Any]) -> Dict[str, Any]:
    validated = validate_and_parse_image_payload(
        params.get("image") or "", params.get("mime_type")
    )
    provider = params.get("provider") or ai_manager.active_provider
    model = params.get("model") or ai_manager.active_model
    base_url = params.get("base_url") or ai_manager.local_endpoint
    category_id = str(params.get("category_id") or params.get("category") or "other").strip().lower()
    if category_id not in VISION_CATEGORIES:
        category_id = "other"

    spec = VISION_CATEGORIES[category_id]
    action_id = str(params.get("action_id") or params.get("action") or "describe_scene").strip()
    custom_prompt = str(params.get("custom_prompt") or params.get("prompt") or "").strip()

    action_obj = next(
        (a for a in spec["actions"] if a["id"] == action_id),
        spec["actions"][0] if spec["actions"] else {"id": action_id, "label": "Visual Analysis", "description": ""},
    )
    action_title = custom_prompt[:60] if custom_prompt else action_obj["label"]

    analysis_prompt = f"""Target Domain: {spec['name']}
Requested Analysis Action: {action_title}
{f'User Instruction: {custom_prompt}' if custom_prompt else f'Task Description: {action_obj.get("description", "")}'}

Perform a thorough, accurate visual inspection of the attached image:
### 1. Observed Visual Information
List strictly what is directly visible and legible in the image (labels, text, equations, components, axes, or layout). Do NOT invent any unseen values.

### 2. {action_title}
Provide the complete step-by-step analysis, transcription (using clean LaTeX $$ ... $$ for equations), derivation, or explanation requested.

### 3. Key Takeaways & Verification Notes
Summarize the main conclusion and note if any region of the image is ambiguous or cropped."""

    res = ai_manager.generate_text(
        messages=[
            {
                "role": "system",
                "content": f"You are VISION AI Multimodal Inspector specialized in {spec['name']}. Separate observed visual facts from analytical interpretation.",
            },
            {
                "role": "user",
                "content": analysis_prompt,
                "images": [
                    {
                        "dataUrl": validated["data_url"],
                        "mimeType": validated["mime_type"],
                    }
                ],
            },
        ],
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=0.25,
        max_tokens=4096,
        operation="vision_analysis",
        allow_vision_fallback=True,
        allow_model_fallback=True,
    )

    text_out = res.get("text", "").strip()
    effective_provider = res.get("provider") or provider
    effective_model = res.get("model") or model
    routed_vision = bool(res.get("routed_vision"))
    routed_model_name = (
        res.get("routed_vision_model_name")
        or res.get("vision_model_name")
        or effective_model
    )
    routing_notice = res.get("vision_routing_notice") or ""
    return {
        "success": True,
        "category_id": category_id,
        "category": spec["name"],
        "action_id": action_id,
        "action_title": action_title,
        "analysis": text_out,
        "solution": text_out,
        "provider": effective_provider,
        "model": effective_model,
        "original_provider": provider,
        "original_model": model,
        "routed_vision": routed_vision,
        "routed_model": effective_model,
        "routed_model_name": routed_model_name,
        "routing_notice": routing_notice,
        "selection_mode": res.get("vision_selection_mode", "automatic"),
        "selection_reason": res.get("vision_selection_reason"),
        "model_fallback": bool(res.get("model_fallback")),
        "model_fallback_reason": res.get("model_fallback_reason"),
    }
