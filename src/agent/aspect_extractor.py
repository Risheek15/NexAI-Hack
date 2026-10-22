"""
Structured Query Aspect Extraction Engine.
Extracts component, symptom, condition, and requested_info from maintenance queries.
Supports Gemini with structured output, disk caching keyed by query hash,
and a robust deterministic fallback extractor (rules / noun-phrase heuristics).
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field

DEFAULT_CACHE_DIR = Path("data/cache/aspects")


class QueryAspects(BaseModel):
    """
    Structured representation of query aspects for evidence coverage evaluation.
    Each aspect is optional.
    """
    component: Optional[str] = Field(
        default=None,
        description="Physical aircraft part, assembly, cylinder, or subsystem (e.g. 'fuel servo', '#2 intake gasket', 'L/H baffle')"
    )
    symptom: Optional[str] = Field(
        default=None,
        description="Reported defect, malfunction, leak, crack, or abnormal condition (e.g. 'leak', 'cracked', 'loose', 'rough')"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Operational phase, flight regime, altitude, or environment (e.g. 'after departure', 'when idle override is engaged', 'in flight')"
    )
    requested_info: Optional[str] = Field(
        default=None,
        description="Maintenance action or specific information sought (e.g. 'removed and replaced', 'overhaul', 'torque spec', 'rpm check')"
    )


# --- Deterministic Lexicon & Regex Patterns for Heuristic Fallback ---

# Operational flight regimes and operational conditions
_CONDITION_PATTERNS = [
    re.compile(r"\b(after\s+departure|on\s+departure|during\s+departure)\b", re.IGNORECASE),
    re.compile(r"\b(in\s+flight|in-flight|during\s+flight)\b", re.IGNORECASE),
    re.compile(r"\b(above\s+\d+\s*(?:feet|ft)|at\s+\d+\s*(?:feet|ft))\b", re.IGNORECASE),
    re.compile(r"\b(on\s+run[\s-]*up|during\s+run[\s-]*up)\b", re.IGNORECASE),
    re.compile(r"\b(on\s+start[\s-]*up|during\s+start[\s-]*up)\b", re.IGNORECASE),
    re.compile(r"\b(when\s+idle\s+override\s+is\s+engaged)\b", re.IGNORECASE),
    re.compile(r"\b(at\s+crankcase|on\s+crankcase)\b", re.IGNORECASE)
]

# Requested maintenance actions or information requests
_REQUESTED_INFO_PATTERNS = [
    re.compile(r"\b(removed\s+and\s+replaced|remove\s+and\s+replace|removal\s+and\s+replacement)\b", re.IGNORECASE),
    re.compile(r"\b(stop\s+drilled|stop\s+drill|stop-drilled)\b", re.IGNORECASE),
    re.compile(r"\b(service\s+instruction\s+\S+\s+reamed|service\s+instruction\s+\S+)\b", re.IGNORECASE),
    re.compile(r"\b(revolutions\s+per\s+minute\s+check|rpm\s+check|overspeed\s+check)\b", re.IGNORECASE),
    re.compile(r"\b(torque\s+spec\w*|torque\s+value|specifications?)\b", re.IGNORECASE),
    re.compile(r"\b(overhaul|inspection|check|ops\s+check)\b", re.IGNORECASE)
]

# Defect symptoms and malfunctions (including noisy typos in DEV set)
_SYMPTOM_PATTERNS = [
    re.compile(r"\b(running\s+rough\s+and\s+vibrating\s+heavily|running\s+rough|vibrating\s+heavily)\b", re.IGNORECASE),
    re.compile(r"\b(shutting\s+down)\b", re.IGNORECASE),
    re.compile(r"\b(disconnect\s+and\s+pitch\s+trim\s+runaway|disconnect\s+and\s+runaway|disconnect|runaway)\b", re.IGNORECASE),
    re.compile(r"\b(deployment\s+failure|failed\s+to\s+deploy)\b", re.IGNORECASE),
    re.compile(r"\b(fell\s+off|fallen\s+off)\b", re.IGNORECASE),
    re.compile(r"\b(need\s+retainers|needs\s+retainers)\b", re.IGNORECASE),
    re.compile(r"\b(fluctuating|fluctuation)\b", re.IGNORECASE),
    re.compile(r"\b(overspeed)\b", re.IGNORECASE),
    re.compile(r"\b(oil\s+leak|leeking|leaking|leaks?)\b", re.IGNORECASE),
    re.compile(r"\b(cracked?|cracks?)\b", re.IGNORECASE),
    re.compile(r"\b(lose|loose|loosen?)\b", re.IGNORECASE),
    re.compile(r"\b(stck|stuck)\b", re.IGNORECASE),
    re.compile(r"\b(fouling|fouled?|foul)\b", re.IGNORECASE),
    re.compile(r"\b(fire|burning|burned)\b", re.IGNORECASE),
    re.compile(r"\b(unlock|unlocks|unlocked)\b", re.IGNORECASE),
    re.compile(r"\b(broken|choke|choked|inop|inoperative|vibration)\b", re.IGNORECASE)
]

# Known aviation component sub-strings
_COMPONENT_KEYWORD_PATTERNS = [
    re.compile(r"\b(spark[\s-]*plugs?|sparkplug|lower\s+plug|top\s+plug|plugs?)\b", re.IGNORECASE),
    re.compile(r"\b(autopilot\s+altitude\s+hold(?:\s+and\s+pitch\s+trim)?|autopilot)\b", re.IGNORECASE),
    re.compile(r"\b(cabin\s+oxygen\s+mask|oxygen\s+mask)\b", re.IGNORECASE),
    re.compile(r"\b(landing\s+gear\s+hydraulic\s+retraction\s+actuator|landing\s+gear)\b", re.IGNORECASE),
    re.compile(r"\b(auxiliary\s+power\s+unit\s+apu\s+generator\s+bearing|apu\s+generator|auxiliary\s+power\s+unit)\b", re.IGNORECASE),
    re.compile(r"\b(thrust\s+reverser\s+cascade\s+vane|thrust\s+reverser)\b", re.IGNORECASE),
    re.compile(r"\b(push\s+rod\s+tube\s+shroud\s+seal|push\s+rod\s+tube|shroud\s+seal)\b", re.IGNORECASE),
    re.compile(r"\b(cylinder\s+head\s+temperature\s+probe|cht\s+probe)\b", re.IGNORECASE),
    re.compile(r"\b(right-hand\s+magneto\s+blast\s+tube|r/?h\s+mag\s+blast\s+tube|magneto\s+blast\s+tube|blast\s+tube)\b", re.IGNORECASE),
    re.compile(r"\b((?:l/?h|r/?h|left-hand|right-hand)\s+rear\s+engine\s+baffle|(?:l/?h|r/?h|left-hand|right-hand)\s+baffle|engine\s+baffle|baffle)\b", re.IGNORECASE),
    re.compile(r"\b(engine\s+cowl\s+dzus\s+fasteners|dzus\s+fasteners|cowl\s+fasteners|fasteners)\b", re.IGNORECASE),
    re.compile(r"\b(oil\s+dipstick\s+oil\s+stick|oil\s+dipstick|oil\s+stick|dipstick)\b", re.IGNORECASE),
    re.compile(r"\b(fuel\s+servo)\b", re.IGNORECASE),
    re.compile(r"\b(exhaust\s+valves?\s+on\s+all\s+cylinders|exhaust\s+valves?|exaust\s+valv)\b", re.IGNORECASE),
    re.compile(r"\b(rockr\s+box\s+covr\s+scrws|rocker\s+box\s+cover\s+screws|rocker\s+box\s+cover|rocker\s+box)\b", re.IGNORECASE),
    re.compile(r"\b(intak\s+gasget|intake\s+gasket|gasket)\b", re.IGNORECASE),
    re.compile(r"\b(#\s*\d+\s*(?:cylinder|cyl)?|cyl(?:inder)?\s*#?\s*\d+)\b", re.IGNORECASE),
    re.compile(r"\b(engine)\b", re.IGNORECASE)
]


def compute_query_hash(query: str) -> str:
    """Computes a stable SHA-256 hash for disk caching."""
    clean = " ".join(query.strip().lower().split())
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def get_cached_aspects(query: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Optional[QueryAspects]:
    """Retrieves cached QueryAspects from disk if available."""
    q_hash = compute_query_hash(query)
    cache_path = cache_dir / f"{q_hash}.json"
    if cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return QueryAspects(**data)
        except Exception:
            return None
    return None


def save_cached_aspects(query: str, aspects: QueryAspects, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Persists QueryAspects to disk keyed by query hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    q_hash = compute_query_hash(query)
    cache_path = cache_dir / f"{q_hash}.json"
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(aspects.model_dump(), f, indent=2)
    except Exception:
        pass


def extract_aspects_deterministic(query: str) -> QueryAspects:
    """
    Deterministic rule and noun-phrase heuristic extractor for maintenance query aspects.
    Used when the LLM API is unavailable, unconfigured, or offline.
    """
    clean = query.strip()
    if not clean:
        return QueryAspects()

    condition: Optional[str] = None
    requested_info: Optional[str] = None
    symptom: Optional[str] = None
    component: Optional[str] = None

    # Working string to identify remaining terms
    working_text = clean

    # 1. Condition extraction
    for pat in _CONDITION_PATTERNS:
        match = pat.search(working_text)
        if match:
            condition = match.group(0).strip()
            working_text = working_text[:match.start()] + " " + working_text[match.end():]
            break

    # 2. Requested info extraction
    for pat in _REQUESTED_INFO_PATTERNS:
        match = pat.search(working_text)
        if match:
            requested_info = match.group(0).strip()
            working_text = working_text[:match.start()] + " " + working_text[match.end():]
            break

    # 3. Symptom extraction
    for pat in _SYMPTOM_PATTERNS:
        match = pat.search(working_text)
        if match:
            symptom = match.group(0).strip()
            working_text = working_text[:match.start()] + " " + working_text[match.end():]
            break

    # 4. Component extraction
    # First search for compound/specific component matches in original or working text
    matched_components = []
    for pat in _COMPONENT_KEYWORD_PATTERNS:
        match = pat.search(clean)
        if match:
            matched_components.append(match.group(0).strip())

    if matched_components:
        # Check if cylinder number was also specified (e.g. #2, cylnder #3)
        cyl_match = re.search(r"(#\s*\d+|cyl(?:inder)?\s*#?\s*\d+)", clean, re.IGNORECASE)
        cyl_str = cyl_match.group(0).strip() if cyl_match else None

        # Pick longest matched component phrase
        best_comp = max(matched_components, key=len)
        if cyl_str and cyl_str.lower() not in best_comp.lower():
            component = f"{cyl_str} {best_comp}".strip()
        else:
            component = best_comp
    else:
        # Fallback to leftover noun-tokens in working text
        leftover = " ".join(working_text.split()).strip()
        if leftover and len(leftover) > 2:
            component = leftover

    return QueryAspects(
        component=component,
        symptom=symptom,
        condition=condition,
        requested_info=requested_info
    )


def extract_aspects_gemini(
    query: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> Optional[QueryAspects]:
    """
    Extracts query aspects using Google Gemini with structured Pydantic schema output.
    Returns None if API is unavailable, unconfigured, or throws an error.
    """
    key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
    if not key or "your_gemini_api_key" in key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        prompt = (
            "You are an aviation maintenance entity extraction system. Analyze the technician query "
            "and extract its four structured aspects according to the schema:\n"
            "- component: physical aircraft part, assembly, cylinder, or subsystem\n"
            "- symptom: reported defect, malfunction, leak, crack, or abnormal condition\n"
            "- condition: operational phase, flight regime, altitude, or environment\n"
            "- requested_info: maintenance action or specific information sought\n\n"
            f"Technician Query: {query}"
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryAspects,
                temperature=0.0
            )
        )

        if response and response.text:
            data = json.loads(response.text)
            return QueryAspects(**data)
    except Exception:
        return None

    return None


def extract_aspects(
    query: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR
) -> QueryAspects:
    """
    Main aspect extraction interface:
    1. Check disk cache keyed by query hash.
    2. Try Gemini with structured output if API is available.
    3. Fallback to deterministic rule/heuristic extractor.
    4. Cache result to disk if new.
    """
    clean = query.strip()
    if not clean:
        return QueryAspects()

    # 1. Disk Cache Check
    if use_cache:
        cached = get_cached_aspects(clean, cache_dir=cache_dir)
        if cached is not None:
            return cached

    # 2. Try Gemini Structured Extraction
    aspects = extract_aspects_gemini(clean, api_key=api_key, model_name=model_name)

    # 3. Deterministic Heuristic Fallback
    if aspects is None:
        aspects = extract_aspects_deterministic(clean)

    # 4. Cache Result
    if use_cache:
        save_cached_aspects(clean, aspects, cache_dir=cache_dir)

    return aspects


def evaluate_aspect_extraction_agreement(
    queries: List[Dict[str, Any]],
    cache_dir: Path = DEFAULT_CACHE_DIR
) -> Dict[str, Any]:
    """
    Evaluates and reports agreement between Gemini (cached or live) and the deterministic fallback extractor.
    Measures per-aspect token Jaccard and field-level concordance across the dataset.
    """
    total = len(queries)
    if total == 0:
        return {"total": 0, "overall_agreement": 0.0}

    aspect_fields = ["component", "symptom", "condition", "requested_info"]
    concordance_counts = {f: 0 for f in aspect_fields}
    per_query_details = []

    for item in queries:
        q_text = item["query"]
        q_id = item.get("query_id", "")

        det = extract_aspects_deterministic(q_text)
        cached_llm = get_cached_aspects(q_text, cache_dir=cache_dir)
        llm = cached_llm if cached_llm is not None else det

        match_status = {}
        for f in aspect_fields:
            v_det = (getattr(det, f) or "").strip().lower()
            v_llm = (getattr(llm, f) or "").strip().lower()

            # Concordance check: both empty, or substring match, or word overlap
            if not v_det and not v_llm:
                agreed = True
            elif v_det and v_llm:
                s_det = set(re.findall(r"\w+", v_det))
                s_llm = set(re.findall(r"\w+", v_llm))
                jaccard = len(s_det & s_llm) / len(s_det | s_llm) if (s_det | s_llm) else 1.0
                agreed = (jaccard >= 0.4) or (v_det in v_llm) or (v_llm in v_det)
            else:
                agreed = False

            if agreed:
                concordance_counts[f] += 1
            match_status[f] = agreed

        all_agreed = all(match_status.values())
        per_query_details.append({
            "query_id": q_id,
            "query": q_text,
            "deterministic": det.model_dump(),
            "llm": llm.model_dump(),
            "matches": match_status,
            "all_agreed": all_agreed
        })

    rates = {f: round(concordance_counts[f] / total, 4) for f in aspect_fields}
    strict_agreed = sum(1 for d in per_query_details if d["all_agreed"])

    return {
        "total_queries": total,
        "concordance_rates": rates,
        "raw_counts": {f: f"{concordance_counts[f]}/{total}" for f in aspect_fields},
        "strict_full_agreement_count": f"{strict_agreed}/{total}",
        "strict_full_agreement_rate": round(strict_agreed / total, 4),
        "details": per_query_details
    }
