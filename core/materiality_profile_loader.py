"""
Materiality profile loader with SASB-style dataset ingestion.

Supports:
1. Local JSON profile map (existing behavior)
2. Remote JSON overlay (profiles map)
3. Remote SASB-like CSV/JSON dataset converted into E/S/G weights
"""

from __future__ import annotations

from typing import Any, Dict, List
import csv
import io
import json
import logging
import os

import requests


logger = logging.getLogger(__name__)


INDUSTRY_ALIASES = {
    "oil & gas": "oil_and_gas",
    "oil and gas": "oil_and_gas",
    "coal": "coal",
    "metals and mining": "mining",
    "mining": "mining",
    "airlines": "aviation",
    "aviation": "aviation",
    "banks": "banking",
    "commercial banks": "banking",
    "banking": "banking",
    "consumer goods": "consumer_goods",
    "consumer staples": "consumer_goods",
    "food & beverage": "food_beverage",
    "food and beverage": "food_beverage",
    "technology": "technology",
    "software": "software",
}


def _normalize_industry_key(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "general"
    text = text.replace("&", " and ")
    text = " ".join(text.split())
    if text in INDUSTRY_ALIASES:
        return INDUSTRY_ALIASES[text]
    return text.replace(" ", "_").replace("-", "_")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_triplet(e: float, s: float, g: float) -> Dict[str, float]:
    e = max(0.01, float(e))
    s = max(0.01, float(s))
    g = max(0.01, float(g))
    total = e + s + g
    return {
        "E": round(e / total, 4),
        "S": round(s / total, 4),
        "G": round(g / total, 4),
    }


def _extract_topics(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    # Accept comma/semicolon/pipe separated values
    topics = []
    for sep in [";", "|", ","]:
        if sep in text:
            topics = [t.strip() for t in text.split(sep) if t.strip()]
            break
    if not topics:
        topics = [text]
    return topics


def _build_profile_from_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    industry_raw = row.get("industry") or row.get("sector") or row.get("industry_name")
    industry = _normalize_industry_key(industry_raw)
    if not industry:
        return None

    e = _safe_float(row.get("E", row.get("environmental", row.get("environment_score", 0.35))), 0.35)
    s = _safe_float(row.get("S", row.get("social", row.get("social_score", 0.30))), 0.30)
    g = _safe_float(row.get("G", row.get("governance", row.get("governance_score", 0.35))), 0.35)
    weights = _normalize_triplet(e, s, g)

    rationale = str(
        row.get("rationale")
        or row.get("notes")
        or row.get("description")
        or "SASB-style materiality profile ingestion."
    ).strip()
    topics = _extract_topics(row.get("material_topics") or row.get("topics") or row.get("kpis"))
    source = str(row.get("source") or "SASB-style dataset").strip()

    return {
        "industry": industry,
        "profile": {
            "weights": weights,
            "rationale": rationale,
            "material_topics": topics,
            "source": source,
        },
    }


def parse_sasb_like_dataset(payload: Any) -> Dict[str, Any]:
    """
    Parse SASB-like CSV/JSON payload into internal profile map format.
    """
    profile_map: Dict[str, Any] = {}

    if isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("profiles"), dict):
            # Already profile map compatible
            return payload.get("profiles", {})
        if isinstance(payload.get("data"), list):
            rows = [r for r in payload.get("data", []) if isinstance(r, dict)]
        else:
            rows = []
    else:
        rows = []

    for row in rows:
        built = _build_profile_from_row(row)
        if not built:
            continue
        profile_map[built["industry"]] = built["profile"]

    return profile_map


def parse_sasb_csv_text(csv_text: str) -> Dict[str, Any]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = [dict(r) for r in reader]
    return parse_sasb_like_dataset(rows)


def load_materiality_profiles(
    local_path: str,
    remote_profile_url: str = "",
    sasb_dataset_url: str = "",
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Load and merge materiality profiles in precedence order:
    local -> remote profile overlay -> SASB-like dataset overlay.
    """
    base: Dict[str, Any] = {}

    try:
        if local_path and os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                base = loaded
    except Exception as exc:
        logger.warning("Failed loading local materiality profiles (%s): %s", local_path, exc)

    # Remote profile overlay
    if remote_profile_url:
        try:
            resp = requests.get(remote_profile_url, timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict):
                    overlay = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else payload
                    if isinstance(overlay, dict):
                        base.update(overlay)
                        logger.info("Applied remote profile overlay: %s entries", len(overlay))
        except Exception as exc:
            logger.warning("Remote profile overlay failed: %s", exc)

    # SASB-style dataset overlay (CSV or JSON)
    if sasb_dataset_url:
        try:
            resp = requests.get(sasb_dataset_url, timeout=timeout)
            if resp.status_code == 200:
                content_type = str(resp.headers.get("Content-Type", "")).lower()
                sasb_profiles: Dict[str, Any] = {}
                if "json" in content_type:
                    sasb_profiles = parse_sasb_like_dataset(resp.json())
                else:
                    # Try csv first, then json fallback
                    text = resp.text or ""
                    sasb_profiles = parse_sasb_csv_text(text)
                    if not sasb_profiles:
                        try:
                            sasb_profiles = parse_sasb_like_dataset(json.loads(text))
                        except Exception:
                            pass
                if sasb_profiles:
                    base.update(sasb_profiles)
                    logger.info("Applied SASB-style dataset overlay: %s entries", len(sasb_profiles))
        except Exception as exc:
            logger.warning("SASB-style dataset overlay failed: %s", exc)

    if "general" not in base:
        base["general"] = {
            "weights": {"E": 0.35, "S": 0.30, "G": 0.35},
            "rationale": "Balanced fallback materiality profile.",
            "material_topics": [],
            "source": "fallback",
        }

    return base

