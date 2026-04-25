"""
Claim Intensity Scorer — Phase 3 of the Risk Scoring Overhaul
=============================================================

Computes **C** (Claim Intensity Score, 0-100) from decomposed sub-claims.

C measures how strong, specific, and verifiable a company's ESG claims are.
A high C means bold, quantitative, time-bound claims.
A low C means vague, aspirational, or unverifiable claims.

C feeds into the Greenwashing Formula:
    GW = α·max(0, C−P)/σ_industry + β·R + γ·(1−D/100) + δ·T

C is NOT an indicator of wrongdoing by itself.  A high C + high P = legitimate
leadership.  A high C + low P = greenwashing signal.
"""

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Claim type weights
# ---------------------------------------------------------------------------

CLAIM_TYPE_WEIGHTS: Dict[str, float] = {
    "quantitative_target": 1.00,   # Specific numbers, dates, baselines
    "alignment_claim":     0.75,   # Paris Agreement, SBTi, net-zero
    "strategic_claim":     0.40,   # "We are committed to …"
    "marketing_claim":     0.15,   # "We are green / sustainable"
    # Legacy types from ClaimDecomposer that map to the above
    "policy_claim":        0.40,
    "production_claim":    0.60,
}


# ---------------------------------------------------------------------------
# Hedging language patterns
# ---------------------------------------------------------------------------

_HEDGING_TOKENS = [
    "aim to", "hope to", "aspire to", "intend to", "plan to",
    "working toward", "exploring", "considering", "endeavor",
    "may", "might", "could potentially", "where feasible",
    "subject to", "dependent on", "contingent upon",
]

_THIRD_PARTY_VALIDATORS = [
    "sbti", "science based targets", "unfccc", "cdp a list",
    "iso 14001", "iso 50001", "leed", "breeam", "verified",
    "assured by", "third-party assurance", "independently verified",
]


# ---------------------------------------------------------------------------
# Per-sub-claim scoring dimensions
# ---------------------------------------------------------------------------

def _score_specificity(text: str) -> float:
    """0-30 points.  Does the claim have numbers, years, baselines?"""
    score = 0.0
    # Year reference (e.g. "by 2030")
    if re.search(r"\b20[2-9]\d\b", text):
        score += 10
    # Percentage or absolute number
    if re.search(r"\b\d+(?:\.\d+)?%", text) or re.search(r"\b\d{1,3}(?:,\d{3})+\b", text):
        score += 10
    # Baseline reference (e.g. "from 2019 levels")
    if re.search(r"(?:from|against|baseline|vs\.?|compared to)\s+\d{4}", text, re.IGNORECASE):
        score += 10
    return min(30.0, score)


def _score_verifiability(text: str) -> float:
    """0-20 points.  Is there an externally checkable reference?"""
    score = 0.0
    lowered = text.lower()
    # Scope reference
    if re.search(r"scope\s*[123]", lowered):
        score += 8
    # Standard or framework reference
    framework_tokens = ["gri", "sasb", "tcfd", "cdp", "issb", "brsr", "tnfd", "sfdr"]
    if any(tok in lowered for tok in framework_tokens):
        score += 7
    # External body reference
    if any(tok in lowered for tok in ["sbti", "science based", "unfccc", "paris agreement"]):
        score += 5
    return min(20.0, score)


def _score_ambiguity_penalty(text: str) -> float:
    """0 to -20 points.  Hedging language reduces score."""
    lowered = text.lower()
    hits = sum(1 for tok in _HEDGING_TOKENS if tok in lowered)
    return min(20.0, hits * 7.0)  # each hedge costs 7 pts, max 20


def _score_scope_clarity(text: str) -> float:
    """0-15 points.  Does it specify which operations, gases, scope?"""
    score = 0.0
    lowered = text.lower()
    if re.search(r"scope\s*[123]", lowered):
        score += 5
    if any(tok in lowered for tok in ["operations", "supply chain", "value chain", "upstream", "downstream"]):
        score += 5
    if any(tok in lowered for tok in ["co2", "ghg", "methane", "ch4", "sf6", "n2o", "tco2e"]):
        score += 5
    return min(15.0, score)


def _score_third_party(text: str) -> float:
    """0 or 15 bonus points.  Third-party validation reference."""
    lowered = text.lower()
    if any(tok in lowered for tok in _THIRD_PARTY_VALIDATORS):
        return 15.0
    return 0.0


def _classify_claim_type(sub_claim: Dict[str, Any]) -> str:
    """Map a decomposed sub-claim dict to one of the canonical claim types."""
    existing_type = str(sub_claim.get("type", "")).lower()
    text = str(sub_claim.get("text", "")).lower()

    # If the decomposer already assigned a known type, use it
    if existing_type in CLAIM_TYPE_WEIGHTS:
        return existing_type

    # Heuristic classification
    if re.search(r"\b\d+(?:\.\d+)?%", text) or re.search(r"\b20[2-9]\d\b", text):
        return "quantitative_target"
    if any(tok in text for tok in ["net zero", "net-zero", "paris", "sbti", "aligned", "1.5"]):
        return "alignment_claim"
    if any(tok in text for tok in ["committed", "strategy", "roadmap", "transition", "pathway"]):
        return "strategic_claim"
    if any(tok in text for tok in ["green", "sustainable", "eco", "clean", "responsible"]):
        return "marketing_claim"

    return "strategic_claim"  # conservative default


def score_sub_claim(sub_claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a single decomposed sub-claim across all five dimensions.

    Returns a dict with the sub-claim text, type, individual dimension scores,
    the type weight, and the final weighted score.
    """
    text = str(sub_claim.get("text", ""))
    claim_type = _classify_claim_type(sub_claim)
    type_weight = CLAIM_TYPE_WEIGHTS.get(claim_type, 0.40)

    specificity    = _score_specificity(text)
    verifiability  = _score_verifiability(text)
    ambiguity_pen  = _score_ambiguity_penalty(text)
    scope_clarity  = _score_scope_clarity(text)
    third_party    = _score_third_party(text)

    raw_score = specificity + verifiability - ambiguity_pen + scope_clarity + third_party
    # Normalise raw to 0-100 (max possible = 30+20+15+15 = 80, min = -20)
    normalised = max(0.0, min(100.0, (raw_score + 20) * (100.0 / 100.0)))
    weighted = normalised * type_weight

    return {
        "text": text,
        "claim_type": claim_type,
        "type_weight": round(type_weight, 2),
        "specificity": round(specificity, 1),
        "verifiability": round(verifiability, 1),
        "ambiguity_penalty": round(ambiguity_pen, 1),
        "scope_clarity": round(scope_clarity, 1),
        "third_party_bonus": round(third_party, 1),
        "raw_score": round(raw_score, 1),
        "normalised_score": round(normalised, 1),
        "weighted_score": round(weighted, 1),
    }


# ---------------------------------------------------------------------------
# Aggregate scorer
# ---------------------------------------------------------------------------

def calculate_claim_intensity(
    claim_text: str,
    sub_claims: List[Dict[str, Any]],
    *,
    company: str = "",
    industry: str = "",
) -> Dict[str, Any]:
    """
    Compute the aggregate Claim Intensity Score (**C**).

    Parameters
    ----------
    claim_text : str
        The original, un-decomposed claim string.
    sub_claims : list[dict]
        Output of ``ClaimDecomposer.decompose_claim()["sub_claims"]``.
    company : str, optional
        Company name (for logging only).
    industry : str, optional
        Industry sector (for logging only).

    Returns
    -------
    dict
        {
            "claim_intensity_score": float,      # C, 0-100
            "sub_claim_scores": [...],
            "dominant_claim_type": str,
            "specificity_level": str,
            "verifiability_level": str,
            "hedging_detected": bool,
            "third_party_validated": bool,
        }
    """
    if not sub_claims:
        # If no decomposed claims, score the raw claim text as a single claim
        if claim_text:
            sub_claims = [{"text": claim_text, "type": "strategic_claim"}]
        else:
            return {
                "claim_intensity_score": 0.0,
                "sub_claim_scores": [],
                "dominant_claim_type": "none",
                "specificity_level": "none",
                "verifiability_level": "none",
                "hedging_detected": False,
                "third_party_validated": False,
            }

    scored = [score_sub_claim(sc) for sc in sub_claims]

    # Aggregate: weighted average across sub-claims
    total_weight = sum(s["type_weight"] for s in scored) or 1.0
    C = sum(s["weighted_score"] for s in scored) / len(scored)
    C = round(max(0.0, min(100.0, C)), 1)

    # Dominant claim type (most frequent)
    type_counts: Dict[str, int] = {}
    for s in scored:
        ct = s["claim_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
    dominant_type = max(type_counts, key=type_counts.get) if type_counts else "strategic_claim"

    # Aggregate specificity level
    avg_spec = sum(s["specificity"] for s in scored) / len(scored)
    if avg_spec >= 20:
        spec_level = "high"
    elif avg_spec >= 10:
        spec_level = "medium"
    elif avg_spec > 0:
        spec_level = "low"
    else:
        spec_level = "vague"

    # Aggregate verifiability level
    avg_verif = sum(s["verifiability"] for s in scored) / len(scored)
    any_third_party = any(s["third_party_bonus"] > 0 for s in scored)
    if any_third_party:
        verif_level = "verified"
    elif avg_verif >= 12:
        verif_level = "checkable"
    elif avg_verif > 0:
        verif_level = "aspirational"
    else:
        verif_level = "vague"

    hedging = any(s["ambiguity_penalty"] > 0 for s in scored)

    return {
        "claim_intensity_score": C,
        "sub_claim_scores": scored,
        "dominant_claim_type": dominant_type,
        "specificity_level": spec_level,
        "verifiability_level": verif_level,
        "hedging_detected": hedging,
        "third_party_validated": any_third_party,
    }
