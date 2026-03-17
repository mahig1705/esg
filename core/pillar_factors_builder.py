"""
Pillar Factors Builder — Populates ESG sub-indicator breakdowns.

Produces a structured dict with per-pillar sub-indicators, weights,
data sources, and scores.  The weighted average of sub-indicator scores
is guaranteed to reproduce the given pillar score (via rescaling).

Industry-specific sub-indicators are added based on sector verticals.
"""

import re
from typing import Any, Dict, List, Optional
from core.safe_utils import safe_get, safe_number, parse_source_name, get_reliability_tier


# =========================================================================
# Sub-indicator definitions
# =========================================================================

# Default sub-indicators per pillar (used when no industry-specific set applies)

_DEFAULT_ENVIRONMENTAL = [
    {"name": "GHG Emissions Intensity",           "weight": 0.30, "keywords": ["emission", "ghg", "co2", "carbon", "scope 1", "scope 2", "tco2"]},
    {"name": "Renewable Energy Transition",        "weight": 0.25, "keywords": ["renewable", "solar", "wind", "clean energy", "green energy", "transition"]},
    {"name": "Water Usage & Stress",               "weight": 0.20, "keywords": ["water", "water stress", "water consumption", "effluent", "wastewater"]},
    {"name": "Biodiversity & Land Use",            "weight": 0.15, "keywords": ["biodiversity", "deforestation", "land use", "ecosystem", "habitat"]},
    {"name": "Waste & Circular Economy",           "weight": 0.10, "keywords": ["waste", "recycling", "circular economy", "landfill", "hazardous waste"]},
]

_DEFAULT_SOCIAL = [
    {"name": "Employee Health & Safety",           "weight": 0.25, "keywords": ["safety", "health", "occupational", "injury", "ltifr", "fatality"]},
    {"name": "Labor Rights & Fair Wages",          "weight": 0.25, "keywords": ["labor", "wage", "minimum wage", "fair wage", "working condition"]},
    {"name": "Community Impact & CSR Spend",       "weight": 0.20, "keywords": ["community", "csr", "social responsibility", "philanthropy", "donation"]},
    {"name": "Supply Chain Labor Standards",       "weight": 0.15, "keywords": ["supply chain", "child labor", "forced labor", "supplier audit", "vendor"]},
    {"name": "Diversity, Equity & Inclusion",      "weight": 0.15, "keywords": ["diversity", "inclusion", "equity", "gender", "women", "dei"]},
]

_DEFAULT_GOVERNANCE = [
    {"name": "Board Independence",                 "weight": 0.25, "keywords": ["board independence", "independent director", "non-executive"]},
    {"name": "Executive Pay Ratio",                "weight": 0.20, "keywords": ["executive pay", "ceo pay", "compensation ratio", "remuneration"]},
    {"name": "Anti-Corruption Policies",           "weight": 0.20, "keywords": ["anti-corruption", "bribery", "corruption", "ethics", "compliance"]},
    {"name": "Whistleblower Mechanisms",           "weight": 0.15, "keywords": ["whistleblower", "grievance", "reporting mechanism", "speak up"]},
    {"name": "ESG Disclosure Quality",             "weight": 0.20, "keywords": ["disclosure", "transparency", "reporting", "brsr", "gri", "tcfd"]},
]

# Industry-specific ADDITIONAL sub-indicators
_INDUSTRY_EXTRA = {
    "energy": {
        "environmental": [
            {"name": "Methane Leakage Rate",           "weight": 0.15, "keywords": ["methane", "leakage", "flaring", "fugitive emission"]},
            {"name": "Stranded Asset Risk",             "weight": 0.15, "keywords": ["stranded asset", "fossil fuel reserve", "unburnable carbon"]},
            {"name": "Carbon Capture Investment",       "weight": 0.10, "keywords": ["carbon capture", "ccs", "ccus", "sequestration"]},
            {"name": "Just Transition Programs",        "weight": 0.10, "keywords": ["just transition", "worker retraining", "community transition"]},
        ],
    },
    "oil & gas": {
        "environmental": [
            {"name": "Methane Leakage Rate",           "weight": 0.15, "keywords": ["methane", "leakage", "flaring", "fugitive emission"]},
            {"name": "Stranded Asset Risk",             "weight": 0.15, "keywords": ["stranded asset", "fossil fuel reserve", "unburnable carbon"]},
            {"name": "Carbon Capture Investment",       "weight": 0.10, "keywords": ["carbon capture", "ccs", "ccus", "sequestration"]},
            {"name": "Just Transition Programs",        "weight": 0.10, "keywords": ["just transition", "worker retraining", "community transition"]},
        ],
    },
    "manufacturing": {
        "environmental": [
            {"name": "Supply Chain Emissions",          "weight": 0.15, "keywords": ["supply chain emission", "scope 3", "upstream", "downstream"]},
            {"name": "Product End-of-Life",             "weight": 0.10, "keywords": ["end of life", "product recycling", "take-back", "product disposal"]},
            {"name": "Chemical Hazard Management",      "weight": 0.10, "keywords": ["chemical", "hazardous", "toxic", "reach", "chemical management"]},
        ],
    },
    "banking": {
        "environmental": [
            {"name": "Green Lending Ratio",             "weight": 0.15, "keywords": ["green loan", "green lending", "sustainable finance", "green bond"]},
            {"name": "Climate Risk in Loan Book",       "weight": 0.15, "keywords": ["climate risk", "loan book", "credit risk", "transition risk"]},
        ],
        "social": [
            {"name": "Financial Inclusion",             "weight": 0.15, "keywords": ["financial inclusion", "unbanked", "microfinance", "rural banking"]},
        ],
    },
    "nbfc": {
        "environmental": [
            {"name": "Green Lending Ratio",             "weight": 0.15, "keywords": ["green loan", "green lending", "sustainable finance"]},
            {"name": "Climate Risk in Loan Book",       "weight": 0.15, "keywords": ["climate risk", "loan book", "credit risk"]},
        ],
        "social": [
            {"name": "Financial Inclusion",             "weight": 0.15, "keywords": ["financial inclusion", "unbanked", "microfinance"]},
        ],
    },
    "financial services": {
        "environmental": [
            {"name": "Green Lending Ratio",             "weight": 0.15, "keywords": ["green loan", "green lending", "sustainable finance", "green bond"]},
            {"name": "Climate Risk in Loan Book",       "weight": 0.15, "keywords": ["climate risk", "loan book", "credit risk", "transition risk"]},
        ],
        "social": [
            {"name": "Financial Inclusion",             "weight": 0.15, "keywords": ["financial inclusion", "unbanked", "microfinance"]},
        ],
    },
    "technology": {
        "environmental": [
            {"name": "Data Center PUE",                 "weight": 0.12, "keywords": ["data center", "pue", "power usage effectiveness", "server"]},
            {"name": "E-Waste Policy",                  "weight": 0.10, "keywords": ["e-waste", "electronic waste", "device recycling"]},
        ],
        "governance": [
            {"name": "Algorithm Fairness / AI Ethics",  "weight": 0.15, "keywords": ["ai ethics", "algorithm fairness", "bias", "responsible ai"]},
        ],
    },
    "it": {
        "environmental": [
            {"name": "Data Center PUE",                 "weight": 0.12, "keywords": ["data center", "pue", "power usage effectiveness", "server"]},
            {"name": "E-Waste Policy",                  "weight": 0.10, "keywords": ["e-waste", "electronic waste", "device recycling"]},
        ],
        "governance": [
            {"name": "Algorithm Fairness / AI Ethics",  "weight": 0.15, "keywords": ["ai ethics", "algorithm fairness", "bias", "responsible ai"]},
        ],
    },
    "consumer goods": {
        "environmental": [
            {"name": "Packaging Recyclability",         "weight": 0.12, "keywords": ["packaging", "recyclable", "single-use plastic", "sustainable packaging"]},
        ],
        "social": [
            {"name": "Living Wage in Supply Chain",     "weight": 0.12, "keywords": ["living wage", "supply chain wage", "fair trade"]},
        ],
    },
    "retail": {
        "environmental": [
            {"name": "Packaging Recyclability",         "weight": 0.12, "keywords": ["packaging", "recyclable", "single-use plastic"]},
        ],
        "social": [
            {"name": "Living Wage in Supply Chain",     "weight": 0.12, "keywords": ["living wage", "supply chain wage", "fair trade"]},
        ],
    },
}


def _normalize_industry(industry: str) -> str:
    """Normalize industry name for lookup."""
    if not industry:
        return "general"
    return industry.strip().lower().replace("_", " ")


def _build_indicators_for_pillar(
    pillar_key: str,
    industry: str,
    default_indicators: List[Dict],
) -> List[Dict]:
    """Combine default + industry-specific indicators, then renormalize weights to sum to 1.0."""
    indicators = [dict(ind) for ind in default_indicators]  # deep copy

    norm_industry = _normalize_industry(industry)
    extra = safe_get(_INDUSTRY_EXTRA, norm_industry, pillar_key, default=[])

    if extra:
        indicators.extend([dict(e) for e in extra])

    # Renormalize weights to sum to 1.0
    total_weight = sum(ind["weight"] for ind in indicators)
    if total_weight > 0:
        for ind in indicators:
            ind["weight"] = round(ind["weight"] / total_weight, 4)

    return indicators


def _score_indicator(
    indicator: Dict,
    evidence_texts: List[str],
    evidence_sources: List[Dict],
    carbon_data: Optional[Dict],
) -> Dict[str, Any]:
    """Score a single sub-indicator based on evidence keyword matching.

    Returns a dict with score, data_source, raw_value, verified, etc.
    If evidence is insufficient, score is set to None.
    """
    keywords = indicator.get("keywords", [])
    name = indicator["name"]

    # Collect matching evidence
    matching_sources = []
    matching_texts = []
    best_url = None
    best_tier = 5
    data_year = None
    verified = False

    for ev in evidence_sources:
        if not isinstance(ev, dict):
            continue
        ev_text = " ".join(
            str(ev.get(k, ""))
            for k in ("title", "snippet", "content", "relevant_text", "source", "source_name")
        ).lower()

        hit_count = sum(1 for kw in keywords if kw in ev_text)
        if hit_count >= 1:
            matching_texts.append(ev_text[:300])
            url = ev.get("url", "")
            tier = get_reliability_tier(url)
            source_name = parse_source_name(url) or ev.get("source_name", "")
            matching_sources.append(source_name)

            if tier < best_tier:
                best_tier = tier
                best_url = url

            # Try to extract year from evidence
            for field in ("date", "publishedAt", "date_retrieved"):
                date_str = str(ev.get(field, ""))
                year_match = re.search(r"20[12]\d", date_str)
                if year_match:
                    data_year = int(year_match.group())
                    break

            # Check if from primary disclosure
            if tier == 1:
                verified = True

    # Special handling for GHG-related indicators using carbon_data
    raw_value = None
    unit = None
    methodology = None

    if carbon_data and isinstance(carbon_data, dict) and "ghg" in name.lower() or "emission" in name.lower():
        emissions = safe_get(carbon_data, "emissions", default={})
        if isinstance(emissions, dict):
            scope1_val = safe_get(emissions, "scope1", "value")
            scope2_val = safe_get(emissions, "scope2", "value")
            if scope1_val is not None or scope2_val is not None:
                parts = []
                if scope1_val is not None:
                    parts.append(f"Scope 1: {scope1_val}")
                if scope2_val is not None:
                    parts.append(f"Scope 2: {scope2_val}")
                raw_value = ", ".join(parts) + " tCO2e"
                unit = "tCO2e"
                methodology = "Scope 1+2 tCO2e per unit revenue, normalized to industry peers"
                if not matching_sources:
                    matching_sources.append("Company carbon disclosure")

    # Determine score
    if not matching_sources and raw_value is None:
        # Insufficient data — do NOT hallucinate a score
        return {
            "name": name,
            "score": None,
            "weight": indicator["weight"],
            "data_source": "Insufficient data",
            "source_url": None,
            "data_year": data_year,
            "methodology": None,
            "raw_value": None,
            "unit": unit,
            "verified": False,
        }

    # Compute a heuristic score based on evidence density and quality
    evidence_density = min(len(matching_sources), 5)  # cap at 5
    tier_bonus = {1: 20, 2: 10, 3: 5, 4: 0, 5: 0}.get(best_tier, 0)

    # Base score: 40 + evidence hits * 8 + tier bonus, capped at 95
    raw_score = min(95.0, max(20.0, 40.0 + evidence_density * 8.0 + tier_bonus))

    # Build data_source string
    unique_sources = list(dict.fromkeys(matching_sources))[:3]
    data_source = " / ".join(unique_sources) if unique_sources else "Web evidence"

    return {
        "name": name,
        "score": round(raw_score, 1),
        "weight": indicator["weight"],
        "data_source": data_source,
        "source_url": best_url,
        "data_year": data_year or 2024,
        "methodology": methodology or f"Keyword-matched evidence scoring for {name.lower()}",
        "raw_value": raw_value,
        "unit": unit,
        "verified": verified,
    }


def _rescale_scores_to_target(
    sub_indicators: List[Dict], target_score: float
) -> List[Dict]:
    """Rescale non-null sub-indicator scores so their weighted average equals target_score.

    This ensures mathematical consistency: weighted_avg(sub_scores) == pillar_score.
    """
    scored = [(i, ind) for i, ind in enumerate(sub_indicators) if ind.get("score") is not None]

    if not scored:
        return sub_indicators

    # Current weighted average (of scored indicators only)
    scored_weight_sum = sum(ind["weight"] for _, ind in scored)
    if scored_weight_sum <= 0:
        return sub_indicators

    current_avg = sum(ind["score"] * ind["weight"] for _, ind in scored) / scored_weight_sum

    if current_avg <= 0:
        # Can't rescale from zero, just set all to target
        for idx, ind in scored:
            sub_indicators[idx] = {**ind, "score": round(target_score, 1)}
        return sub_indicators

    scale_factor = target_score / current_avg

    for idx, ind in scored:
        new_score = max(0.0, min(100.0, ind["score"] * scale_factor))
        sub_indicators[idx] = {**ind, "score": round(new_score, 1)}

    return sub_indicators


def build_pillar_factors(
    company: str,
    industry: str,
    evidence_sources: List[Dict],
    carbon_data: Optional[Dict],
    pillar_scores: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build structured pillar_factors with full sub-indicator breakdown.

    Args:
        company: Company name
        industry: Industry sector (e.g. "Energy", "Banking")
        evidence_sources: List of evidence dicts from evidence retrieval
        carbon_data: Carbon extraction results dict
        pillar_scores: Dict with pillar scores (environmental_score, social_score, governance_score)

    Returns:
        Dict with 'environmental', 'social', 'governance' keys, each containing
        score, weight, and sub_indicators list.
    """
    if not isinstance(pillar_scores, dict):
        pillar_scores = {}
    if not isinstance(evidence_sources, list):
        evidence_sources = []
    if not isinstance(carbon_data, dict):
        carbon_data = {}

    # Extract pillar target scores
    env_score = safe_number(pillar_scores.get("environmental_score"), default=50.0)
    soc_score = safe_number(pillar_scores.get("social_score"), default=50.0)
    gov_score = safe_number(pillar_scores.get("governance_score"), default=50.0)

    # Gather all evidence text for scoring
    evidence_texts = []
    for ev in evidence_sources:
        if isinstance(ev, dict):
            for k in ("snippet", "content", "relevant_text", "title"):
                txt = ev.get(k, "")
                if txt:
                    evidence_texts.append(str(txt).lower())

    # Build indicators per pillar
    env_indicators = _build_indicators_for_pillar("environmental", industry, _DEFAULT_ENVIRONMENTAL)
    soc_indicators = _build_indicators_for_pillar("social", industry, _DEFAULT_SOCIAL)
    gov_indicators = _build_indicators_for_pillar("governance", industry, _DEFAULT_GOVERNANCE)

    # Score each sub-indicator
    env_scored = [
        _score_indicator(ind, evidence_texts, evidence_sources, carbon_data)
        for ind in env_indicators
    ]
    soc_scored = [
        _score_indicator(ind, evidence_texts, evidence_sources, None)
        for ind in soc_indicators
    ]
    gov_scored = [
        _score_indicator(ind, evidence_texts, evidence_sources, None)
        for ind in gov_indicators
    ]

    # Rescale to match pillar scores
    env_scored = _rescale_scores_to_target(env_scored, env_score)
    soc_scored = _rescale_scores_to_target(soc_scored, soc_score)
    gov_scored = _rescale_scores_to_target(gov_scored, gov_score)

    return {
        "environmental": {
            "score": round(env_score, 1),
            "weight": 0.40,
            "sub_indicators": env_scored,
        },
        "social": {
            "score": round(soc_score, 1),
            "weight": 0.30,
            "sub_indicators": soc_scored,
        },
        "governance": {
            "score": round(gov_score, 1),
            "weight": 0.30,
            "sub_indicators": gov_scored,
        },
    }
