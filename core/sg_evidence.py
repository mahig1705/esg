from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


HIGH_TRUST_SOURCE_TYPES = {
    "Government/Regulatory",
    "Government/International Data",
    "Legal/Court Documents",
    "Compliance/Sanctions Database",
    "UK/EU Regulatory",
    "NGO",
    "Climate NGO",
    "Supply Chain Database",
    "Tier-1 Financial Media",
}

GOVERNANCE_PRIORITY_SOURCE_TYPES = {
    "Government/Regulatory",
    "Legal/Court Documents",
    "Compliance/Sanctions Database",
    "UK/EU Regulatory",
}

TRACKS: Dict[str, List[Dict[str, Any]]] = {
    "social": [
        {
            "key": "employee_health_and_safety",
            "indicator_name": "Employee Health & Safety",
            "keywords": ["health and safety", "occupational", "worker safety", "injury", "fatality", "ltifr", "trir", "safety"],
            "target_sections": ["health and safety", "occupational health", "workforce health", "asset integrity", "worker safety"],
            "target_evidence_types": ["metric", "policy", "training", "incident", "assurance"],
        },
        {
            "key": "labor_rights_and_wages",
            "indicator_name": "Labor Rights & Wages",
            "keywords": ["labor", "labour", "collective bargaining", "union", "wage", "adequate wage", "living wage", "remediation", "worker dialogue"],
            "target_sections": ["workforce policies", "employee dialogue", "adequate wages", "collective bargaining", "remediation channels"],
            "target_evidence_types": ["policy", "metric", "grievance", "remediation", "assurance"],
        },
        {
            "key": "community_impact",
            "indicator_name": "Community Impact",
            "keywords": ["community", "local community", "social investment", "indigenous", "consumer", "end-user", "customer health", "community impact"],
            "target_sections": ["community impact", "consumer and end-user", "local communities", "social investment"],
            "target_evidence_types": ["policy", "metric", "impact_assessment", "program", "controversy"],
        },
        {
            "key": "supply_chain_labor_standards",
            "indicator_name": "Supply-Chain Labor Standards",
            "keywords": ["supply chain", "supplier", "forced labor", "child labor", "vendor", "conflict minerals", "responsible sourcing", "supplier audit"],
            "target_sections": ["supply chain", "human rights", "supplier standards", "responsible sourcing"],
            "target_evidence_types": ["policy", "audit", "due_diligence", "metric", "controversy"],
        },
        {
            "key": "dei",
            "indicator_name": "Diversity, Equity & Inclusion",
            "keywords": ["diversity", "equity", "inclusion", "dei", "gender", "women", "ethnicity", "disability", "pay gap"],
            "target_sections": ["diversity metrics", "dei", "workforce composition", "equal opportunity"],
            "target_evidence_types": ["policy", "metric", "target", "assurance"],
        },
    ],
    "governance": [
        {
            "key": "board_structure",
            "indicator_name": "Board Structure",
            "keywords": ["board", "independent director", "board independence", "audit committee", "chair", "non-executive"],
            "target_sections": ["board composition", "board oversight", "audit committee", "director independence"],
            "target_evidence_types": ["metric", "policy", "committee", "filing"],
        },
        {
            "key": "executive_compensation",
            "indicator_name": "Executive Compensation",
            "keywords": ["executive compensation", "remuneration", "pay ratio", "ceo pay", "incentive", "lti", "compensation"],
            "target_sections": ["remuneration", "compensation discussion", "proxy statement"],
            "target_evidence_types": ["metric", "policy", "filing"],
        },
        {
            "key": "anti_corruption",
            "indicator_name": "Anti-Corruption",
            "keywords": ["anti-corruption", "anti bribery", "bribery", "corruption", "ethics", "compliance"],
            "target_sections": ["ethics and compliance", "anti-corruption", "code of conduct"],
            "target_evidence_types": ["policy", "training", "controversy", "assurance"],
        },
        {
            "key": "whistleblower_protection",
            "indicator_name": "Whistleblower Protection",
            "keywords": ["whistleblower", "speak up", "helpline", "hotline", "no retaliation", "grievance mechanism"],
            "target_sections": ["whistleblower", "speak up", "helpline", "ethics hotline"],
            "target_evidence_types": ["policy", "metric", "channel", "controversy"],
        },
        {
            "key": "sustainability_governance_controls",
            "indicator_name": "Sustainability Governance Controls",
            "keywords": ["sustainability governance", "esg oversight", "tcfd", "controls", "internal controls", "assurance", "governance of sustainability"],
            "target_sections": ["sustainability governance", "management oversight", "internal controls", "assurance"],
            "target_evidence_types": ["policy", "metric", "assurance", "committee", "filing"],
        },
    ],
}

POLICY_TERMS = {
    "policy",
    "policies",
    "code of conduct",
    "framework",
    "oversight",
    "governance",
    "management system",
    "principles",
    "remediation",
    "hotline",
    "helpline",
}
ASSURANCE_TERMS = {"assurance", "assured", "verified", "externally assured", "limited assurance", "reasonable assurance", "audited"}
CONTROVERSY_TERMS = {"violation", "fine", "penalty", "lawsuit", "investigation", "breach", "corruption", "bribery", "fatality", "injury", "complaint", "sanction"}
METRIC_TERMS = {"%", "rate", "ratio", "score", "hours", "cases", "employees", "workers", "incidents", "fatalities", "training"}


def _flatten_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, ev in enumerate(evidence or [], start=1):
        if not isinstance(ev, dict):
            continue
        rows.append(
            {
                "row_id": f"ev_{idx}",
                "text": " ".join(str(ev.get(key, "") or "") for key in ("title", "snippet", "relevant_text", "content")).strip(),
                "source_name": str(ev.get("source_name") or ev.get("source") or "Unknown"),
                "source_type": str(ev.get("source_type") or ""),
                "api_source": str(ev.get("data_source_api") or "Unknown"),
                "url": str(ev.get("url") or ev.get("link") or ""),
                "year": _extract_year(ev.get("year") or ev.get("date") or ev.get("published_at")),
            }
        )
        nested = ev.get("evidence", [])
        if not isinstance(nested, list):
            continue
        for n_idx, item in enumerate(nested, start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "row_id": f"ev_{idx}_{n_idx}",
                    "text": " ".join(str(item.get(key, "") or "") for key in ("title", "snippet", "relevant_text", "content")).strip(),
                    "source_name": str(item.get("source_name") or ev.get("source_name") or ev.get("source") or "Unknown"),
                    "source_type": str(item.get("source_type") or ev.get("source_type") or ""),
                    "api_source": str(item.get("data_source_api") or ev.get("data_source_api") or "Unknown"),
                    "url": str(item.get("url") or ev.get("url") or ""),
                    "year": _extract_year(item.get("year") or item.get("date") or ev.get("year") or ev.get("date")),
                }
            )
    return [row for row in rows if row.get("text")]


def _extract_year(value: Any) -> Optional[int]:
    if isinstance(value, int) and 1900 <= value <= 2100:
        return value
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z]{4,}", text.lower())}


def _trust_level(source_type: str) -> str:
    if source_type in HIGH_TRUST_SOURCE_TYPES:
        return "high"
    if source_type in {"Company-Controlled", "Corporate Website", "Sustainability Report", "Annual Report"}:
        return "medium"
    return "low"


def _detect_evidence_type(text: str) -> str:
    if any(term in text for term in ASSURANCE_TERMS):
        return "assurance"
    if any(term in text for term in CONTROVERSY_TERMS):
        return "controversy"
    if any(ch.isdigit() for ch in text) and any(term in text for term in METRIC_TERMS):
        return "metric"
    if any(term in text for term in POLICY_TERMS):
        return "policy"
    return "mention"


def _match_track(text: str, claim_text: str, pillar: str, track: Dict[str, Any], row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    low = text.lower()
    keyword_hits = [kw for kw in track["keywords"] if kw in low]
    if not keyword_hits:
        return None
    sections = [name for name in track["target_sections"] if name in low]
    evidence_types = [kind for kind in track["target_evidence_types"] if kind.replace("_", " ") in low]
    inferred_type = _detect_evidence_type(low)
    if inferred_type not in evidence_types:
        evidence_types.append(inferred_type)
    trust_level = _trust_level(str(row.get("source_type") or ""))
    has_metric = inferred_type == "metric" or bool(re.search(r"\b\d+(?:\.\d+)?\b", low))
    has_policy = inferred_type == "policy" or any(term in low for term in POLICY_TERMS)
    has_assurance = inferred_type == "assurance" or any(term in low for term in ASSURANCE_TERMS)
    controversy = inferred_type == "controversy" or any(term in low for term in CONTROVERSY_TERMS)
    claim_tokens = _token_set(claim_text or "")
    claim_linked = len(claim_tokens.intersection(_token_set(text))) >= 2
    evidence_state = "verified_evidence" if trust_level == "high" or (trust_level == "medium" and has_metric and row.get("url")) else "unverified_mention"
    return {
        "fact_id": f"{pillar}_{track['key']}_{row['row_id']}",
        "pillar": pillar,
        "track": track["key"],
        "indicator_name": track["indicator_name"],
        "source_name": row.get("source_name", "Unknown"),
        "source_type": row.get("source_type", ""),
        "trust_level": trust_level,
        "year": row.get("year"),
        "metric_present": has_metric,
        "policy_present": has_policy,
        "assurance_present": has_assurance,
        "controversy_present": controversy,
        "claim_linked": claim_linked,
        "evidence_state": evidence_state,
        "target_sections_matched": sections,
        "target_evidence_types_matched": evidence_types,
        "text": text[:800],
        "url": row.get("url", ""),
        "api_source": row.get("api_source", "Unknown"),
        "priority_source": bool(row.get("source_type") in GOVERNANCE_PRIORITY_SOURCE_TYPES),
    }


def _disclosure_stage(facts: List[Dict[str, Any]]) -> str:
    policy = any(f.get("policy_present") for f in facts)
    metric = any(f.get("metric_present") for f in facts)
    assurance = any(f.get("assurance_present") for f in facts)
    if not facts:
        return "no_disclosure"
    if policy and metric and assurance:
        return "policy_metric_assurance"
    if policy and metric:
        return "policy_and_metric"
    if policy:
        return "policy_only"
    if metric:
        return "metric_only"
    return "mention_only"


def _score_track(track_summary: Dict[str, Any]) -> float:
    if track_summary["evidence_state"] == "insufficient_evidence":
        return 50.0
    stage = track_summary["disclosure_stage"]
    score_map = {
        "no_disclosure": 35.0,
        "mention_only": 48.0,
        "policy_only": 58.0,
        "metric_only": 62.0,
        "policy_and_metric": 74.0,
        "policy_metric_assurance": 86.0,
    }
    score = score_map.get(stage, 50.0)
    if track_summary.get("controversy_count", 0) > 0:
        score -= min(28.0, 12.0 + (track_summary["controversy_count"] * 4.0))
    if track_summary["evidence_state"] == "unverified_mention":
        score = min(score, 55.0)
    return max(0.0, min(100.0, round(score, 1)))


def _summarize_track(pillar: str, track: Dict[str, Any], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    verified = [f for f in facts if f.get("evidence_state") == "verified_evidence"]
    mentions = [f for f in facts if f.get("evidence_state") == "unverified_mention"]
    years = sorted({int(f["year"]) for f in facts if isinstance(f.get("year"), int)})
    summary = {
        "pillar": pillar,
        "track": track["key"],
        "indicator_name": track["indicator_name"],
        "target_sections": track["target_sections"],
        "target_evidence_types": track["target_evidence_types"],
        "evidence_state": "verified_evidence" if verified else ("unverified_mention" if facts else "insufficient_evidence"),
        "verified_fact_count": len(verified),
        "unverified_mention_count": len(mentions),
        "fact_count": len(facts),
        "source_names": sorted({str(f.get("source_name") or "") for f in facts if f.get("source_name")}),
        "source_types": sorted({str(f.get("source_type") or "") for f in facts if f.get("source_type")}),
        "api_sources": sorted({str(f.get("api_source") or "") for f in facts if f.get("api_source")}),
        "years": years,
        "metric_count": sum(1 for f in facts if f.get("metric_present")),
        "policy_count": sum(1 for f in facts if f.get("policy_present")),
        "assurance_count": sum(1 for f in facts if f.get("assurance_present")),
        "controversy_count": sum(1 for f in facts if f.get("controversy_present")),
        "priority_source_hits": sum(1 for f in facts if f.get("priority_source")),
        "claim_linked_fact_count": sum(1 for f in facts if f.get("claim_linked")),
        "disclosure_stage": _disclosure_stage(facts),
        "recent_year": max(years) if years else None,
        "multi_year_memory": len(years) >= 2,
        "facts": facts[:12],
    }
    summary["track_score"] = _score_track(summary)
    return summary


def _build_gate(pillar: str, facts: List[Dict[str, Any]], tracks: List[Dict[str, Any]], adequacy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    distinct_sources = {str(f.get("source_name") or "").strip().lower() for f in facts if f.get("source_name")}
    distinct_apis = {str(f.get("api_source") or "").strip().lower() for f in facts if f.get("api_source")}
    high_trust_items = sum(1 for f in facts if f.get("trust_level") == "high")
    verified_items = sum(1 for f in facts if f.get("evidence_state") == "verified_evidence")
    thresholds = {
        "min_items": int(adequacy_cfg.get("min_items_per_pillar", 4) or 4),
        "min_sources": int(adequacy_cfg.get("min_distinct_sources_per_pillar", 3) or 3),
        "min_apis": int(adequacy_cfg.get("min_distinct_apis_per_pillar", 2) or 2),
        "min_high_trust_items": int(adequacy_cfg.get("min_high_trust_items_per_pillar", 2) or 2),
    }
    passed = (
        len(facts) >= thresholds["min_items"]
        and len([s for s in distinct_sources if s]) >= thresholds["min_sources"]
        and len([a for a in distinct_apis if a]) >= thresholds["min_apis"]
        and high_trust_items >= thresholds["min_high_trust_items"]
    )
    state = "verified_evidence" if passed and verified_items > 0 else ("unverified_mention" if facts else "insufficient_evidence")
    warnings: List[str] = []
    if not passed:
        warnings.append(
            f"{pillar.title()} pre-score gate failed: items={len(facts)}/{thresholds['min_items']}, "
            f"sources={len(distinct_sources)}/{thresholds['min_sources']}, "
            f"apis={len(distinct_apis)}/{thresholds['min_apis']}, "
            f"high_trust={high_trust_items}/{thresholds['min_high_trust_items']}."
        )
    if pillar == "governance" and sum(1 for f in facts if f.get("priority_source")) == 0:
        warnings.append("Governance extraction has no regulatory or filing-priority evidence.")
    if len([t for t in tracks if t["evidence_state"] != "insufficient_evidence"]) == 0:
        warnings.append(f"{pillar.title()} has no populated extraction tracks.")
    return {
        "passed": passed,
        "evidence_state": state,
        "items": len(facts),
        "distinct_sources": len(distinct_sources),
        "distinct_apis": len(distinct_apis),
        "high_trust_items": high_trust_items,
        "verified_items": verified_items,
        "thresholds": thresholds,
        "warnings": warnings,
    }


def _temporal_summary(track_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    years = sorted({year for track in track_summaries for year in track.get("years", [])})
    multi_year_tracks = [track["track"] for track in track_summaries if track.get("multi_year_memory")]
    return {
        "years_covered": years,
        "multi_year_track_count": len(multi_year_tracks),
        "multi_year_tracks": multi_year_tracks,
        "mode": "multi_year" if len(years) >= 2 else "single_year_snapshot",
    }


def _normalize_external_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score <= 1.0:
        score *= 100.0
    elif score <= 5.0:
        score *= 20.0
    return max(0.0, min(100.0, score))


def _benchmark_reconciliation(pillar: str, track_summaries: List[Dict[str, Any]], external_benchmarks: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    internal_scores = [float(track.get("track_score", 50.0)) for track in track_summaries if isinstance(track, dict)]
    internal_score = round(sum(internal_scores) / len(internal_scores), 1) if internal_scores else 50.0
    ext_scores = external_benchmarks.get("scores", {}) if isinstance(external_benchmarks, dict) else {}
    external_score = _normalize_external_score(ext_scores.get("social" if pillar == "social" else "governance"))
    if external_score is None:
        return {"internal_score": internal_score, "external_score": None, "difference": None, "blending_guidance": "internal_only"}
    difference = round(external_score - internal_score, 1)
    return {
        "internal_score": internal_score,
        "external_score": external_score,
        "difference": difference,
        "blending_guidance": "secondary_correction_only",
    }


def build_sg_evidence_pack(
    evidence: List[Dict[str, Any]],
    claim_text: str = "",
    adequacy_cfg: Optional[Dict[str, Any]] = None,
    external_benchmarks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    adequacy_cfg = adequacy_cfg or {}
    rows = _flatten_evidence(evidence)
    facts_by_pillar_track: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    normalized_facts: List[Dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        for pillar, pillar_tracks in TRACKS.items():
            for track in pillar_tracks:
                fact = _match_track(text=text, claim_text=claim_text, pillar=pillar, track=track, row=row)
                if not fact:
                    continue
                facts_by_pillar_track[pillar][track["key"]].append(fact)
                normalized_facts.append(fact)

    pack: Dict[str, Any] = {
        "normalized_facts": normalized_facts,
        "claim_linked_fact_count": sum(1 for fact in normalized_facts if fact.get("claim_linked")),
        "pillars": {},
        "summary": {
            "normalized_fact_count": len(normalized_facts),
            "verified_fact_count": sum(1 for fact in normalized_facts if fact.get("evidence_state") == "verified_evidence"),
            "claim_linked_fact_count": sum(1 for fact in normalized_facts if fact.get("claim_linked")),
        },
    }

    for pillar, pillar_tracks in TRACKS.items():
        track_summaries = [
            _summarize_track(pillar=pillar, track=track, facts=facts_by_pillar_track[pillar].get(track["key"], []))
            for track in pillar_tracks
        ]
        pillar_facts = [fact for track in track_summaries for fact in track.get("facts", [])]
        gate = _build_gate(pillar=pillar, facts=pillar_facts, tracks=track_summaries, adequacy_cfg=adequacy_cfg)
        pack["pillars"][pillar] = {
            "evidence_state": gate["evidence_state"],
            "pre_score_gate": gate,
            "tracks": track_summaries,
            "temporal_memory": _temporal_summary(track_summaries),
            "benchmark_reconciliation": _benchmark_reconciliation(pillar, track_summaries, external_benchmarks),
            "scoreable": bool(gate["passed"]),
        }

    overall_ready = all(pack["pillars"][pillar]["pre_score_gate"]["passed"] for pillar in ("social", "governance"))
    warnings = []
    for pillar in ("social", "governance"):
        warnings.extend(pack["pillars"][pillar]["pre_score_gate"].get("warnings", []))
    pack["summary"].update(
        {
            "overall_ready": overall_ready,
            "warnings": warnings,
            "social_evidence_state": pack["pillars"]["social"]["evidence_state"],
            "governance_evidence_state": pack["pillars"]["governance"]["evidence_state"],
        }
    )
    return pack


def build_legacy_sg_adequacy(sg_pack: Dict[str, Any]) -> Dict[str, Any]:
    pillars = sg_pack.get("pillars", {}) if isinstance(sg_pack, dict) else {}
    social = pillars.get("social", {}) if isinstance(pillars.get("social"), dict) else {}
    governance = pillars.get("governance", {}) if isinstance(pillars.get("governance"), dict) else {}

    def legacy_block(pillar_data: Dict[str, Any]) -> Dict[str, Any]:
        gate = pillar_data.get("pre_score_gate", {}) if isinstance(pillar_data.get("pre_score_gate"), dict) else {}
        return {
            "is_adequate": bool(gate.get("passed", False)),
            "evidence_state": pillar_data.get("evidence_state", "insufficient_evidence"),
            "items": int(gate.get("items", 0) or 0),
            "distinct_sources": int(gate.get("distinct_sources", 0) or 0),
            "distinct_apis": int(gate.get("distinct_apis", 0) or 0),
            "high_trust_items": int(gate.get("high_trust_items", 0) or 0),
            "verified_items": int(gate.get("verified_items", 0) or 0),
            "pre_score_gate": gate,
            "tracks": pillar_data.get("tracks", []),
            "temporal_memory": pillar_data.get("temporal_memory", {}),
            "benchmark_reconciliation": pillar_data.get("benchmark_reconciliation", {}),
        }

    warnings = sg_pack.get("summary", {}).get("warnings", []) if isinstance(sg_pack.get("summary"), dict) else []
    return {
        "enabled": True,
        "overall_ready": bool(sg_pack.get("summary", {}).get("overall_ready", False)),
        "social": legacy_block(social),
        "governance": legacy_block(governance),
        "warnings": warnings if isinstance(warnings, list) else [],
    }
