"""
Adversarial audit framework for agentic ESG runs.

Computes coordination and challenge metrics from multi-agent outputs so we can
track when a run is potentially brittle or overconfident.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_conf(val: Any) -> float:
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return 0.0


def build_adversarial_audit(state: Dict[str, Any]) -> Dict[str, Any]:
    outputs = state.get("agent_outputs", [])
    if not isinstance(outputs, list):
        outputs = []

    successful = 0
    failed = 0
    confidences: List[float] = []
    agent_names = set()

    contradiction_count = 0
    debate_conflict_ratio = 0.0
    regulatory_gaps = 0

    for out in outputs:
        if not isinstance(out, dict):
            continue
        agent = str(out.get("agent") or "").strip()
        if agent:
            agent_names.add(agent)
        if "error" in out:
            failed += 1
            continue
        successful += 1
        conf = out.get("confidence")
        if isinstance(conf, (int, float)):
            confidences.append(_safe_conf(conf))

        if agent == "contradiction_analysis":
            contradiction_count = max(
                contradiction_count,
                int(out.get("contradictions_count", 0) or 0),
            )
        elif agent in {"debate_orchestrator", "debate_resolution", "debate"}:
            try:
                debate_conflict_ratio = max(debate_conflict_ratio, float(out.get("conflict_ratio", 0) or 0))
            except (TypeError, ValueError):
                pass
        elif agent == "regulatory_scanning":
            payload = out.get("output", {})
            if isinstance(payload, dict):
                results = payload.get("compliance_results", [])
                if isinstance(results, list):
                    regulatory_gaps = max(
                        regulatory_gaps,
                        sum(1 for r in results if isinstance(r, dict) and (r.get("gap_details") or [])),
                    )

    mean_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0.0
    spread = 0.0
    if len(confidences) >= 2:
        spread = max(confidences) - min(confidences)

    coordination_risk = 0.0
    if failed >= 3:
        coordination_risk += 0.30
    elif failed >= 1:
        coordination_risk += 0.12
    if spread >= 0.45:
        coordination_risk += 0.25
    elif spread >= 0.25:
        coordination_risk += 0.12
    if contradiction_count >= 4:
        coordination_risk += 0.18
    if debate_conflict_ratio >= 0.60:
        coordination_risk += 0.20
    if regulatory_gaps >= 3:
        coordination_risk += 0.10

    coordination_risk = min(1.0, coordination_risk)
    if coordination_risk >= 0.60:
        band = "HIGH"
        penalty = 0.12
    elif coordination_risk >= 0.35:
        band = "MEDIUM"
        penalty = 0.07
    else:
        band = "LOW"
        penalty = 0.0

    return {
        "agents_seen": sorted(agent_names),
        "successful_agents": successful,
        "failed_agents": failed,
        "mean_agent_confidence": round(mean_conf, 3),
        "confidence_spread": round(spread, 3),
        "contradictions_count": contradiction_count,
        "debate_conflict_ratio": round(debate_conflict_ratio, 3),
        "regulatory_gap_count": regulatory_gaps,
        "coordination_risk": round(coordination_risk, 3),
        "coordination_risk_band": band,
        "confidence_penalty": penalty,
    }

