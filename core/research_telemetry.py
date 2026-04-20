"""
Research telemetry utilities for experiment logging and aggregate summaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_agent_output(state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    outputs = state.get("agent_outputs", [])
    if not isinstance(outputs, list):
        return {}
    for item in reversed(outputs):
        if isinstance(item, dict) and item.get("agent") == agent_name:
            out = item.get("output", {})
            return out if isinstance(out, dict) else {}
    return {}


def extract_run_metrics(
    state: Dict[str, Any],
    structured: Dict[str, Any],
    quality: Dict[str, Any],
    json_export_path: str | None = None,
) -> Dict[str, Any]:
    metadata = structured.get("metadata", {}) if isinstance(structured, dict) else {}
    calibration = structured.get("calibration", {}) if isinstance(structured, dict) else {}

    risk = _latest_agent_output(state, "risk_scoring")
    sentiment = _latest_agent_output(state, "sentiment_analysis")
    confidence_node = _latest_agent_output(state, "confidence_scoring")

    gsi_score = _safe_float(sentiment.get("gsi_score", 0), 0.0)
    boilerplate = sentiment.get("boilerplate_assessment", {}) if isinstance(sentiment.get("boilerplate_assessment"), dict) else {}
    boilerplate_score = _safe_float(boilerplate.get("score", 0), 0.0)

    archive_quality = risk.get("historical_archive_quality", {}) if isinstance(risk.get("historical_archive_quality"), dict) else {}
    adversarial_audit = state.get("adversarial_audit", {}) if isinstance(state.get("adversarial_audit"), dict) else {}
    fact_graph = state.get("fact_graph", {}) if isinstance(state.get("fact_graph"), dict) else {}
    fact_summary = fact_graph.get("summary", {}) if isinstance(fact_graph.get("summary"), dict) else {}
    company_kg = state.get("company_knowledge_graph", {}) if isinstance(state.get("company_knowledge_graph"), dict) else {}

    report_id = metadata.get("report_id") or f"RUN_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    analysis_dt = metadata.get("timestamp_dt")
    if hasattr(analysis_dt, "isoformat"):
        analysis_date = analysis_dt.isoformat()
    else:
        analysis_date = datetime.now(timezone.utc).isoformat()

    run_metrics = {
        "report_id": report_id,
        "analysis_date": analysis_date,
        "company": state.get("company"),
        "industry": state.get("industry"),
        "workflow_path": state.get("workflow_path"),
        "scores": {
            "greenwashing_score": _safe_float(risk.get("greenwashing_risk_score"), 0.0),
            "greenwashing_score_raw": _safe_float(risk.get("greenwashing_risk_score_raw"), 0.0),
            "esg_score": _safe_float(risk.get("esg_score"), 0.0),
            "confidence": _safe_float(state.get("confidence"), 0.0),
            "confidence_penalty": _safe_float(risk.get("confidence_penalty"), 0.0),
            "confidence_penalty_applied": _safe_float(risk.get("confidence_penalty_applied"), 0.0),
            "report_tier": risk.get("report_tier"),
        },
        "abstention": {
            "abstain_recommended": bool(risk.get("abstain_recommended", False)),
            "decision_status": risk.get("decision_status"),
            "abstention_reason": risk.get("abstention_reason"),
            "trigger_count": len(risk.get("abstention_triggers", []) if isinstance(risk.get("abstention_triggers"), list) else []),
        },
        "calibration": {
            "status": calibration.get("calibration_status"),
            "dataset_size": calibration.get("dataset_size"),
            "optimal_threshold": calibration.get("optimal_threshold"),
            "spearman_r": calibration.get("spearman_r"),
            "point_biserial_r": calibration.get("point_biserial_r"),
            "recalibration_alpha": (risk.get("recalibration", {}) or {}).get("alpha") if isinstance(risk.get("recalibration"), dict) else None,
            "recalibration_beta": (risk.get("recalibration", {}) or {}).get("beta") if isinstance(risk.get("recalibration"), dict) else None,
        },
        "disclosure_quality": {
            "gsi_score": round(gsi_score, 2),
            "boilerplate_score": round(boilerplate_score, 2),
            "gsi_level": (sentiment.get("greenwashing_severity_index", {}) or {}).get("level")
            if isinstance(sentiment.get("greenwashing_severity_index"), dict)
            else None,
        },
        "historical_archive_quality": archive_quality,
        "adversarial_audit": adversarial_audit,
        "fact_graph": {
            "fact_count": fact_summary.get("fact_count", 0),
            "verified_fact_count": fact_summary.get("verified_fact_count", 0),
            "claim_linked_fact_count": fact_summary.get("claim_linked_fact_count", 0),
            "is_decision_ready": fact_summary.get("is_decision_ready", False),
        },
        "fact_graph_path": state.get("fact_graph_path"),
        "company_knowledge_graph": {
            "status": company_kg.get("status"),
            "configured": company_kg.get("configured"),
            "organization_anchor": company_kg.get("organization_anchor"),
            "entity_count": company_kg.get("entity_count"),
            "relationship_count": company_kg.get("relationship_count"),
            "payload_path": company_kg.get("payload_path"),
        },
        "quality": {
            "report_confidence_level": quality.get("report_confidence_level"),
            "quality_warning_count": len(quality.get("quality_warnings", []) if isinstance(quality.get("quality_warnings"), list) else []),
        },
        "confidence_node": confidence_node if isinstance(confidence_node, dict) else {},
        "json_export_path": json_export_path,
    }
    return run_metrics


def append_run_metrics(metrics: Dict[str, Any], log_path: str = "reports/research_runs.jsonl") -> str:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    return str(path)


def load_runs(log_path: str = "reports/research_runs.jsonl") -> List[Dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    return rows


def summarize_runs(log_path: str = "reports/research_runs.jsonl") -> Dict[str, Any]:
    runs = load_runs(log_path=log_path)
    n = len(runs)
    if n == 0:
        return {
            "run_count": 0,
            "abstention_rate": 0.0,
            "mean_greenwashing_score": 0.0,
            "mean_gsi_score": 0.0,
            "mean_boilerplate_score": 0.0,
            "mean_archive_confidence": 0.0,
            "mean_coordination_risk": 0.0,
            "decision_ready_fact_graph_rate": 0.0,
            "tier_distribution": {},
        }

    abstains = 0
    sum_gw = 0.0
    sum_gsi = 0.0
    sum_bp = 0.0
    sum_arch = 0.0
    sum_coord = 0.0
    decision_ready = 0
    tier_distribution: Dict[str, int] = {}

    for r in runs:
        abstains += 1 if bool((r.get("abstention", {}) or {}).get("abstain_recommended", False)) else 0
        sum_gw += _safe_float(((r.get("scores", {}) or {}).get("greenwashing_score")), 0.0)
        sum_gsi += _safe_float(((r.get("disclosure_quality", {}) or {}).get("gsi_score")), 0.0)
        sum_bp += _safe_float(((r.get("disclosure_quality", {}) or {}).get("boilerplate_score")), 0.0)
        sum_arch += _safe_float(((r.get("historical_archive_quality", {}) or {}).get("archive_confidence")), 0.0)
        sum_coord += _safe_float(((r.get("adversarial_audit", {}) or {}).get("coordination_risk")), 0.0)
        if bool((r.get("fact_graph", {}) or {}).get("is_decision_ready", False)):
            decision_ready += 1
        tier = str(((r.get("scores", {}) or {}).get("report_tier") or "UNKNOWN"))
        tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

    return {
        "run_count": n,
        "abstention_rate": round(abstains / n, 4),
        "mean_greenwashing_score": round(sum_gw / n, 3),
        "mean_gsi_score": round(sum_gsi / n, 3),
        "mean_boilerplate_score": round(sum_bp / n, 3),
        "mean_archive_confidence": round(sum_arch / n, 3),
        "mean_coordination_risk": round(sum_coord / n, 4),
        "decision_ready_fact_graph_rate": round(decision_ready / n, 4),
        "tier_distribution": tier_distribution,
    }
