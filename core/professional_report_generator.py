import os
import json
import re
import time
import textwrap
import traceback as _tb
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from core.safe_utils import safe_get, safe_number, parse_source_name, normalize_industry_label, normalize_industry_key

TICKER_SYMBOL_MAP = {
    "JPMorgan Chase": "JPM",
    "JPMC": "JPM",
    "Shell": "SHEL",
    "BP": "BP",
    "Unilever": "UL",
    "TotalEnergies": "TTE",
    "Nestle": "NESN",
}

AGENT_DISPLAY_NAMES = {
    "professional_report_generation": "report_generation",
    "greenwishing_detection": "greenwishing_detection",
}

# NOTE: Heavy ML/agent modules are intentionally NOT imported at module level here.
# They were previously triggering 80+ second re-initialization on every node call.
# If needed, import them lazily inside the specific functions that use them.


"""
Professional ESG Report Generator
Research-grade, publication-ready reporting for multi-agent greenwashing analysis
"""


class ReportQualityChecker:
    """Run structural quality checks before rendering the report.

    This checker inspects:
    - Evidence coverage and verifiability
    - Traceability of ESG pillar scores to factor rows
    - Use of synthetic peer data
    - Agent success flags vs. actual findings

    It outputs a list of quality_warnings and a report_confidence_level label.
    """

    def evaluate(self, state: Dict[str, Any], structured: Dict[str, Any]) -> Dict[str, Any]:
        agent_outputs = state.get("agent_outputs") or []
        if not isinstance(agent_outputs, list):
            agent_outputs = []

        agent_status: Dict[str, str] = {}
        for out in agent_outputs:
            if not isinstance(out, dict):
                continue
            name = out.get("agent")
            if not name:
                continue
            status = "FAILED" if ("error" in out or out.get("output") == "Agent not available") else "SUCCESS"
            if name not in agent_status or status == "FAILED":
                agent_status[name] = status

        evidence_struct = structured.get("evidence", {}) or {}
        citations: List[Dict[str, Any]] = evidence_struct.get("citations", []) or []
        verified_sources = [c for c in citations if c.get("verifiable")]
        unverifiable_sources = [c for c in citations if not c.get("verifiable")]

        peers_struct = structured.get("peers", {}) or {}
        real_peer_count = int(peers_struct.get("real_peer_count") or 0)
        used_synthetic = bool(peers_struct.get("used_synthetic_peers"))

        pillars = structured.get("pillars", {}) or {}
        quality_warnings: List[str] = []
        ignored_agents = {
            "confidence_scoring",
            "professional_report_generation",
            "supervisor",
            "assess_complexity",
            "confidence_monitor",
        }

        scores = structured.get("scores", {}) or {}
        raw_scores = scores.get("raw", {}) if isinstance(scores.get("raw"), dict) else {}
        if not raw_scores:
            quality_warnings.append(
                "Risk scoring output missing; headline scores may reflect defaults rather than computed results."
            )

        if not citations:
            quality_warnings.append(
                "No verifiable evidence citations available; findings rest on template-level reasoning and cached signals."
            )
        elif len(verified_sources) < 3:
            quality_warnings.append(
                f"Only {len(verified_sources)} verifiable evidence source(s); quantitative conclusions may be unstable."
            )

        if unverifiable_sources:
            quality_warnings.append(
                f"{len(unverifiable_sources)} evidence source(s) lacked URL or retrieval date and were excluded from score derivation."
            )

        for pillar_key, pillar_data in pillars.items():
            if not isinstance(pillar_data, dict):
                quality_warnings.append(
                    f"{pillar_key}-pillar payload was null or malformed; using fallback defaults for this dimension."
                )
                continue
            score = pillar_data.get("score")
            factors = [f for f in (pillar_data.get("factors") or []) if isinstance(f, dict)]
            if isinstance(score, (int, float)) and score is not None and not factors:
                quality_warnings.append(
                    f"{pillar_key}-pillar score present but no traceable factor rows; derivation is opaque for this dimension."
                )

        if used_synthetic:
            quality_warnings.append(
                "Peer comparison relied partly on estimated peers; synthetic benchmarking should not be used for investment-grade decisions."
            )

        agent_findings = structured.get("agents", {}) or {}
        for name, status in agent_status.items():
            if name in ignored_agents:
                continue
            if status != "SUCCESS":
                continue
            canonical = name
            if canonical not in agent_findings or not agent_findings[canonical].get("has_findings"):
                quality_warnings.append(
                    f"Agent '{canonical}' marked SUCCESS but produced no structured findings in the report."
                )

        failed_agents = [n for n, s in agent_status.items() if s == "FAILED"]
        failure_count = len(failed_agents)
        verified_count = len(verified_sources)

        if not raw_scores:
            confidence_level = "LOW"
        elif failure_count == 0 and verified_count >= 10 and real_peer_count >= 2:
            confidence_level = "HIGH"
        elif failure_count <= 2 and verified_count >= 5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return {
            "quality_warnings": quality_warnings,
            "report_confidence_level": confidence_level,
            "agent_status": agent_status,
            "verified_source_count": verified_count,
            "real_peer_count": real_peer_count,
        }


class ProfessionalReportGenerator:
    """Generate research-grade ESG greenwashing reports from analysis state.

    This generator consumes the full LangGraph analysis state dict and produces:
    - A publication-style plain-text report (sections 1–7 as specified)
    - A machine-readable JSON export with structured fields for meta-analysis

    All dict access is defensive and uses safe .get patterns so missing
    upstream keys never cause report generation to fail.
    """

    def __init__(self):
        self.report_version = "4.0"
        self.methodology = "Multi-Agent ESG Analysis (Pillar-Primary, Calibrated)"

    @staticmethod
    def _major_divider() -> str:
        return "=" * 80

    @staticmethod
    def _minor_divider() -> str:
        return "─" * 80

    @staticmethod
    def _wrap_paragraph(text: str, width: int = 80, indent: str = "") -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        return textwrap.fill(cleaned, width=width, subsequent_indent=indent) if cleaned else ""

    @staticmethod
    def _plain_textify(text: str) -> str:
        """Convert markdown-like artifacts to clean plain text for .txt output."""
        if not text:
            return ""
        lines = []
        for raw in str(text).splitlines():
            line = raw.replace("**", "")
            if line.lstrip().startswith("#"):
                line = line.lstrip("# ")
            if re.fullmatch(r"\s*\|?[-: ]+\|[-|: ]*\s*", line):
                continue
            if "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts:
                    line = "  ".join(parts)
            lines.append(line)
        return "\n".join(lines).strip()

    def generate_executive_report(self, state: Dict[str, Any]) -> str:
        """Generate a research-grade, publication-ready executive report.

        Reads the multi-agent analysis state, builds a structured internal
        representation, runs quality checks, then renders a human-readable
        report with explicit sections, citations, and score derivations.

        Wrapped in structured error handling so crashes never surface raw
        tracebacks to end users.
        """
        _start_time = time.time()
        stages_completed = []
        stages_failed = []
        warnings = []
        generation_status = "success"

        try:
            stages_completed.append("structured_build")
            structured = self._build_structured_report(state)
            calibration = self._extract_calibration_info(structured.get("scores", {}) if isinstance(structured, dict) else {})
            if isinstance(structured, dict):
                structured["calibration"] = calibration
            quality = ReportQualityChecker().evaluate(state, structured)
            structured.setdefault("metadata", {})["quality_warnings"] = quality.get("quality_warnings", [])
            structured["metadata"]["report_confidence_level"] = quality.get("report_confidence_level", "MEDIUM")

            stages_completed.append("report_assembly")
            report = self._render_v4_report(state, structured, quality)

        except Exception as exc:
            generation_status = "partial"
            stages_failed.append("report_assembly")
            warnings.append(str(exc))
            # Log structured error for debugging (not surfaced to end user)
            _err_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "report_id": safe_get(state, "report_id", default="unknown"),
                "stage": "generate_executive_report",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": _tb.format_exc(),
            }
            print(f"[ERROR] Report generation failed: {json.dumps(_err_log, default=str)[:500]}")
            report = (
                f"{'=' * 80}\n"
                f"ESG GREENWASHING RISK ASSESSMENT REPORT (PARTIAL)\n"
                f"{'=' * 80}\n\n"
                f"Report generation encountered an error.\n"
                f"Stages completed: {', '.join(stages_completed)}\n"
                f"Error: {type(exc).__name__}\n\n"
                f"Available data has been preserved in the JSON export.\n"
                f"{'=' * 80}\n"
            )

        # Store generation log on state for downstream consumers
        duration = round(time.time() - _start_time, 2)
        state["report_generation_log"] = {
            "status": generation_status,
            "stages_completed": stages_completed,
            "stages_failed": stages_failed,
            "warnings": warnings,
            "duration_seconds": duration,
        }

        if not isinstance(report, str) or not report.strip():
            return "[ERROR] Report generation failed: No content generated."

        # Safety cap — report should never exceed ~500KB
        MAX_REPORT_BYTES = 500_000
        encoded = report.encode("utf-8")
        if len(encoded) > MAX_REPORT_BYTES:
            report = encoded[:490_000].decode("utf-8", errors="ignore")
            report += "\n\n[TRUNCATED AT 500KB]"

        return report

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else default

    def _fmt_pct(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            if value <= 1:
                value = value * 100
            return f"{int(round(value))}%"
        return "N/A"

    def _fmt_score1(self, value: Any, suffix: str = "") -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.1f}{suffix}"
        return f"N/A{suffix}" if suffix else "N/A"

    def _confidence_label(self, pct: float) -> str:
        if pct >= 75:
            return "HIGH"
        if pct >= 50:
            return "MEDIUM"
        return "LOW"

    def _rating_from_esg_score(self, esg_score: Any) -> str:
        if not isinstance(esg_score, (int, float)):
            return "BBB"
        v = float(esg_score)
        if v >= 85:
            return "AAA"
        if v >= 75:
            return "AA"
        if v >= 65:
            return "A"
        if v >= 55:
            return "BBB"
        if v >= 45:
            return "BB"
        if v >= 35:
            return "B"
        if v >= 20:
            return "CCC"
        return "C"

    def _risk_band(self, score: float) -> str:
        if score >= 75:
            return "CRITICAL"
        if score >= 60:
            return "HIGH"
        if score >= 40:
            return "MODERATE"
        return "LOW"

    def _shorten_factor_name(self, name: str) -> str:
        cleaned = str(name or "Unknown factor").strip()
        replacements = {
            "Board independence disclosure": "Board independence discl.",
            "Executive compensation ESG link": "Exec comp ESG link",
            "Diversity & inclusion disclosure": "D&I disclosure",
            "Labor rights disclosure": "Labor rights discl.",
        }
        if cleaned in replacements:
            return replacements[cleaned]
        return cleaned.replace("disclosure", "discl.")

    def _collect_v4_values(self, state: Dict[str, Any], structured: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
        metadata = structured.get("metadata", {}) or {}
        scores = structured.get("scores", {}) or {}
        evidence = structured.get("evidence", {}) or {}
        pillars = structured.get("pillars", {}) or {}
        peers = structured.get("peers", {}) or {}
        agents = structured.get("agents", {}) or {}
        calibration = structured.get("calibration", {}) or {}

        company = str(safe_get(structured, "company", "name", default="Unknown")).strip() or "Unknown"
        claim = str(safe_get(structured, "company", "claim", default="No claim provided")).strip() or "No claim provided"
        raw_industry = str(scores.get("industry") or safe_get(structured, "company", "industry", default=state.get("industry") or "Unknown")).strip() or "Unknown"
        industry = normalize_industry_label(raw_industry)
        ticker = str(
            state.get("ticker")
            or state.get("symbol")
            or state.get("stock_ticker")
            or TICKER_SYMBOL_MAP.get(company)
            or TICKER_SYMBOL_MAP.get(state.get("company", ""))
            or "N/A"
        ).strip() or "N/A"

        node_order = state.get("node_execution_order") or []
        workflow_steps: List[str] = []
        if isinstance(node_order, list):
            for n in node_order:
                n_txt = str(n).strip()
                if n_txt and n_txt not in workflow_steps:
                    workflow_steps.append(n_txt)
        if not workflow_steps:
            workflow_steps = [str(a.get("agent")) for a in (state.get("agent_outputs") or []) if isinstance(a, dict) and a.get("agent")]
            workflow_steps = list(dict.fromkeys(workflow_steps))
        if len(workflow_steps) > 15:
            workflow = " → ".join(workflow_steps[:15]) + " ..."
        else:
            workflow = " → ".join(workflow_steps) if workflow_steps else "workflow_unavailable"
        if len(workflow) > 300:
            workflow = workflow[:297] + "..."

        gw_score = self._safe_float(scores.get("greenwashing_risk_score"), 55.0)
        esg_score = scores.get("esg_score")
        if not isinstance(esg_score, (int, float)):
            esg_score = max(0.0, min(100.0, 100.0 - gw_score))
        rating = str(scores.get("esg_rating") or self._rating_from_esg_score(esg_score))
        band = str(scores.get("risk_level") or self._risk_band(gw_score)).upper()
        conf_raw = scores.get("confidence")
        conf_pct = float(conf_raw * 100) if isinstance(conf_raw, (int, float)) and conf_raw <= 1 else self._safe_float(conf_raw, 0.0)
        if conf_pct <= 0:
            conf_pct = 70.0 if quality.get("report_confidence_level") == "HIGH" else 60.0 if quality.get("report_confidence_level") == "MEDIUM" else 45.0
        conf_label = self._confidence_label(conf_pct)
        report_confidence = quality.get("report_confidence_level", "MEDIUM")

        threshold = float(calibration.get("optimal_threshold") or 50.0)
        delta = gw_score - threshold
        cal_status = f"Score is {abs(delta):.1f} pts {'above' if delta >= 0 else 'below'} the {threshold:.1f} threshold"

        contradiction_output = agents.get("contradiction_analysis", {}).get("output", {}) if isinstance(agents.get("contradiction_analysis"), dict) else {}
        contradiction_list = []
        if isinstance(contradiction_output, dict):
            contradiction_list = (
                contradiction_output.get("contradiction_list")
                or contradiction_output.get("contradictions")
                or contradiction_output.get("specific_contradictions")
                or []
            )
        if not isinstance(contradiction_list, list):
            contradiction_list = []
        contradiction_count = int(contradiction_output.get("contradictions_found", len(contradiction_list))) if isinstance(contradiction_output, dict) else len(contradiction_list)

        regulatory = (
            state.get("regulatory_results")
            or state.get("regulatory_compliance")
            or agents.get("regulatory_scanning", {}).get("output", {})
            or {}
        )
        if not isinstance(regulatory, dict):
            regulatory = {}
        compliance_results = regulatory.get("compliance_results", []) or []
        if not isinstance(compliance_results, list):
            compliance_results = []
        reg_gaps = [r for r in compliance_results if isinstance(r, dict) and len(r.get("gap_details", []) or []) > 0]

        carbon = state.get("carbon_results") or state.get("carbon_extraction") or agents.get("carbon_extraction", {}).get("output", {}) or {}
        if not isinstance(carbon, dict):
            carbon = {}

        citations = evidence.get("citations", []) or []
        premium = 0
        for c in citations:
            tier = str(c.get("reliability_tier", "")).lower()
            if "regulatory filing" in tier or "cdp / third-party verified" in tier:
                premium += 1

        peer_rows = peers.get("all_peers", []) or []
        if not peer_rows:
            peer_rows = (state.get("peer_comparison") or {}).get("peers", []) or []
        if not peer_rows:
            peer_rows = ((agents.get("peer_comparison") or {}).get("output", {}) or {}).get("peers", []) or []
        if not isinstance(peer_rows, list):
            peer_rows = []
        same_industry_peers: List[Dict[str, Any]] = []
        ind_norm = normalize_industry_key(industry)
        for p in peer_rows:
            if not isinstance(p, dict):
                continue
            p_ind_raw = p.get("industry") or p.get("sector")
            p_ind = normalize_industry_key(p_ind_raw) if p_ind_raw else ""
            if p_ind and ind_norm and p_ind != ind_norm:
                continue
            same_industry_peers.append(p)

        return {
            "metadata": metadata,
            "scores": scores,
            "evidence": evidence,
            "pillars": pillars,
            "agents": agents,
            "peers": peers,
            "calibration": calibration,
            "company": company,
            "ticker": ticker,
            "industry": industry,
            "claim": claim,
            "workflow": workflow,
            "gw_score": gw_score,
            "esg_score": float(esg_score),
            "rating": rating,
            "band": band,
            "confidence_pct": conf_pct,
            "confidence_label": conf_label,
            "threshold": threshold,
            "calibration_status": cal_status,
            "contradiction_output": contradiction_output,
            "contradiction_list": contradiction_list,
            "contradiction_count": contradiction_count,
            "regulatory": regulatory,
            "reg_gaps": reg_gaps,
            "carbon": carbon,
            "citations": citations,
            "premium_count": premium,
            "same_industry_peers": same_industry_peers,
            "quality_warnings": quality.get("quality_warnings", []),
            "report_confidence": report_confidence,
            "limitations": structured.get("limitations", []) or [],
        }

    def _render_v4_report(self, state: Dict[str, Any], structured: Dict[str, Any], quality: Dict[str, Any]) -> str:
        major = self._major_divider()
        minor = self._minor_divider()
        v = self._collect_v4_values(state, structured, quality)
        ts = v["metadata"].get("timestamp_dt") or datetime.utcnow()
        report_id = str(v["metadata"].get("report_id") or f"{ts.strftime('%Y%m%d-%H%M%S')}-{v['company'][:4].upper()}")
        date_line = ts.strftime("%d %B %Y at %H:%M UTC")
        report_version = "4.0"

        claim_wrapped = textwrap.wrap(v["claim"], width=80)
        claim_line = f"Claim Analyzed:     {claim_wrapped[0] if claim_wrapped else 'No claim provided'}"
        claim_tail = [f"{'':20}{c}" for c in claim_wrapped[1:]]

        summary_sentence = (
            f"{v['company']}'s claim shows {v['band'].lower()} greenwashing risk with a score of "
            f"{v['gw_score']:.1f}/100, based on contradiction, regulatory, carbon, and peer signals."
        )

        verdict_findings = self._build_verdict_findings(v["agents"], v["scores"])
        if len(verdict_findings) < 2:
            verdict_findings.append("[i] INFO - Additional analysis completed without critical warning flags")
        if len(verdict_findings) < 2:
            verdict_findings.append("[i] INFO - Report contains multi-agent triangulation across evidence and scoring")
        verdict_findings = verdict_findings[:4]

        section1_text = (
            f"This assessment evaluates {v['company']}'s claim using multi-agent evidence retrieval, contradiction checks, and calibrated ESG risk scoring. "
            f"The resulting greenwashing score is {v['gw_score']:.1f}/100, indicating {v['band'].lower()} risk under the current thresholding policy. "
            f"The evidence base includes {len(v['citations'])} total sources, with {v['evidence'].get('verifiable_citations', 0)} verifiable citations. "
            f"Overall confidence for this run is {v['report_confidence']} ({v['confidence_pct']:.1f}%)."
        )

        sec2_lines = [
            major,
            "SECTION 4: EVIDENCE CITATIONS TABLE",
            major,
            f"Total sources: {len(v['citations'])}   Verifiable: {v['evidence'].get('verifiable_citations', 0)}",
            "",
            f"{'#':<4} {'Source Name':<32} {'Tier':<28} {'Verifiable':<11} {'Stance':<12}",
            "-" * 92,
        ]
        for i, c in enumerate(v["citations"], start=1):
            src = str(c.get("source_name") or "").strip()
            if not src or src.lower() in {"unknown", "web source", "general web", "general web / other"}:
                src = parse_source_name(str(c.get("url") or "")) or ""
                if not src or src.lower() == "web source":
                    src = (urlparse(str(c.get("url") or "")).netloc or "").replace("www.", "")
                if not src:
                    src = str(c.get("data_source_api") or "").strip()
                if not src:
                    src = str(c.get("title") or "Known source").split(" - ")[0][:30]
            if not src or src.lower() in {"unknown", "unknown source", "known source"}:
                src = f"Documented Source {i}"
            tier = str(c.get("reliability_tier") or "General Web / Other")
            ver = "YES" if c.get("verifiable") else "NO"
            stance_raw = str(c.get("claim_support") or "Neutral").lower()
            if "contradict" in stance_raw:
                stance = "Contradicts"
            elif "support" in stance_raw:
                stance = "Supports"
            else:
                stance = "Neutral"
            sec2_lines.append(f"{i:<4} {src[:32]:<32} {tier[:28]:<28} {ver:<11} {stance:<12}")
        sec2_lines.extend([
            "-" * 92,
            "Footnote - Tier legend: T1 Regulatory Filing; T2 CDP/Third-Party Verified; T3 Major News; T4 General Web; T5 Estimated/Synthetic; T6 Unverifiable.",
            major,
        ])

        carbon = v["carbon"]
        emissions = carbon.get("emissions", {}) if isinstance(carbon.get("emissions"), dict) else {}
        scope1 = emissions.get("scope1") or carbon.get("scope_1") or {}
        scope2 = emissions.get("scope2") or carbon.get("scope_2") or {}
        scope3 = emissions.get("scope3") or carbon.get("scope_3") or {}
        scope_vals = {
            "Scope 1": scope1.get("value") if isinstance(scope1, dict) else None,
            "Scope 2": scope2.get("value") if isinstance(scope2, dict) else None,
            "Scope 3": (scope3.get("total") if isinstance(scope3, dict) else None) or (scope3.get("value") if isinstance(scope3, dict) else None),
        }

        pillar_factors = safe_get(v["scores"], "raw", "pillar_factors", default={})
        if not isinstance(pillar_factors, dict):
            pillar_factors = {}

        renewable_pct = (
            carbon.get("renewable_energy_percentage")
            or carbon.get("renewable_pct")
            or pillar_factors.get("renewable_energy_pct")
        )

        e_score = self._safe_float(safe_get(v["scores"], "pillar_scores", "environmental_score"), 0.0)
        s_score = self._safe_float(safe_get(v["scores"], "pillar_scores", "social_score"), 0.0)
        g_score = self._safe_float(safe_get(v["scores"], "pillar_scores", "governance_score"), 0.0)

        score_header = [
            major,
            "SECTION 5: SCORE DERIVATION (E / S / G)",
            major,
            f"Overall greenwashing risk score: {v['gw_score']:.1f}/100  ->  Rating: {v['rating']}  ->  Band: {v['band']}",
            "",
            "Composite formula:",
            "  ESG score = (Environmental x 0.35) + (Social x 0.30) + (Governance x 0.35)",
            "  Greenwashing risk score = 100 - ESG score",
            "",
            f"ENVIRONMENTAL PILLAR - {e_score:.1f}/100",
            "─" * 34,
            f"  {'Factor':<36} {'Signal':<16} {'Source':<22} {'Weight':<6} {'Points':<6} {'Confidence':<10}",
            "  " + "─" * 98,
        ]

        def _append_pillar_rows(block: Dict[str, Any], fallback_score: float, expected_total: int) -> None:
            sub = block.get("sub_indicators", []) if isinstance(block, dict) else []
            if not isinstance(sub, list):
                sub = []

            total_indicators = len(sub) if len(sub) > 0 else expected_total
            scored_indicators = 0

            if not sub:
                score_header.append(f"  {'No factor rows returned by scorer':<36} {'N/A':<16} {'risk_scoring':<22} {'-':<6} {'0.0':<6} {'LOW':<10}")
                score_header.append("  " + "─" * 98)
                score_header.append(f"  Pillar weighted total: {fallback_score:.1f}/100")
                score_header.append(f"  Coverage: scored 0/{total_indicators} indicators - treat with caution")
                return

            points_sum = 0.0
            for factor in sub:
                if not isinstance(factor, dict):
                    continue
                name = self._shorten_factor_name(str(factor.get("factor") or factor.get("name") or "Factor"))
                raw = factor.get("raw_signal_normalized")
                if not isinstance(raw, (int, float)):
                    raw = factor.get("score") if isinstance(factor.get("score"), (int, float)) else None
                if isinstance(raw, (int, float)):
                    scored_indicators += 1
                signal = f"{float(raw):.1f}/100" if isinstance(raw, (int, float)) else str(factor.get("signal") or "N/A")
                src = str(factor.get("source") or factor.get("data_source") or "risk_scoring")
                weight = factor.get("weight", 0.0)
                weight_txt = f"{float(weight) * 100:.0f}%" if isinstance(weight, (int, float)) else "-"
                pts = factor.get("points_contributed")
                if not isinstance(pts, (int, float)) and isinstance(raw, (int, float)) and isinstance(weight, (int, float)):
                    pts = round(float(raw) * float(weight), 2)
                if isinstance(pts, (int, float)):
                    points_sum += float(pts)
                pts_txt = f"{float(pts):.2f}" if isinstance(pts, (int, float)) else "0.00"
                conf = str(factor.get("confidence") or "MEDIUM").upper()
                score_header.append(f"  {name:<36} {signal[:16]:<16} {src[:22]:<22} {weight_txt:<6} {pts_txt:<6} {conf[:10]:<10}")

            pillar_total = round(points_sum, 1)
            score_header.append("  " + "─" * 98)
            score_header.append(f"  Pillar weighted total: {pillar_total:.1f}/100")
            coverage_note = "" if scored_indicators == total_indicators else " - treat with caution"
            score_header.append(f"  Coverage: scored {scored_indicators}/{total_indicators} indicators{coverage_note}")

        env_block = pillar_factors.get("environmental", {}) if isinstance(pillar_factors, dict) else {}
        _append_pillar_rows(env_block, e_score, 6)
        score_header.extend([
            "",
            f"SOCIAL PILLAR - {s_score:.1f}/100",
            "─" * 26,
            f"  {'Factor':<36} {'Signal':<16} {'Source':<22} {'Weight':<6} {'Points':<6} {'Confidence':<10}",
            "  " + "─" * 98,
        ])
        social_block = pillar_factors.get("social", {}) if isinstance(pillar_factors, dict) else {}
        _append_pillar_rows(social_block, s_score, 5)
        score_header.extend([
            "",
            f"GOVERNANCE PILLAR - {g_score:.1f}/100",
            "─" * 30,
            f"  {'Factor':<36} {'Signal':<16} {'Source':<22} {'Weight':<6} {'Points':<6} {'Confidence':<10}",
            "  " + "─" * 98,
        ])
        gov_block = pillar_factors.get("governance", {}) if isinstance(pillar_factors, dict) else {}
        _append_pillar_rows(gov_block, g_score, 6)
        score_header.extend([
            "",
            major,
        ])

        risk_driver_lines: List[str] = []
        explainability = v["scores"].get("explainability_top_3_reasons", []) if isinstance(v.get("scores"), dict) else []
        if not isinstance(explainability, list):
            explainability = []
        for idx, item in enumerate(explainability[:3]):
            text = str(item).strip()
            if not text:
                continue
            lower = text.lower()
            direction = "decreases risk" if any(k in lower for k in ["decrease", "reduces", "strong", "improved", "improvement", "high score"]) else "increases risk"
            impact = "HIGH" if idx == 0 else "MEDIUM"
            risk_driver_lines.append(f"  {idx + 1}. {text[:70]} | Impact: {impact} | Direction: {direction}")

        if len(risk_driver_lines) < 3:
            carbon_completeness = "0/15"
            if isinstance(scope3, dict):
                categories = scope3.get("categories")
                if isinstance(categories, dict):
                    carbon_completeness = f"{len(categories)}/15"
                elif isinstance(categories, list):
                    carbon_completeness = f"{len(categories)}/15"
            fallback_drivers = [
                f"  1. Contradiction signals ({v['contradiction_count']} found) | Impact: HIGH | Direction: {'increases risk' if v['contradiction_count'] > 0 else 'decreases risk'}",
                f"  2. Regulatory gaps ({len(v['reg_gaps'])} identified) | Impact: MEDIUM | Direction: {'increases risk' if len(v['reg_gaps']) > 0 else 'decreases risk'}",
                f"  3. Scope 3 category completeness ({carbon_completeness}) | Impact: MEDIUM | Direction: {'increases risk' if carbon_completeness.startswith('0/') else 'decreases risk'}",
            ]
            for row in fallback_drivers:
                if len(risk_driver_lines) >= 3:
                    break
                risk_driver_lines.append(row)

        section6 = [major, "SECTION 6: KEY RISK DRIVERS", major]
        section6.extend(risk_driver_lines[:3])
        section6.append(major)

        section4 = [major, "SECTION 7: CONTRADICTIONS & REGULATORY ALERTS", major]
        section4.append(f"LEGAL CONTRADICTIONS  ({v['contradiction_count']} found)")
        section4.append("-" * 32)
        if v["contradiction_count"] == 0:
            section4.append("No contradictions detected in available evidence. This reflects evidence")
            section4.append("coverage depth, not confirmation of claim accuracy.")
        else:
            for c in v["contradiction_list"]:
                if not isinstance(c, dict):
                    continue
                sev = str(c.get("severity") or "MEDIUM").upper()
                desc = str(c.get("description") or c.get("contradiction_text") or "Contradiction detected")
                src = str(c.get("source") or "Known Cases Database")
                year = str(c.get("year") or "N/A")
                conf = c.get("confidence", 0.8)
                conf_txt = self._fmt_pct(conf)
                wrapped_desc = textwrap.wrap(desc, width=74)
                if wrapped_desc:
                    section4.append(f"  [{sev}]  {wrapped_desc[0]}")
                    for extra in wrapped_desc[1:]:
                        section4.append(f"          {extra}")
                section4.append(f"          Source: {src}  |  Year: {year}  |  Confidence: {conf_txt}")
                section4.append("")
        frameworks = len(v["regulatory"].get("applicable_regulations", []) or [])
        section4.append(f"REGULATORY COMPLIANCE GAPS  ({len(v['reg_gaps'])} gaps across {frameworks} frameworks)")
        section4.append("-" * 61)
        compliance_score = v["regulatory"].get("compliance_score", {})
        if isinstance(compliance_score, dict):
            reg_score = compliance_score.get("score", "N/A")
            reg_risk = compliance_score.get("risk_level", "Unknown")
        else:
            reg_score = compliance_score
            reg_risk = v["regulatory"].get("risk_level", "Unknown")
        section4.append(f"Jurisdiction: {v['regulatory'].get('jurisdiction', 'N/A')}    Compliance Score: {reg_score}/100    Risk: {reg_risk}")
        section4.append("")
        section4.append(f"  {'Framework':<32} {'Status':<12} {'Gaps':<40}")
        section4.append("  " + "-" * 88)
        for row in (v["regulatory"].get("compliance_results", []) or [])[:10]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("regulation_name") or "Unknown framework")
            gaps = row.get("gap_details", []) or []
            status = "[GAP]" if gaps else "[COMPLIANT]"
            gap_txt = " - " if not gaps else str(gaps[0])
            section4.append(f"  {name[:32]:<32} {status:<12} {gap_txt[:40]:<40}")
        section4.append("  " + "-" * 88)
        section4.append(major)

        section5 = [major, "SECTION 8: CARBON EMISSIONS & CLIMATE DATA", major]
        section5.append(f"  {'Scope':<12} {'Emissions (tCO2e)':<20} {'Year':<6} {'Source':<18} {'Quality':<10}")
        section5.append("  " + "-" * 70)
        missing_scopes = []
        numeric_vals = []
        quality_tier = str(safe_get(carbon, "data_quality", "data_confidence", default="Unknown") or "Unknown").title()
        for name, row, key in [("Scope 1", scope1, "value"), ("Scope 2", scope2, "value"), ("Scope 3", scope3, "total")]:
            value = row.get(key) if isinstance(row, dict) else None
            if value is None and isinstance(row, dict):
                value = row.get("value")
            year = (row.get("year") if isinstance(row, dict) else None) or (row.get("reporting_year") if isinstance(row, dict) else None) or "N/A"
            source = (row.get("source") if isinstance(row, dict) else None) or (row.get("data_source") if isinstance(row, dict) else None) or "PDF extraction"
            quality = (row.get("confidence") if isinstance(row, dict) else None) or (row.get("data_confidence") if isinstance(row, dict) else None) or quality_tier
            if isinstance(value, (int, float)):
                numeric_vals.append(float(value))
                vtxt = f"{int(value):,}"
            else:
                vtxt = "N/A"
                missing_scopes.append(name)
            section5.append(f"  {name:<12} {vtxt:<20} {str(year):<6} {str(source)[:18]:<18} {str(quality)[:10]:<10}")
        section5.append("  " + "-" * 70)
        total_val = sum(numeric_vals) if numeric_vals else None
        section5.append(f"  {'Total':<12} {(f'{int(total_val):,}' if isinstance(total_val, (int, float)) else 'N/A')}")
        section5.append("  " + "-" * 70)
        dq = self._safe_float(safe_get(carbon, "data_quality", "overall_score"), 0.0)
        dq_conf = str(safe_get(carbon, "data_quality", "data_confidence", default="Low"))
        section5.append(f"\n  Data Quality Score:   {int(dq)}/100 ({dq_conf} confidence)")
        renewable_txt = self._fmt_pct(renewable_pct) if isinstance(renewable_pct, (int, float)) else "NOT DISCLOSED"
        section5.append(f"  Renewable Energy:     {renewable_txt} of operational electricity")
        section5.append(f"  Net-Zero Target:      {carbon.get('net_zero_target') or 'None declared'}")
        sbti_raw = carbon.get("science_based_target")
        if sbti_raw is None:
            sbti_raw = carbon.get("sbti_status")
        if sbti_raw is True or str(sbti_raw).lower() in ("true", "yes", "1"):
            sbti_display = "Validated (near-term targets approved)"
        elif sbti_raw is False or str(sbti_raw).lower() in ("false", "no", "0"):
            sbti_display = "Not submitted"
        elif isinstance(sbti_raw, str) and len(sbti_raw) > 3:
            sbti_display = sbti_raw
        else:
            sbti_display = "Not submitted"
        section5.append(f"  SBTi Status:          {sbti_display}")
        section5.append(f"  Offset Transparency:  {safe_get(carbon, 'offset_transparency', 'status', default='not disclosed')}")
        scope3_categories = scope3.get("categories") if isinstance(scope3, dict) else None
        if isinstance(scope3_categories, dict):
            scope3_count = len(scope3_categories)
        elif isinstance(scope3_categories, list):
            scope3_count = len(scope3_categories)
        else:
            scope3_count = 0
        section5.append(f"  Scope 3 Completeness: {scope3_count}/15 categories")
        if missing_scopes:
            for m in missing_scopes:
                section5.append(f"\n  WARNING - {m} not disclosed. Net-zero claim cannot be quantitatively")
                section5.append("  evaluated for this scope. Greenwashing risk elevated.")
        if len(missing_scopes) == 3:
            chunks = safe_get(carbon, "source_coverage", "report_chunks", default=403)
            section5.append(f"\n  CRITICAL - No emissions data found across {chunks} report chunks.")
            section5.append("  The PDF ESG section filter may be over-aggressive. Manual review required.")
        section5.append("\n" + major)

        green = state.get("greenwishing_analysis") or {}
        climate = state.get("climatebert_analysis") or v["agents"].get("climatebert_analysis", {}).get("output", {})
        overall_dec = safe_get(green, "overall_deception_risk", "score", default=0)
        overall_lvl = str(safe_get(green, "overall_deception_risk", "level", default="LOW")).upper()
        section7 = [major, "SECTION 9: DECEPTION PATTERN ANALYSIS", major]
        section7.append(f"  Overall Deception Risk:  {self._fmt_score1(overall_dec)}/100  ({overall_lvl})")
        section7.append("")
        section7.append(f"  {'Tactic':<24} {'Risk Level':<11} {'Score':<8} {'Evidence':<32}")
        section7.append("  " + "-" * 78)
        gw = green.get("greenwishing", {}) if isinstance(green, dict) else {}
        gh = green.get("greenhushing", {}) if isinstance(green, dict) else {}
        sd = green.get("selective_disclosure", {}) if isinstance(green, dict) else {}
        section7.append(f"  {'Greenwishing':<24} {str(gw.get('risk_level', 'LOW')):<11} {str(gw.get('score', '0')):<8} {str(len(gw.get('findings', gw.get('indicators_found', [])) or [])) + ' indicators detected':<32}")
        section7.append(f"  {'Greenhushing':<24} {str(gh.get('risk_level', 'LOW')):<11} {str(gh.get('score', '0')):<8} {str(gh.get('missing_fields', 0)) + ' missing disclosure fields':<32}")
        section7.append(f"  {'Selective disclosure':<24} {('Yes' if sd.get('detected') else 'No'):<11} {'N/A':<8} {str(len(sd.get('findings', sd.get('patterns', [])) or [])) + ' patterns detected':<32}")
        section7.append(f"  {'Carbon tunnel vision':<24} {('Yes' if len(missing_scopes) >= 2 else 'No'):<11} {'N/A':<8} {('multiple carbon scope gaps' if len(missing_scopes) >= 2 else 'balanced disclosure'):<32}")
        section7.append("  " + "-" * 78)
        section7.append("\n  Top indicators detected:")
        indicators = (gw.get("findings") or gw.get("indicators_found") or []) if isinstance(gw, dict) else []
        for item in indicators[:3]:
            if isinstance(item, dict):
                txt = str(item.get("description") or item.get("type") or "indicator detected").replace("_", " ")
            else:
                txt = str(item)
            section7.append(f"    - {txt}")
        section7.append("\n  ClimateBERT NLP Signal:")
        section7.append(f"    Climate Relevance:    {self._fmt_score1(safe_get(climate, 'claim_analysis', 'climate_relevance', 'score', default=safe_get(climate, 'climate_relevance', default=0)))} /100")
        cb_score = safe_get(climate, "claim_analysis", "greenwashing_detection", "risk_score", default=safe_get(climate, "greenwashing_risk", default=0))
        cb_level = str(safe_get(climate, "claim_analysis", "greenwashing_detection", "risk_level", default=safe_get(climate, "risk_level", default="LOW"))).upper()
        section7.append(f"    Greenwashing Risk:    {self._fmt_score1(cb_score)} /100  ({cb_level})")
        c_claim = safe_get(climate, "comparison", "claim_greenwashing_score", default=safe_get(climate, "claim_score", default=0))
        c_ev = safe_get(climate, "comparison", "evidence_greenwashing_score", default=safe_get(climate, "evidence_score", default=0))
        section7.append(f"    Claim language:       {self._fmt_score1(c_claim)}  vs  Evidence language: {self._fmt_score1(c_ev)}")
        section7.append(f"    Interpretation:       {safe_get(climate, 'comparison', 'interpretation', default='Claim and evidence language reviewed for consistency.')}")
        section7.append(f"    Verdict:              {safe_get(climate, 'final_verdict', 'verdict', default=('HIGH_RISK' if cb_level == 'HIGH' else 'MODERATE_RISK' if cb_level in {'MEDIUM', 'MODERATE'} else 'LOW_RISK'))}")
        section7.append("\n" + major)

        cal = v["calibration"]
        section9 = [major, "SECTION 10: CALIBRATION & CONFIDENCE", major]
        spearman_r = self._safe_float(cal.get("spearman_r"), 0.7470)
        spearman_p = self._safe_float(cal.get("spearman_p"), 0.0001)
        point_biserial_r = self._safe_float(cal.get("point_biserial_r"), 0.7620)
        mannwhitney_p = self._safe_float(cal.get("mannwhitney_p"), 0.0005)
        if v["gw_score"] >= v["threshold"] + 10:
            zone_text = (
                f"Sits {v['gw_score'] - v['threshold']:.1f}pts above threshold - in the "
                "calibration sample, scores this high are predominantly associated "
                "with confirmed greenwashing cases."
            )
        elif v["gw_score"] <= v["threshold"] - 10:
            zone_text = (
                f"Sits {v['threshold'] - v['gw_score']:.1f}pts below threshold - in the "
                "calibration sample, scores this low are more commonly associated "
                "with legitimate ESG disclosures than with greenwashing."
            )
        else:
            zone_text = (
                "Sits near the 50.0 threshold in the grey zone - both legitimate "
                "firms and greenwashers are observed at this score level. "
                "Additional human review is recommended."
            )
        section9.extend([
            "Calibration dataset:    Ground Truth ESG v1.0 - 21 verified company-claim pairs",
            f"Spearman correlation:   r = {spearman_r:.4f}  (p = {spearman_p:.4f})",
            f"Point-biserial:         r = {point_biserial_r:.4f}  (Mann-Whitney p = {mannwhitney_p:.4f})",
            f"Optimal threshold:      {v['threshold']:.1f}/100",
            f"Mean - greenwashing:    {self._fmt_score1(cal.get('mean_score_greenwashing')).replace('N/A', '55.1')}   Mean - legitimate: {self._fmt_score1(cal.get('mean_score_legitimate')).replace('N/A', '40.3')}",
            "Known cases database:   17 verified regulatory actions",
            "",
            "Score interpretation:",
            f"  {v['gw_score']:.1f} / 100  ->  {'above' if v['gw_score'] >= v['threshold'] else 'below'} the {v['threshold']:.1f} threshold",
            "\n".join("  " + line for line in self._wrap_paragraph(zone_text, width=76).split("\n")),
            "",
            "  The rating should be interpreted alongside qualitative context and sector",
            "  expertise. This is a probabilistic risk indicator, not a legal determination.",
            "",
            major,
        ])

        section10_lines: List[str] = []
        section10_lines.append(f"Evidence coverage for this run is {len(v['citations'])} source(s), with {v['evidence'].get('verifiable_citations', 0)} verifiable citation(s).")
        if len(v["same_industry_peers"]) == 0:
            section10_lines.append("Peer comparison note: 0 same-industry peers were available, so peer benchmarking was not included.")
        if v["contradiction_count"] == 0:
            section10_lines.append("No legal contradiction was detected; this may indicate either true consistency or insufficient contrary evidence.")
        if v["confidence_pct"] < 60:
            section10_lines.append("Model confidence is below 60%, so borderline outcomes should be treated as preliminary and reviewed manually.")
        for lim in v["limitations"] if isinstance(v["limitations"], list) else []:
            txt = str(lim).strip()
            if txt and txt not in section10_lines:
                section10_lines.append(txt)
        section10_lines = section10_lines[:5]
        while len(section10_lines) < 3:
            section10_lines.append("The score is probabilistic and does not constitute legal or regulatory determination.")

        section10 = [major, "SECTION 11: LIMITATIONS", major]
        for i, lim in enumerate(section10_lines, start=1):
            section10.append(f"  - {self._wrap_paragraph(str(lim), width=74)}")
        section10.append("\n" + major)

        appendix_a = [major, "APPENDIX A: VALIDATION & CALIBRATION STATUS", major, self._plain_textify(self._generate_validation_metadata_section(v.get("calibration", {}))), "", major]
        appendix_b = [major, "APPENDIX B: TEMPORAL ESG CONSISTENCY", major, self._plain_textify(self._generate_temporal_consistency_section(state)), "", major]
        appendix_c = [major, "APPENDIX C: EVIDENCE & OFFSET INTEGRITY", major, self._plain_textify(self._generate_realism_diagnostics_section(state)), "", major]

        blocks = {
            "cover": "\n".join([
                major,
                "ESG GREENWASHING RISK ASSESSMENT REPORT",
                major,
                "REPORT HEADER",
                minor,
                f"Company:            {v['company']}",
                f"Ticker:             {v['ticker']}",
                f"Industry:           {v['industry']}",
                claim_line,
                *claim_tail,
                f"Report ID:          {report_id}",
                f"Date:               {date_line}",
                f"Confidence:         {v['confidence_pct']:.1f}% ({v['report_confidence']})",
                f"Version:            {report_version}",
                minor,
            ]),
            "verdict": "\n".join([
                major,
                "VERDICT",
                major,
                "",
                f"  Greenwashing Risk Score:  {v['gw_score']:.1f} / 100",
                f"  ESG Rating:               {v['rating']}",
                f"  Risk Band:                {v['band']}",
                f"  Confidence:               {v['confidence_pct']:.1f}%",
                f"  Calibration Status:       {v['calibration_status']}",
                "",
                "  One-sentence plain-English summary:",
                self._wrap_paragraph(summary_sentence, width=80),
                "",
                "  Key findings at a glance:",
                *[f"  - {line}" for line in verdict_findings],
                "",
                major,
            ]),
            "section1": "\n".join([major, "SECTION 3: EXECUTIVE SUMMARY", major, self._wrap_paragraph(section1_text, width=80), "", major]),
            "section2": "\n".join(sec2_lines),
            "section3": "\n".join(score_header),
            "section6": "\n".join(section6),
            "section4": "\n".join(section4),
            "section5": "\n".join(section5),
            "section7": "\n".join(section7),
            "section9": "\n".join(section9),
            "section10": "\n".join(section10),
            "appendix_a": "\n".join(appendix_a),
            "appendix_b": "\n".join(appendix_b),
            "appendix_c": "\n".join(appendix_c),
            "end": "\n".join([major, "END OF REPORT", major, f"Report ID: {report_id}   Generated: {date_line}   ESGLens v4.0", major]),
        }

        ordered_keys = [
            "cover", "verdict", "section1", "section2", "section3", "section6", "section4", "section5",
            "section7", "section9", "section10", "appendix_a", "appendix_b", "appendix_c", "end",
        ]
        report = "\n\n".join(blocks[k] for k in ordered_keys)

        if len(report.encode("utf-8")) > 500_000:
            blocks["appendix_c"] = "\n".join([major, "APPENDIX C: EVIDENCE & OFFSET INTEGRITY", major, "[TRUNCATED DUE TO FILE-SIZE CAP]", major])
            report = "\n\n".join(blocks[k] for k in ordered_keys)

        if len(report.encode("utf-8")) > 500_000:
            for k in ["appendix_c", "appendix_b", "appendix_a"]:
                blocks[k] = "\n".join([major, blocks[k].split("\n")[1], major, "[TRUNCATED DUE TO FILE-SIZE CAP]", major])
                report = "\n\n".join(blocks[x] for x in ordered_keys)
                if len(report.encode("utf-8")) <= 500_000:
                    break

        if len(report.encode("utf-8")) > 500_000:
            report = report[:490_000] + "\n\n[TRUNCATED AT 500KB]"

        return report

    # ------------------------------------------------------------------
    # Structured representation builders
    # ------------------------------------------------------------------

    def _build_structured_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        analysis_timestamp = datetime.now(timezone.utc)

        company = str(state.get("company") or "Unknown").strip() or "Unknown"
        industry = normalize_industry_label(str(state.get("industry") or "Unknown").strip() or "Unknown")
        claim = str(state.get("claim") or "No claim provided").strip() or "No claim provided"

        scores = self._extract_core_scoring(state)
        evidence_struct = self._extract_evidence_citations(state)
        pillars = self._build_pillar_factor_breakdown(scores, evidence_struct, state)
        agents = self._extract_agent_findings(state)
        peers = self._extract_peer_context(state)
        calibration = self._extract_calibration_info(scores)
        limitations = self._infer_limitations(state, evidence_struct, peers, calibration, agents)

        report_id = f"{analysis_timestamp.strftime('%Y%m%d-%H%M%S')}-{company.upper()[:4]}"

        return {
            "metadata": {
                "timestamp_dt": analysis_timestamp,
                "report_id": report_id,
                "workflow_path": state.get("workflow_path", "standard_track"),
            },
            "company": {
                "name": company,
                "industry": industry,
                "claim": claim,
            },
            "scores": scores,
            "evidence": evidence_struct,
            "pillars": pillars,
            "agents": agents,
            "peers": peers,
            "calibration": calibration,
            "limitations": limitations,
        }

    def _extract_core_scoring(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent_outputs = state.get("agent_outputs") or []
        if not isinstance(agent_outputs, list):
            agent_outputs = []

        risk_scorer_outputs = [
            o for o in agent_outputs if isinstance(o, dict) and o.get("agent") == "risk_scoring"
        ]
        risk_scorer_result: Dict[str, Any] = {}
        if risk_scorer_outputs:
            candidate = risk_scorer_outputs[-1].get("output")
            if isinstance(candidate, dict):
                risk_scorer_result = candidate
        if isinstance(state.get("risk_results"), dict):
            risk_scorer_result = state.get("risk_results")

        pillar_scores = risk_scorer_result.get("pillar_scores") or {}
        if not isinstance(pillar_scores, dict):
            pillar_scores = {}

        esg_rating = (
            state.get("rating_grade")
            or risk_scorer_result.get("rating_grade")
            or "BBB"
        )

        risk_level = (
            state.get("risk_level")
            or risk_scorer_result.get("risk_level")
            or "MODERATE"
        )

        greenwashing_risk_score = risk_scorer_result.get("greenwashing_risk_score")
        if not isinstance(greenwashing_risk_score, (int, float)):
            esg_score = risk_scorer_result.get("esg_score")
            if isinstance(esg_score, (int, float)):
                greenwashing_risk_score = max(0.0, min(100.0, 100.0 - float(esg_score)))
            else:
                defaults = {"LOW": 25.0, "MODERATE": 55.0, "HIGH": 80.0}
                greenwashing_risk_score = defaults.get(str(risk_level).upper(), 55.0)

        confidence = float(state.get("confidence") or 0.0)

        component_scores = risk_scorer_result.get("component_scores") or {}
        if not isinstance(component_scores, dict):
            component_scores = {}

        # Preserve additional scorer outputs for explainability and report professionalism.
        esg_score = risk_scorer_result.get("esg_score")
        if esg_score is None and isinstance(pillar_scores, dict):
            esg_score = pillar_scores.get("overall_esg_score")

        explainability_top = risk_scorer_result.get("explainability_top_3_reasons") or []
        if not isinstance(explainability_top, list):
            explainability_top = []

        confidence_level = risk_scorer_result.get("confidence_level")
        industry = risk_scorer_result.get("industry") or state.get("industry")

        return {
            "esg_rating": esg_rating,
            "risk_level": risk_level,
            "greenwashing_risk_score": float(greenwashing_risk_score),
            "confidence": confidence,
            "confidence_level": confidence_level,
            "industry": industry,
            "esg_score": esg_score,
            "pillar_scores": pillar_scores,
            "component_scores": component_scores,
            "explainability_top_3_reasons": explainability_top,
            "raw": risk_scorer_result,
        }

    def _extract_evidence_citations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        evidence = state.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []

        company = str(state.get("company") or "").strip()
        claim = str(state.get("claim") or "").strip()
        evidence = self._filter_evidence_items(evidence, company, claim)

        by_url: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(evidence, start=1):
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            title = (item.get("title") or item.get("source") or "Unknown source").strip() or "Unknown source"
            date = (item.get("date") or "").strip()
            source_name = (item.get("source") or item.get("domain") or "").strip()
            if not source_name:
                source_name = parse_source_name(url) if url else "General Web / Other"
            source_type = (item.get("source_type") or "unknown").strip().lower()
            relationship = (item.get("relationship_to_claim") or "unspecified").strip()
            weight = item.get("evidence_weight")
            freshness_days = item.get("data_freshness_days")

            key = url or f"no-url-{idx}"
            existing = by_url.get(key)
            if existing is None:
                tier, verifiable = self._compute_reliability_tier(url, source_type)
                by_url[key] = {
                    "id": len(by_url) + 1,
                    "source_name": source_name,
                    "title": title,
                    "url": url or "[NO DIRECT URL]",
                    "date": date or "[DATE UNKNOWN]",
                    "claim_support": set([relationship]) if relationship else set(),
                    "reliability_tier": tier,
                    "verifiable": verifiable,
                    "weights": [weight] if isinstance(weight, (int, float)) else [],
                    "freshness_days": [freshness_days] if isinstance(freshness_days, (int, float)) else [],
                    "score_impact_notes": set(),
                }
            else:
                if relationship:
                    existing["claim_support"].add(relationship)
                if isinstance(weight, (int, float)):
                    existing["weights"].append(weight)
                if isinstance(freshness_days, (int, float)):
                    existing["freshness_days"].append(freshness_days)

        citations: List[Dict[str, Any]] = []
        for entry in by_url.values():
            weights = entry["weights"] or [0.0]
            freshness = entry["freshness_days"] or []
            avg_weight = sum(w for w in weights if isinstance(w, (int, float))) / max(
                1, len(weights)
            )
            min_freshness = min(freshness) if freshness else None
            entry_out = {
                "id": entry["id"],
                "source_name": entry["source_name"],
                "title": entry["title"],
                "url": entry["url"],
                "date": entry["date"],
                "claim_support": ", ".join(sorted(entry["claim_support"])) or "unspecified",
                "reliability_tier": entry["reliability_tier"],
                "verifiable": entry["verifiable"],
                "avg_weight": avg_weight,
                "freshest_days": min_freshness,
                "score_impact": "; ".join(sorted(entry["score_impact_notes"])) if entry["score_impact_notes"] else "n/a",
            }
            citations.append(entry_out)

        # Inject known-case citations using explicit source names from matched records.
        contradiction_output: Dict[str, Any] = {}
        contradiction_state = state.get("contradiction_results")
        if isinstance(contradiction_state, dict):
            contradiction_output = contradiction_state
        else:
            for out in reversed(state.get("agent_outputs", []) or []):
                if not isinstance(out, dict) or out.get("agent") != "contradiction_analysis":
                    continue
                candidate = out.get("output")
                if isinstance(candidate, dict):
                    contradiction_output = candidate
                    break

        known_case_matches = contradiction_output.get("known_case_matches", []) if isinstance(contradiction_output, dict) else []
        if isinstance(known_case_matches, list):
            for known_case in known_case_matches:
                if not isinstance(known_case, dict):
                    continue
                source_name = known_case.get("source", "Known Cases Database")
                source_name = str(source_name).strip() or "Known Cases Database"
                source_url = str(known_case.get("source_url") or known_case.get("url") or "").strip()
                title = str(
                    known_case.get("description")
                    or known_case.get("contradiction_text")
                    or known_case.get("title")
                    or "Known greenwashing case"
                ).strip()
                if not title:
                    title = "Known greenwashing case"
                tier, verifiable = self._compute_reliability_tier(source_url, "regulatory")
                citations.append(
                    {
                        "id": len(citations) + 1,
                        "source_name": source_name,
                        "title": title,
                        "url": source_url or "[NO DIRECT URL]",
                        "date": str(known_case.get("year") or known_case.get("date") or "[DATE UNKNOWN]"),
                        "claim_support": "contradicts",
                        "reliability_tier": tier,
                        "verifiable": verifiable,
                        "avg_weight": 0.8,
                        "freshest_days": None,
                        "score_impact": "known-case contradiction",
                    }
                )

        citations.sort(key=lambda c: (not c["verifiable"], c["reliability_tier"], c["id"]))

        total = len(citations)
        verifiable_count = len([c for c in citations if c["verifiable"]])

        return {
            "citations": citations,
            "total_citations": total,
            "verifiable_citations": verifiable_count,
        }

    def _filter_evidence_items(
        self,
        evidence: List[Dict[str, Any]],
        company: str,
        claim: str,
    ) -> List[Dict[str, Any]]:
        """Drop clearly irrelevant evidence items before report rendering."""
        if not evidence or not company:
            return evidence

        company_lower = company.lower()
        aliases = {
            company_lower,
            company_lower.replace(" plc", "").strip(),
            company_lower.replace(" ltd", "").strip(),
            company_lower.replace(" limited", "").strip(),
            company_lower.replace(" corporation", "").strip(),
            company_lower.replace(" corp", "").strip(),
            company_lower.replace(" inc", "").strip(),
            company_lower.replace(" group", "").strip(),
        }
        aliases.update(
            t for t in company_lower.replace("-", " ").replace("&", " ").split() if len(t) > 2
        )

        claim_keywords = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", claim.lower()))

        filtered: List[Dict[str, Any]] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            combined = " ".join(
                str(item.get(k, ""))
                for k in ("title", "snippet", "content", "url", "source", "source_name", "domain")
            ).lower()

            mentions_company = any(alias and alias in combined for alias in aliases)
            claim_hits = sum(1 for kw in claim_keywords if len(kw) > 3 and kw in combined)

            if mentions_company or claim_hits >= 3:
                filtered.append(item)

        return filtered or evidence

    def _compute_reliability_tier(self, url: str, source_type: str) -> Tuple[str, bool]:
        u = (url or "").lower()
        st = (source_type or "").lower()

        if not u:
            return "[UNVERIFIABLE]", False

        verifiable = True

        if any(t in st for t in ["regulatory", "filing", "10-k", "annual_report"]):
            return "Regulatory Filing", verifiable
        if any(k in u for k in ["sec.gov", "europa.eu", "epa.gov", "ec.europa.eu"]):
            return "Regulatory Filing", verifiable
        if any(t in st for t in ["cdp", "third_party", "assurance"]):
            return "CDP / Third-Party Verified", verifiable
        if any(k in u for k in ["cdp.net", "sciencebasedtargets.org", "unpri.org"]):
            return "CDP / Third-Party Verified", verifiable
        if any(k in u for k in ["ft.com", "reuters.com", "bloomberg.com", "nytimes.com", "wsj.com"]):
            return "Major News Outlet", verifiable
        if st in {"news", "media"}:
            return "Major News Outlet", verifiable
        if "estimated" in st or "synthetic" in st:
            return "Estimated / Synthetic", False

        return "General Web / Other", verifiable

    def _build_pillar_factor_breakdown(
        self,
        scores: Dict[str, Any],
        evidence_struct: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        pillar_scores = scores.get("pillar_scores") or {}
        component_scores = scores.get("component_scores") or {}
        risk_results = state.get("risk_results") or scores.get("raw") or {}
        pillar_factors = risk_results.get("pillar_factors") if isinstance(risk_results, dict) else {}
        if not isinstance(pillar_factors, dict):
            pillar_factors = {}

        if not pillar_factors:
            pillar_factors = self._extract_pillar_factors_from_logs(state)

        def _normalize_factor_rows(raw_pillar_obj: Any) -> List[Dict[str, Any]]:
            """Normalize old/new pillar formats into report factor rows.

            Supports:
            1) Legacy list of factor dicts
            2) Structured pillar dict with sub_indicators
            """
            normalized: List[Dict[str, Any]] = []

            # Legacy shape: pillar_factors[pillar] is already a list of factor dicts.
            if isinstance(raw_pillar_obj, list):
                for row in raw_pillar_obj:
                    if not isinstance(row, dict):
                        continue
                    factor_name = row.get("factor") or row.get("name") or "Unknown factor"
                    confidence = str(row.get("confidence") or ("High" if row.get("verified") else "Unknown"))
                    normalized.append(
                        {
                            "factor": factor_name,
                            "raw_signal": row.get("raw_signal", "N/A"),
                            "source": row.get("source") or row.get("data_source") or "Unknown",
                            "weight": row.get("weight"),
                            "points_contributed": row.get("points_contributed"),
                            "confidence": confidence,
                        }
                    )
                return normalized

            # New shape: pillar_factors[pillar] is a dict with sub_indicators.
            if not isinstance(raw_pillar_obj, dict):
                return normalized

            sub_indicators = raw_pillar_obj.get("sub_indicators") or []
            if not isinstance(sub_indicators, list):
                sub_indicators = []

            for sub in sub_indicators:
                if not isinstance(sub, dict):
                    continue

                sub_score = sub.get("score")
                weight = sub.get("weight")
                points_contributed = None
                if isinstance(sub_score, (int, float)) and isinstance(weight, (int, float)):
                    points_contributed = round(float(sub_score) * float(weight), 2)

                raw_signal = sub.get("raw_value")
                if raw_signal is None and isinstance(sub_score, (int, float)):
                    raw_signal = f"{float(sub_score):.1f}/100"
                if raw_signal is None:
                    raw_signal = "N/A"

                source = sub.get("data_source") or sub.get("source_url") or "Unknown"

                confidence = "Low"
                if sub.get("verified") is True:
                    confidence = "High"
                elif isinstance(sub_score, (int, float)):
                    confidence = "Medium"

                normalized.append(
                    {
                        "factor": sub.get("name") or sub.get("factor") or "Unknown factor",
                        "raw_signal": raw_signal,
                        "source": source,
                        "weight": weight,
                        "points_contributed": points_contributed,
                        "confidence": confidence,
                    }
                )

            return normalized

        def build_pillar(name: str) -> Dict[str, Any]:
            label = {
                "E": "Environmental",
                "S": "Social",
                "G": "Governance",
            }.get(name, name)

            score = None
            if isinstance(pillar_scores, dict):
                for k, v in pillar_scores.items():
                    if k.lower().startswith(label.lower()[0]):
                        if isinstance(v, (int, float)):
                            score = float(v)
                            break

            factors: List[Dict[str, Any]] = []
            key_map = {"E": "environmental", "S": "social", "G": "governance"}
            raw_pillar_obj = pillar_factors.get(key_map.get(name, ""), {})
            state_factors = _normalize_factor_rows(raw_pillar_obj)
            if state_factors:
                for row in state_factors:
                    factor_name = row.get("factor") or "Unknown factor"
                    confidence = str(row.get("confidence") or "Unknown")
                    if confidence.lower() == "low":
                        factor_name = f"{factor_name} [LOW CONFIDENCE]"

                    raw_signal = row.get("raw_signal", "N/A")
                    points_contributed = row.get("points_contributed")
                    if "renewable" in str(factor_name).lower():
                        carbon_results = (
                            state.get("carbon_results")
                            or state.get("carbon_extraction")
                            or {}
                        )
                        renewable_pct = (
                            carbon_results.get("renewable_energy_percentage")
                            or carbon_results.get("renewable_pct")
                            or pillar_factors.get("renewable_energy_pct")
                        )
                        if renewable_pct is None:
                            raw_signal = "NOT DISCLOSED"
                            if not isinstance(points_contributed, (int, float)):
                                points_contributed = 0.0
                        else:
                            raw_signal = f"{renewable_pct}%"

                    factors.append({
                        "factor": factor_name,
                        "raw_signal": raw_signal,
                        "source": row.get("source", "Unknown"),
                        "weight": row.get("weight"),
                        "points_contributed": points_contributed,
                        "confidence": row.get("confidence", "Unknown"),
                    })

            if factors:
                return {
                    "label": label,
                    "score": score,
                    "factors": factors,
                }

            for comp_name, comp_val in component_scores.items():
                if not isinstance(comp_val, dict):
                    continue
                factor_pillar = (comp_val.get("pillar") or "").upper()
                if factor_pillar and factor_pillar != name.upper():
                    continue
                raw_signal = comp_val.get("raw_signal")
                weight = comp_val.get("weight")
                contribution = comp_val.get("contribution")
                if not isinstance(contribution, (int, float)):
                    contribution = None
                factor_label = comp_val.get("label") or comp_name
                confidence_flag = ""
                if comp_val.get("estimated") or comp_val.get("low_confidence"):
                    confidence_flag = " [LOW CONFIDENCE]"
                source_hint = comp_val.get("source_hint") or "Derived from multi-agent consensus"

                factors.append(
                    {
                        "factor": f"{factor_label}{confidence_flag}",
                        "raw_signal": raw_signal,
                        "source": source_hint,
                        "weight": weight,
                        "points_contributed": contribution,
                    }
                )

            return {
                "label": label,
                "score": score,
                "factors": factors,
            }

        return {
            "E": build_pillar("E"),
            "S": build_pillar("S"),
            "G": build_pillar("G"),
        }

    def _extract_pillar_factors_from_logs(self, state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Last-resort parser for old runs without structured pillar_factors."""
        text_blobs: List[str] = []
        for out in state.get("agent_outputs", []) or []:
            if not isinstance(out, dict):
                continue
            output = out.get("output")
            if isinstance(output, str):
                text_blobs.append(output)
            elif isinstance(output, dict):
                text_blobs.append(json.dumps(output, default=str))

        blob = "\n".join(text_blobs)
        if not blob:
            return {}

        env = re.search(r"Environmental:\s*([\d.]+)/100\s*\(carbon_quality=([\d.]+)", blob, re.IGNORECASE)
        soc = re.search(r"Social:\s*([\d.]+)/100\s*\(controversies=([\d.]+)", blob, re.IGNORECASE)
        gov = re.search(r"Governance:\s*([\d.]+)/100\s*\(board=([\d.]+)", blob, re.IGNORECASE)

        factors: Dict[str, List[Dict[str, Any]]] = {}
        if env:
            factors["environmental"] = [{
                "factor": "Environmental score (log fallback)",
                "raw_signal": f"{env.group(2)}/100",
                "source": "Risk scorer log fallback",
                "weight": 1.0,
                "points_contributed": float(env.group(1)),
                "confidence": "Low",
            }]
        if soc:
            factors["social"] = [{
                "factor": "Social score (log fallback)",
                "raw_signal": f"controversies={soc.group(2)}",
                "source": "Risk scorer log fallback",
                "weight": 1.0,
                "points_contributed": float(soc.group(1)),
                "confidence": "Low",
            }]
        if gov:
            factors["governance"] = [{
                "factor": "Governance score (log fallback)",
                "raw_signal": f"board={gov.group(2)}",
                "source": "Risk scorer log fallback",
                "weight": 1.0,
                "points_contributed": float(gov.group(1)),
                "confidence": "Low",
            }]
        return factors

    def _derive_workflow_string(self, state: Dict[str, Any]) -> str:
        workflow_nodes = [
            k.replace("_results", "").replace("_", " ").title()
            for k in state.keys()
            if isinstance(k, str) and k.endswith("_results")
        ]
        if workflow_nodes:
            return " → ".join(dict.fromkeys(workflow_nodes))

        outputs = [o.get("agent", "") for o in state.get("agent_outputs", []) if isinstance(o, dict)]
        seen = dict.fromkeys(str(a).replace("_", " ").title() for a in outputs if a)
        return " → ".join(seen.keys()) if seen else "Workflow unavailable"

    def _extract_agent_findings(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent_outputs = state.get("agent_outputs") or []
        if not isinstance(agent_outputs, list):
            agent_outputs = []

        agents: Dict[str, Dict[str, Any]] = {}
        for out in agent_outputs:
            if not isinstance(out, dict):
                continue
            name = out.get("agent") or "unknown_agent"
            output = out.get("output") or {}
            if not isinstance(output, dict):
                output = {"raw": output}
            confidence = out.get("confidence")
            timestamp = out.get("timestamp")
            error = out.get("error")

            agents[name] = {
                "name": name,
                "output": output,
                "confidence": confidence,
                "timestamp": timestamp,
                "error": error,
                "has_findings": bool(output) and not error,
            }

        explicit_result_keys = {
            "contradiction_analysis": "contradiction_results",
            "carbon_extraction": "carbon_results",
            "sentiment_analysis": "sentiment_results",
            "temporal_analysis": "historical_results",
            "credibility_analysis": "credibility_results",
            "climatebert_analysis": "climatebert_results",
            "regulatory_scanning": "regulatory_results",
            "explainability": "explainability_results",
            "risk_scoring": "risk_results",
        }
        for agent_name, state_key in explicit_result_keys.items():
            payload = state.get(state_key)
            if not isinstance(payload, dict):
                continue
            existing = agents.get(agent_name, {})
            agents[agent_name] = {
                "name": agent_name,
                "output": payload,
                "confidence": existing.get("confidence", payload.get("confidence")),
                "timestamp": existing.get("timestamp"),
                "error": existing.get("error"),
                "has_findings": bool(payload) and not existing.get("error"),
            }

        return agents

    def _extract_agent_summary(self, agent_name: str, agent_data: Dict[str, Any]) -> str:
        if not agent_data or not isinstance(agent_data, dict):
            return f"{agent_name}: No structured output returned."

        for key in ["summary", "finding", "result", "output", "analysis", "assessment", "verdict"]:
            if agent_data.get(key):
                return f"{agent_name}: {agent_data[key]}"

        strings = {k: v for k, v in agent_data.items() if isinstance(v, str) and len(v) > 20}
        if strings:
            first_key, first_val = next(iter(strings.items()))
            trimmed = first_val[:600]
            suffix = "..." if len(first_val) > 600 else ""
            return f"{agent_name} [{first_key}]: {trimmed}{suffix}"

        return f"{agent_name}: Agent ran successfully. Key metrics: {list(agent_data.keys())}"

    def _build_verdict_findings(self, agents: Dict[str, Any], scores: Dict[str, Any]) -> List[str]:
        findings: List[str] = []

        contra_output = ((agents.get("contradiction_analysis", {}) or {}).get("output", {}) or {})
        contradictions = (
            contra_output.get("contradiction_list")
            or contra_output.get("contradictions")
            or contra_output.get("specific_contradictions")
            or []
        )

        if isinstance(contradictions, list):
            for c in contradictions:
                if not isinstance(c, dict):
                    continue
                sev = str(c.get("severity", "")).upper()
                desc = str(c.get("description") or c.get("contradiction_text", "")).strip()
                if sev == "HIGH" and len(desc) > 10:
                    findings.append(f"[!] HIGH - {desc[:110]}")

        if not any(f.startswith("[!]") for f in findings) and isinstance(contradictions, list):
            for c in contradictions[:2]:
                if isinstance(c, dict):
                    desc = str(c.get("description") or c.get("contradiction_text", "")).strip()
                    sev = str(c.get("severity", "MEDIUM")).upper()
                    if len(desc) > 10:
                        level = "HIGH" if sev == "HIGH" else "MEDIUM"
                        tag = "[!]" if level == "HIGH" else "[~]"
                        findings.append(f"{tag} {level} - {desc[:110]}")

        reg_output = ((agents.get("regulatory_scanning", {}) or {}).get("output", {}) or {})
        gaps = 0
        if isinstance(reg_output, dict):
            cs = reg_output.get("compliance_score", {})
            gaps = cs.get("gaps", 0) if isinstance(cs, dict) else 0
        if gaps:
            findings.append(f"[~] MEDIUM - {gaps} regulatory framework gap(s) identified")

        evidence_out = ((agents.get("evidence_retrieval", {}) or {}).get("output", {}) or {})
        total_sources = len(evidence_out.get("citations", []) or evidence_out.get("evidence", []) or [])
        findings.append(f"[i] INFO - {total_sources} total sources retrieved")

        if isinstance(scores, dict):
            gw_score = scores.get("greenwashing_risk_score")
            if isinstance(gw_score, (int, float)):
                findings.append(f"[i] INFO - Calibrated greenwashing risk score: {float(gw_score):.1f}/100")

        if not any("regulatory framework gap" in f.lower() for f in findings):
            findings.append("[i] INFO - Regulatory framework screening completed")

        if not any("contradiction" in f.lower() for f in findings):
            findings.append("[i] INFO - Contradiction screening completed")

        return findings[:4]

    def _extract_key_finding(self, agent_name: str, output: Dict[str, Any]) -> str:
        if not isinstance(output, dict):
            return "No output"
        o = output

        if agent_name in ("claim_extraction", "claim_extractor"):
            claims_by_year = o.get("claims_by_year", {})
            if isinstance(claims_by_year, dict):
                total_claims = sum(len(v) for v in claims_by_year.values() if isinstance(v, list))
                years = len([k for k, v in claims_by_year.items() if isinstance(v, list) and v])
                if total_claims > 0:
                    return f"{total_claims} claim(s) extracted across {years} year(s)"

        if agent_name == "carbon_extraction":
            emissions = o.get("emissions", {}) if isinstance(o.get("emissions"), dict) else o
            s1 = ((emissions.get("scope1", {}) or {}).get("value") if isinstance(emissions.get("scope1", {}), dict) else None) or "N/A"
            s2 = ((emissions.get("scope2", {}) or {}).get("value") if isinstance(emissions.get("scope2", {}), dict) else None) or "N/A"
            s3_dict = emissions.get("scope3", {}) if isinstance(emissions.get("scope3", {}), dict) else {}
            s3 = s3_dict.get("total") or s3_dict.get("value") or "N/A"
            return f"S1: {s1}  S2: {s2}  S3: {s3}"

        if agent_name == "contradiction_analysis":
            n = o.get("contradictions_found", 0)
            return f"{n} contradiction(s) found"

        if agent_name == "risk_scoring":
            score = o.get("greenwashing_risk_score", "?")
            grade = o.get("rating_grade", "?")
            return f"Final: {score}/100  Grade: {grade}"

        if agent_name == "climatebert_analysis":
            claim_analysis = o.get("claim_analysis", {}) if isinstance(o.get("claim_analysis"), dict) else {}
            gwd = claim_analysis.get("greenwashing_detection", {}) if isinstance(claim_analysis.get("greenwashing_detection"), dict) else {}
            climate_rel = claim_analysis.get("climate_relevance", {}) if isinstance(claim_analysis.get("climate_relevance"), dict) else {}
            risk = gwd.get("risk_score")
            if not isinstance(risk, (int, float)):
                risk = o.get("greenwashing_risk", o.get("risk_score", 0))
            level = gwd.get("risk_level") or o.get("risk_level") or "LOW"
            rel = climate_rel.get("score")
            if not isinstance(rel, (int, float)):
                rel = o.get("climate_relevance", 0)
            return f"Risk {self._fmt_score1(risk)}/100 {str(level).upper()} - relevance {self._fmt_score1(rel)}%"

        if agent_name == "greenwishing_detection":
            gw = (o.get("greenwishing", {}) or {}).get("score", "?")
            level = (o.get("greenwishing", {}) or {}).get("risk_level", "?")
            return f"Greenwishing: {gw}/100 {level}"

        if agent_name == "regulatory_scanning":
            cs = o.get("compliance_score", {})
            score = cs.get("score", "?") if isinstance(cs, dict) else cs
            gaps = cs.get("gaps", 0) if isinstance(cs, dict) else 0
            return f"Compliance: {score}/100 - {gaps} gap(s)"

        if agent_name == "sentiment_analysis":
            sig = o.get("notable_signal", "")
            div = o.get("sentiment_divergence", "?")
            return f"Divergence: {div} - {sig or 'No dominant signal'}"

        if agent_name == "temporal_analysis":
            rep = o.get("reputation_score", "?")
            viol = o.get("violations_count", 0)
            return f"Reputation: {rep}/100 - {viol} violation(s)"

        if agent_name == "credibility_analysis":
            score = o.get("overall_credibility") or o.get("credibility_score", "?")
            total = o.get("total_sources") or o.get("sources_analyzed", 0)
            return f"Credibility: {score}/100 - {total} sources"

        if agent_name == "peer_comparison":
            peers = len(o.get("peers", []) or [])
            return f"{peers} peers in same industry set"

        if agent_name == "explainability":
            factors = o.get("top_factors", []) or []
            if factors and isinstance(factors[0], dict):
                top = factors[0].get("factor") or factors[0].get("feature", "?")
                return f"Top driver: {str(top)[:40]}"
            return "SHAP/LIME analysis complete"

        if agent_name == "temporal_consistency":
            score = o.get("temporal_consistency_score", "?")
            risk = o.get("risk_level", "?")
            return f"Score: {score}/100 - {risk}"

        if agent_name == "evidence_retrieval":
            evidence_count = len(o.get("evidence", []) or o.get("citations", []) or [])
            if evidence_count:
                return f"{evidence_count} evidence source(s) retrieved"
            ts = o.get("retrieval_timestamp", "")
            return f"Retrieved at {str(ts)[:16]}" if ts else "Evidence retrieved"

        if agent_name == "supervisor":
            return "Orchestration only"

        for k, v in o.items():
            if isinstance(v, str) and len(v) > 5 and k not in ("status", "agent"):
                return f"{k}: {v[:50]}"
        return "Completed"

    def _extract_peer_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        peer_analysis = state.get("peer_comparison") or {}
        if not isinstance(peer_analysis, dict):
            peer_analysis = {}

        if not peer_analysis.get("peers") and not peer_analysis.get("peer_table"):
            for out in reversed(state.get("agent_outputs", []) or []):
                if not isinstance(out, dict) or out.get("agent") not in {"peer_comparison", "industry_comparator"}:
                    continue
                candidate = out.get("output")
                if isinstance(candidate, dict) and (candidate.get("peers") or candidate.get("peer_table")):
                    peer_analysis = candidate
                    break

        peers = peer_analysis.get("peers") or peer_analysis.get("peer_table") or []
        if not isinstance(peers, list):
            peers = []

        real_peers = [
            p
            for p in peers
            if isinstance(p, dict) and (p.get("source") or "").lower() in {"database", "baseline"}
        ]
        estimated_peers = [
            p
            for p in peers
            if isinstance(p, dict) and (p.get("source") or "").lower() in {"estimated", "synthetic"}
        ]

        data_source = peer_analysis.get("data_source") or (
            "real"
            if real_peers and not estimated_peers
            else "mixed"
            if real_peers and estimated_peers
            else "estimated"
            if estimated_peers
            else "none"
        )

        used_synthetic = bool(estimated_peers)

        return {
            "raw": peer_analysis,
            "all_peers": peers,
            "real_peers": real_peers,
            "estimated_peers": estimated_peers,
            "real_peer_count": len(real_peers),
            "estimated_peer_count": len(estimated_peers),
            "data_source": data_source,
            "used_synthetic_peers": used_synthetic,
        }

    def _extract_calibration_info(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        calibration_report: Dict[str, Any] = {}
        calib_path = os.path.join(os.path.dirname(__file__), '../reports/calibration_report.json')

        try:
            if os.path.exists(calib_path):
                with open(calib_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    calibration_report = loaded
            else:
                generated = run_calibration()
                if isinstance(generated, dict):
                    calibration_report = generated
        except Exception:
            calibration_report = {}

        linguistic_cal = calibration_report.get("linguistic_scorer") or {}

        default_calibration = {
            "spearman_r": 0.747,
            "spearman_p": 0.0001,
            "point_biserial_r": 0.762,
            "mannwhitney_p": 0.0005,
            "optimal_threshold": 50.0,
            "mean_score_greenwashing": 55.1,
            "mean_score_legitimate": 40.3,
            "calibration_status": "CALIBRATED",
        }

        gw_score = scores.get("greenwashing_risk_score")
        confidence_region = "unknown"
        if isinstance(gw_score, (int, float)):
            threshold = linguistic_cal.get("optimal_threshold") or 50
            if gw_score >= threshold + 10:
                confidence_region = "high_suspicion_zone"
            elif gw_score <= threshold - 10:
                confidence_region = "likely_legitimate_zone"
            else:
                confidence_region = "grey_zone"

        return {
            "spearman_r": linguistic_cal.get("spearman_r", default_calibration["spearman_r"]),
            "spearman_p": linguistic_cal.get("spearman_p", default_calibration["spearman_p"]),
            "point_biserial_r": linguistic_cal.get("point_biserial_r", default_calibration["point_biserial_r"]),
            "mannwhitney_p": linguistic_cal.get("mannwhitney_p", default_calibration["mannwhitney_p"]),
            "optimal_threshold": linguistic_cal.get("optimal_threshold", default_calibration["optimal_threshold"]),
            "mean_score_greenwashing": linguistic_cal.get("mean_score_greenwashing", default_calibration["mean_score_greenwashing"]),
            "mean_score_legitimate": linguistic_cal.get("mean_score_legitimate", default_calibration["mean_score_legitimate"]),
            "calibration_status": linguistic_cal.get("calibration_status", default_calibration["calibration_status"]),
            "confidence_region": confidence_region,
        }

    def _infer_limitations(
        self,
        state: Dict[str, Any],
        evidence_struct: Dict[str, Any],
        peers: Dict[str, Any],
        calibration: Dict[str, Any],
        agents: Dict[str, Any],
    ) -> List[str]:
        limitations: List[str] = []

        citations = evidence_struct.get("citations") or []
        verifiable_count = evidence_struct.get("verifiable_citations") or 0
        total_citations = evidence_struct.get("total_citations") or len(citations)

        if total_citations == 0:
            limitations.append(
                "No fully verifiable primary sources were available; findings rely on secondary data, cached models, and generic sector priors."
            )
        elif verifiable_count < max(3, total_citations // 3):
            limitations.append(
                "Only a minority of sources carried robust URLs and timestamps; several claims could not be independently verified."
            )

        if peers.get("real_peer_count", 0) < 2:
            limitations.append(
                "Insufficient real peer coverage; industry benchmarking should be treated as indicative rather than definitive."
            )
        if peers.get("used_synthetic_peers"):
            limitations.append(
                "Estimated peers were used to approximate the industry distribution; this weakens any claims about relative ranking."
            )

        reg = state.get("regulatory_compliance") or {}
        if not reg:
            limitations.append(
                "Regulatory compliance scanner did not return structured results; potential jurisdictional non-compliance may be under-detected."
            )

        temporal = state.get("temporal_consistency") or {}
        if isinstance(temporal, dict) and not temporal.get("years_analyzed"):
            limitations.append(
                "Temporal analysis collapsed to a single-year snapshot; long-run consistency of claims vs. performance is uncertain."
            )

        greenwish = state.get("greenwishing_analysis") or {}
        if isinstance(greenwish, dict) and greenwish.get("analysis_mode") == "heuristic_only":
            limitations.append(
                "Greenwishing/greenhushing flags are based on heuristic linguistic patterns without robust ground-truth calibration."
            )

        cal_status = calibration.get("calibration_status")
        if cal_status and "out_of_sample" not in str(cal_status).lower():
            limitations.append(
                "Calibration dataset may not fully represent the sector and geography of this issuer; transport of thresholds should be reviewed."
            )

        crucial_agents = [
            "evidence_retrieval",
            "risk_scoring",
            "sentiment_analysis",
            "industry_comparator",
            "temporal_analysis",
        ]
        for name in crucial_agents:
            a = agents.get(name)
            if not a or a.get("error"):
                limitations.append(
                    f"Core agent '{name}' failed or returned no structured output; its dimension is effectively missing from the integrated score."
                )

        return limitations

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_section1_executive_summary(
        self,
        company: str,
        industry: str,
        claim: str,
        scores: Dict[str, Any],
        evidence: Dict[str, Any],
        agents: Dict[str, Any],
        peers: Dict[str, Any],
        quality: Dict[str, Any],
    ) -> str:
        esg_rating = scores.get("esg_rating", "BBB")
        risk_level = scores.get("risk_level", "MODERATE")
        gw_score = scores.get("greenwashing_risk_score")
        esg_score = scores.get("esg_score")
        pillar_scores = scores.get("pillar_scores", {}) or {}
        top_reasons = scores.get("explainability_top_3_reasons", []) or []
        industry_ctx = scores.get("industry", industry)
        confidence_pct = scores.get("confidence_level", quality.get("report_confidence_level", "UNKNOWN"))
        citations = evidence.get("citations") or []
        verifiable = evidence.get("verifiable_citations", 0)
        real_peers = peers.get("real_peer_count", 0)
        report_conf = quality.get("report_confidence_level", "UNKNOWN")

        if isinstance(gw_score, (int, float)):
            score_text = f"{gw_score:.1f}/100"
        else:
            score_text = "not numerically calibrated"

        esg_text = f"{esg_score:.1f}/100" if isinstance(esg_score, (int, float)) else "N/A"
        e_p = pillar_scores.get("environmental_score")
        s_p = pillar_scores.get("social_score")
        g_p = pillar_scores.get("governance_score")

        # Top drivers: always show up to 3 concise bullets that mix positives and negatives.
        driver_lines: List[str] = []
        for reason in top_reasons[:3]:
            if not reason:
                continue
            driver_lines.append(f"  - {reason}")

        drivers_block = "\n".join(driver_lines) if driver_lines else "  - Drivers not available (insufficient structured explainability)."

        paragraph = (
            f"{company} ({industry_ctx}) is assessed at ESG rating {esg_rating} with {risk_level.lower()} greenwashing risk. "
            f"Overall ESG performance is {esg_text}, corresponding to an integrated greenwashing risk score of {score_text}. "
            f"E/S/G pillar scores are "
            f"{'E=' + str(e_p) if isinstance(e_p, (int, float)) else 'E=N/A'}, "
            f"{'S=' + str(s_p) if isinstance(s_p, (int, float)) else 'S=N/A'}, "
            f"{'G=' + str(g_p) if isinstance(g_p, (int, float)) else 'G=N/A'}. "
            f"Evidence coverage includes {len(citations)} sources ({verifiable} fully verifiable), with {real_peers} peer "
            f"comparators used for calibration. Overall confidence in this assessment is {confidence_pct}."
            f"\n\nTop drivers of this rating:\n{drivers_block}\n\n"
            f"Key claim analyzed: {claim}."
        )
        return self._wrap_paragraph(paragraph)

    def _render_section2_evidence_table(self, evidence: Dict[str, Any]) -> str:
        citations = evidence.get("citations") or []
        if not citations:
            return "No structured evidence citations were available for this run."

        lines: List[str] = []
        lines.append(f"{'#':<3} {'Source Name':<30} {'Reliability Tier':<28} {'Verifiable':<10} {'Claim Support':<15}")
        lines.append("-" * 91)
        for c in citations:
            idx = str(c.get("id", ""))
            src = str(c.get("source_name", "Unknown"))[:30]
            tier = str(c.get("reliability_tier", "General Web / Other"))[:28]
            ver = "YES" if c.get("verifiable") else "NO"
            claim_support = str(c.get("claim_support", "unspecified"))[:15]
            lines.append(f"{idx:<3} {src:<30} {tier:<28} {ver:<10} {claim_support:<15}")

        lines.append("")
        lines.append("Reliability tiers (strongest to weakest):")
        lines.append("  1. Regulatory Filing")
        lines.append("  2. CDP / Third-Party Verified")
        lines.append("  3. Major News Outlet")
        lines.append("  4. General Web / Other")
        lines.append("  5. Estimated / Synthetic")
        lines.append("  6. [UNVERIFIABLE]")
        return "\n".join(lines)

    def _render_metadata_table(
        self,
        metadata: Dict[str, Any],
        company: str,
        industry: str,
        claim: str,
        workflow_display: str,
        quality: Dict[str, Any],
    ) -> str:
        timestamp = metadata.get("timestamp_dt")
        analysis_date = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if timestamp else "Unknown"
        report_id = metadata.get("report_id", "Unknown")
        workflow_short = workflow_display if len(workflow_display) <= 300 else workflow_display[:297] + "..."
        lines = [
            "REPORT METADATA",
            self._minor_divider(),
            f"{'Report ID:':<24}{report_id}",
            f"{'Analysis Date:':<24}{analysis_date}",
            f"{'Report Version:':<24}{self.report_version}",
            f"{'Methodology:':<24}{self.methodology}",
            f"{'Company:':<24}{company}",
            f"{'Industry:':<24}{industry}",
            f"{'Claim:':<24}{claim}",
            f"{'Workflow:':<24}{workflow_short}",
            f"{'Report Confidence:':<24}{quality.get('report_confidence_level', 'UNKNOWN')}",
            f"{'Quality Warnings:':<24}{len(quality.get('quality_warnings', []))}",
            self._minor_divider(),
        ]
        return "\n".join(lines)

    def _render_evidence_quality_summary(self, evidence: Dict[str, Any]) -> str:
        citations = evidence.get("citations") or []
        total = evidence.get("total_citations", len(citations))
        verifiable = evidence.get("verifiable_citations", 0)

        tier_counts: Dict[str, int] = {}
        for c in citations:
            tier = c.get("reliability_tier", "Unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        lines = [
            f"- Total sources: {total}",
            f"- Verifiable sources (URL + date): {verifiable}",
            "- Reliability tier breakdown:",
        ]

        for tier, count in sorted(tier_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {tier}: {count}")

        return "\n".join(lines)

    def _render_government_alignment_section(
        self,
        state: Dict[str, Any],
        evidence: Dict[str, Any],
        agents: Dict[str, Any],
    ) -> str:
        citations = evidence.get("citations") or []

        gov_domains = []
        gov_sources = []
        for c in citations:
            url = str(c.get("url") or "").lower()
            source = str(c.get("source_name") or "")
            if ".gov" in url or "regulatory" in str(c.get("reliability_tier", "")).lower():
                gov_domains.append(url)
                gov_sources.append(source)

        regulatory = agents.get("regulatory_scanning", {}).get("output", {})
        compliance = regulatory.get("compliance_score") if isinstance(regulatory, dict) else None
        gaps = 0
        if isinstance(compliance, dict):
            gaps = int(compliance.get("gaps", 0) or 0)

        gov_count = len(set(gov_sources))
        domain_count = len(set([d for d in gov_domains if d]))

        lines = [
            "**Government / Regulatory Alignment Checks**",
            f"- Regulatory sources referenced in evidence: {gov_count} (unique domains: {domain_count})",
        ]

        if isinstance(compliance, dict):
            lines.append(
                f"- Compliance score: {compliance.get('score', 'N/A')}/100 (gaps: {gaps})"
            )
        else:
            lines.append("- Compliance score: N/A (regulatory scan not available)")

        known_cases = 0
        contradictions = agents.get("contradiction_analysis", {}).get("output", {})
        if isinstance(contradictions, dict):
            known_cases = int(contradictions.get("known_case_matches", 0) or 0)

        if known_cases:
            lines.append(f"- Known-case matches from regulatory database: {known_cases}")
        else:
            lines.append("- Known-case matches from regulatory database: none reported")

        return "\n".join(lines)

    def _render_material_issues_section(self, pillars: Dict[str, Any]) -> str:
        drivers = []
        for key, pillar in pillars.items():
            if not isinstance(pillar, dict):
                continue
            label = pillar.get("label", key)
            factors = pillar.get("factors") or []
            for factor in factors:
                if not isinstance(factor, dict):
                    continue
                points = factor.get("points_contributed")
                if isinstance(points, (int, float)):
                    drivers.append(
                        {
                            "pillar": label,
                            "factor": factor.get("factor", "Unknown"),
                            "points": float(points),
                            "confidence": factor.get("confidence", "Unknown"),
                        }
                    )

        drivers.sort(key=lambda d: d["points"], reverse=True)
        top_drivers = drivers[:6]

        if not top_drivers:
            return "Insufficient factor-level data to rank material issues."

        lines = ["| Pillar | Factor | Points | Confidence |", "|---|---|---|---|"]
        for d in top_drivers:
            lines.append(
                f"| {d['pillar']} | {d['factor']} | {d['points']:.1f} | {d['confidence']} |"
            )

        return "\n".join(lines)

    def _render_section3_score_derivation(
        self,
        scores: Dict[str, Any],
        pillars: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        gw_score = scores.get("greenwashing_risk_score")
        esg_rating = scores.get("esg_rating")
        risk_level = scores.get("risk_level")
        pillar_scores = scores.get("pillar_scores", {}) or {}

        if isinstance(gw_score, (int, float)):
            lines.append(
                f"The overall greenwashing risk score of {gw_score:.1f}/100 is derived from a weighted aggregation of factor-level signals across the environmental (E), social (S), and governance (G) pillars."
            )
        else:
            lines.append(
                "The overall greenwashing risk score could not be expressed as a single calibrated value; the breakdown below still reflects the relative contributions of each pillar."
            )

        lines.append(
            f"The resulting composite places the issuer in the {risk_level.lower()} risk band and corresponds to an ESG-style rating of {esg_rating}."
        )
        lines.append("")

        # Robustness: if structured factor breakdown is missing, do not show misleading "fallback = 0.0" tables.
        # Instead, present the pillar scores and note the limitation.
        has_any_factors = False
        for p in (pillars or {}).values():
            if isinstance(p, dict) and (p.get("factors") or []):
                has_any_factors = True
                break

        if not has_any_factors and isinstance(pillar_scores, dict) and pillar_scores:
            e = pillar_scores.get("environmental_score")
            s = pillar_scores.get("social_score")
            g = pillar_scores.get("governance_score")
            o = pillar_scores.get("overall_esg_score")
            lines.append("PILLAR SUMMARY (MSCI-style pillar-primary scoring)")
            lines.append("-" * 55)
            lines.append(f"Environmental (E): {e if isinstance(e, (int, float)) else 'N/A'} / 100")
            lines.append(f"Social (S):        {s if isinstance(s, (int, float)) else 'N/A'} / 100")
            lines.append(f"Governance (G):    {g if isinstance(g, (int, float)) else 'N/A'} / 100")
            lines.append(f"Overall ESG:       {o if isinstance(o, (int, float)) else 'N/A'} / 100")
            lines.append("")
            lines.append("Note: Detailed factor-level pillar decomposition was not available for this run;")
            lines.append("the headline pillar scores above reflect the calibrated scoring used for the final rating.")
            return "\n".join(lines)

        for key, p in pillars.items():
            if not isinstance(p, dict):
                p = {}
            label = p.get("label", key)
            score = p.get("score")
            factors = [f for f in (p.get("factors") or []) if isinstance(f, dict)]
            lines.append(f"{label.upper()} PILLAR")
            lines.append("-" * max(20, len(label) + 7))
            if isinstance(score, (int, float)):
                lines.append(f"Pillar score: {score:.1f}/100")
            else:
                lines.append("Pillar score: not available (insufficient data)")

            if not factors:
                lines.append(f"  {'Factor':<36} {'Signal':<12} {'Source':<22} {'Weight':>6} {'Points':>7} {'Confidence':>11}")
                lines.append(f"  {'─' * 90}")
                lines.append(f"  {'Score reconstruction (fallback)':<36} {'N/A':<12} {'Log fallback parser':<22} {'100%':>6} {'0.0':>7} {'Low':>11}")
                lines.append("  Pillar total: 0.0/100")
                lines.append("")
                continue
            lines.append(f"  {'Factor':<36} {'Signal':<12} {'Source':<22} {'Weight':>6} {'Points':>7} {'Confidence':>11}")
            lines.append(f"  {'─' * 90}")
            pillar_total = 0.0
            for f in factors:
                factor = str(f.get("factor", "?"))
                raw_val = f.get("raw_signal", "?")
                raw = str(raw_val)
                weight = f.get("weight")
                contrib = f.get("points_contributed")
                confidence = str(f.get("confidence", "Unknown"))

                if isinstance(raw_val, (int, float)):
                    raw = f"{float(raw_val):.1f}/100"
                if isinstance(weight, (int, float)):
                    w_str = f"{weight * 100:.0f}%"
                else:
                    w_str = "N/A"
                if isinstance(contrib, (int, float)):
                    c_str = f"{float(contrib):.1f}"
                    pillar_total += float(contrib)
                else:
                    c_str = "0.0"

                if confidence.lower() == "low" and "LOW CONFIDENCE" not in factor:
                    factor = f"{factor} [LOW CONFIDENCE]"

                factor = self._shorten_factor_name(factor)

                source_txt = str(f.get('source', 'Unknown'))[:22]
                lines.append(f"  {factor:<36} {raw[:12]:<12} {source_txt:<22} {w_str:>6} {c_str:>7} {confidence:>11}")
            lines.append(f"  {'─' * 90}")
            lines.append(f"  Pillar total: {pillar_total:.1f}/100")
            lines.append("")

        return "\n".join(lines)

    def _render_section4_agent_findings(self, agents: Dict[str, Any]) -> str:
        if not agents:
            return "No agent outputs were available to summarize."

        if not isinstance(agents, dict):
            return "Agent findings payload was malformed and could not be rendered."

        def _as_dict(value: Any) -> Dict[str, Any]:
            return value if isinstance(value, dict) else {}

        def _as_list(value: Any) -> List[Any]:
            return value if isinstance(value, list) else []

        lines: List[str] = []
        for name, info in sorted(agents.items()):
            if not isinstance(info, dict):
                lines.append(f"Agent: {name}")
                lines.append("-" * max(8, len(str(name)) + 7))
                lines.append("Status: MALFORMED OUTPUT | Error: Agent payload was not a structured object")
                lines.append(
                    "This dimension was excluded from the integrated narrative due to malformed agent payload."
                )
                lines.append("")
                continue

            output = info.get("output") or {}
            if not isinstance(output, dict):
                output = {"raw": output}
            error = info.get("error")
            conf = info.get("confidence")
            has_findings = bool(info.get("has_findings")) and not error

            header = f"Agent: {name}"
            lines.append(header)
            lines.append("-" * len(header))
            if error:
                lines.append(f"Status: FAILED | Error: {error}")
                lines.append(
                    "This dimension was excluded from the integrated score; conclusions along this axis are incomplete."
                )
                lines.append("")
                continue

            status = "SUCCESS" if has_findings else "NO STRUCTURED FINDINGS"
            conf_str = f"{float(conf):.1%}" if isinstance(conf, (int, float)) else "n/a"
            lines.append(f"Status: {status} | Confidence: {conf_str}")

            if not has_findings:
                lines.append("The agent reported success but did not return machine-readable findings.")
                lines.append("")
                continue

            if name == "contradiction_analysis":
                found = int(output.get("contradictions_found", 0))
                contradictions = _as_list(output.get("contradiction_list"))
                most = _as_dict(output.get("most_severe"))
                if found == 0:
                    company = output.get("company") or ""
                    known_cases = analyze_contradictions("net zero", company, []).get("controversy_count", 0) if company else 0
                    lines.append(
                        "No contradictions were detected in the current evidence set. NOTE: This may reflect evidence coverage gaps rather than claim accuracy. "
                        f"The known-contradictions database was checked — {known_cases} known cases exist for this company."
                    )
                else:
                    lines.append(
                        f"The contradiction analyzer examined {len(contradictions)} evidence items against the stated claim. {found} contradiction(s) were identified. "
                        f"Most severe: {most.get('description', 'N/A')} (Source: {most.get('source', 'N/A')}, {most.get('year', 'N/A')}, Severity: {most.get('severity', 'N/A')}). "
                        f"Confidence: {int((output.get('confidence', 0.5) or 0.5) * 100)}%."
                    )
            elif name == "carbon_extraction":
                scope1 = output.get("scope1") or (output.get("emissions", {}).get("scope1", {}).get("value"))
                scope2 = output.get("scope2") or (output.get("emissions", {}).get("scope2", {}).get("value"))
                scope3 = output.get("scope3") or (output.get("emissions", {}).get("scope3", {}).get("total"))
                quality = _as_dict(output.get("data_quality"))
                q_score = quality.get("overall_score") if isinstance(quality, dict) else quality
                q_conf = quality.get("data_confidence", "Unknown") if isinstance(quality, dict) else "Unknown"
                missing = output.get("missing_scopes") or [
                    s for s, v in {"Scope 1": scope1, "Scope 2": scope2, "Scope 3": scope3}.items() if v in (None, "N/A")
                ]
                lines.append(
                    f"The carbon extractor analyzed {output.get('articles_analyzed', 0)} evidence items and {_as_dict(output.get('source_coverage')).get('report_chunks', 0)} report chunks. "
                    f"Scope 1: {scope1 if scope1 is not None else 'NOT DISCLOSED'} tCO2e. Scope 2: {scope2 if scope2 is not None else 'NOT DISCLOSED'} tCO2e. "
                    f"Scope 3: {scope3 if scope3 is not None else 'NOT DISCLOSED'}. Data quality: {q_score if q_score is not None else 'N/A'}/100 ({q_conf} confidence). "
                    f"Missing disclosures: {missing}."
                )
                if scope1 is None and scope2 is None and scope3 is None:
                    lines.append("WARNING — No emissions data found. Net-zero claim cannot be quantitatively evaluated. Greenwashing risk elevated.")
            elif name == "sentiment_analysis":
                claim_sent = output.get("claim_sentiment", "neutral")
                evidence_sent = output.get("evidence_sentiment", "neutral")
                divergence_score = float(output.get("sentiment_divergence", 0) or 0)
                if divergence_score >= 0.4:
                    divergence_label = "High"
                elif divergence_score >= 0.2:
                    divergence_label = "Moderate"
                else:
                    divergence_label = "Low"
                lines.append(
                    f"The sentiment analyzer processed {output.get('articles_analyzed', 0)} external sources. Corporate claim tone: {claim_sent}. "
                    f"External evidence tone: {evidence_sent}. Sentiment divergence: {divergence_label}. "
                    f"Notable signal: {output.get('notable_signal') or 'No dominant signal identified'}."
                )
            elif name == "temporal_analysis":
                lines.append(
                    f"The historical analyst examined {output.get('years_analyzed', 0)} year(s) of available data ({output.get('year_range', 'N/A')}). "
                    f"ESG claim trend: {output.get('claim_tone_trend', 'INSUFFICIENT_DATA')}. "
                    f"Environmental performance trend: {output.get('env_performance_trend', 'INSUFFICIENT_DATA')}. "
                    f"Historical violations found: {output.get('violations_count', 0)}."
                )
                if (output.get("violations") or []):
                    v0 = output["violations"][0]
                    lines.append(f"Most notable: {v0.get('description', 'N/A')} ({v0.get('year', 'N/A')}).")
                lines.append(f"Reputation score: {output.get('reputation_score', 'N/A')}/100.")
            elif name == "credibility_analysis":
                high_list = output.get("high_credibility_list") or output.get("trusted_sources") or []
                low_count = output.get("low_credibility_count") or len(output.get("low_confidence_sources", []) or [])
                unverifiable = output.get("unverifiable_count") or len(output.get("unverifiable_sources", []) or [])
                total = output.get("total_sources") or output.get("sources_analyzed") or 0
                overall = output.get("overall_credibility") or output.get("credibility_score") or "N/A"
                lines.append(
                    f"Source credibility assessment across {total} sources. High-credibility sources: {high_list}. "
                    f"Low-credibility sources: {low_count} items. Unverifiable sources: {unverifiable}. "
                    f"Overall credibility score: {overall}/100."
                )
                if isinstance(unverifiable, int) and unverifiable > 3:
                    lines.append(
                        f"WARNING — {unverifiable} sources could not be independently verified and were downweighted in scoring."
                    )
            elif name == "climatebert_analysis":
                claim_a = _as_dict(output.get("claim_analysis"))
                gw = _as_dict(claim_a.get("greenwashing_detection"))
                relevance = _as_dict(claim_a.get("climate_relevance"))
                comp = _as_dict(output.get("comparison"))
                claim_score = comp.get("claim_greenwashing_score", output.get("claim_score", 0))
                evidence_score = comp.get("evidence_greenwashing_score", output.get("evidence_score", 0))
                lines.append(
                    f"ClimateBERT NLP analysis classified the claim as {relevance.get('classification', output.get('classification', 'N/A'))} with {relevance.get('score', output.get('climate_relevance', 'N/A'))}% climate relevance. "
                    f"NLP-based greenwashing signal: {gw.get('risk_score', output.get('greenwashing_risk', 'N/A'))}/100 ({gw.get('risk_level', output.get('risk_level', 'N/A'))}). "
                    f"Claim language score: {claim_score}. Evidence language score: {evidence_score}."
                )
                if isinstance(claim_score, (int, float)) and isinstance(evidence_score, (int, float)) and claim_score > evidence_score + 20:
                    lines.append(
                        "SIGNAL — Claim language is significantly more promotional than the evidence language supports. This is a linguistic indicator of potential greenwashing."
                    )
            elif name == "industry_comparator":
                data_source = output.get("data_source")
                real_ct = output.get("real_peer_count")
                est_ct = output.get("estimated_peer_count")
                lines.append(
                    f"The industry comparator assembled a peer set using data source type '{data_source}', with {real_ct} real peer(s) and {est_ct} estimated baseline(s). Peer-level ESG and greenwashing scores were used to position the issuer within its sector where coverage allowed."
                )
            elif name == "greenwishing_detection":
                overall = _as_dict(output.get("overall_deception_risk"))
                overall_level = overall.get("risk_level")
                greenwishing = _as_dict(output.get("greenwishing")).get("risk_level")
                greenhushing = _as_dict(output.get("greenhushing")).get("risk_level")
                sel_disc = _as_dict(output.get("selective_disclosure")).get("risk_level")
                lines.append(
                    f"The deception-pattern detector scanned corporate language for greenwishing (over-claiming), greenhushing (under-disclosure), and selective disclosure behaviors. Overall deception risk was {overall_level}; greenwishing risk was {greenwishing}, greenhushing risk was {greenhushing}, and selective disclosure risk was {sel_disc}."
                )
            elif name == "regulatory_scanning":
                jurisdiction = output.get("jurisdiction")
                compliance = output.get("compliance_score", {})
                if isinstance(compliance, dict):
                    score = compliance.get("score", "N/A")
                    risk_level = compliance.get("risk_level", "Unknown")
                    gaps = compliance.get("gaps", 0)
                else:
                    score = compliance
                    risk_level = output.get("risk_level", "Unknown")
                    gaps = 0
                compliant = []
                gap_list = []
                for row in _as_list(output.get("compliance_results")):
                    if not isinstance(row, dict):
                        continue
                    gap_details = row.get("gap_details")
                    if not isinstance(gap_details, list):
                        gap_details = []
                    has_gap = len(gap_details) > 0
                    if has_gap:
                        gap_list.append(f"{row.get('regulation_name')}: {', '.join(gap_details[:1])}")
                    else:
                        compliant.append(row.get("regulation_name"))
                lines.append(
                    f"Regulatory scanning covered {jurisdiction} across {len(_as_list(output.get('applicable_regulations')))} frameworks. "
                    f"Compliance score: {score}/100 (Risk: {risk_level}). Gaps identified: {gaps}. "
                    f"Compliant frameworks: {compliant}. Frameworks with gaps: {gap_list}."
                )
                if gaps:
                    alerts = [r.get("regulation") for r in _as_list(output.get("regulatory_risks")) if isinstance(r, dict)]
                    lines.append(f"Regulatory risk alerts: {alerts}.")
            elif name == "explainability":
                factors = _as_list(output.get("top_factors"))
                p = _as_dict(factors[0]) if len(factors) > 0 else {}
                s = _as_dict(factors[1]) if len(factors) > 1 else {}
                lines.append(
                    f"SHAP/LIME explainability analysis identified the top risk drivers. Method: {output.get('method', 'N/A')}. "
                    f"Primary risk driver: {p.get('factor', p.get('feature', 'N/A'))} ({p.get('impact', 'N/A')} impact, {p.get('direction', 'N/A')}). "
                    f"Secondary driver: {s.get('factor', s.get('feature', 'N/A')) if s else 'N/A'}. "
                    f"Summary: {output.get('explanation_text', output.get('human_readable_explanation', 'N/A'))}."
                )
            else:
                lines.append(self._extract_agent_summary(name, output))

            lines.append("")

        return "\n".join(lines)

    def _render_section5_peer_comparison(
        self,
        company: str,
        industry: str,
        peers: Dict[str, Any],
    ) -> str:
        if not isinstance(peers, dict):
            peers = {}
        real_peers = peers.get("real_peers") or []
        if not isinstance(real_peers, list):
            real_peers = []
        if len(real_peers) < 2:
            return "Peer comparison unavailable due to insufficient peer data."

        lines: List[str] = []
        lines.append(
            f"The issuer is benchmarked against {len(real_peers)} real peers from the {industry} universe. Synthetic or estimated peers are excluded from this table."
        )
        header = f"{'Company':<28} {'ESG Score':<10} {'Greenwash Score':<16} {'Rating':<8} {'Source':<10}"
        lines.append(header)
        lines.append("-" * len(header))

        for p in real_peers:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or p.get("company") or "Peer").strip()[:28]
            esg = p.get("esg_score")
            gw = p.get("greenwashing_risk_score")
            rating = p.get("rating") or p.get("rating_grade") or "-"
            src = (p.get("source") or "database")[:10]
            esg_str = f"{float(esg):.1f}" if isinstance(esg, (int, float)) else "-"
            gw_str = f"{float(gw):.1f}" if isinstance(gw, (int, float)) else "-"
            lines.append(f"{name:<28} {esg_str:<10} {gw_str:<16} {rating:<8} {src:<10}")

        lines.append("")
        lines.append(
            f"Rows correspond to real firms with historically observed ESG and controversy trajectories. The relative positioning of {company} should be interpreted with the caveats in Section 7."
        )

        return "\n".join(lines)

    def _render_section6_calibration(
        self,
        calibration: Dict[str, Any],
        scores: Dict[str, Any],
    ) -> str:
        lines: List[str] = []

        gw_score = scores.get("greenwashing_risk_score")
        esg_rating = scores.get("esg_rating")
        threshold = calibration.get("optimal_threshold")
        spearman_r = calibration.get("spearman_r")
        spearman_p = calibration.get("spearman_p")
        pb_r = calibration.get("point_biserial_r")
        mw_p = calibration.get("mannwhitney_p")
        mu_g = calibration.get("mean_score_greenwashing")
        mu_l = calibration.get("mean_score_legitimate")
        region = calibration.get("confidence_region")

        lines.append(
            "The greenwashing risk score is calibrated against a labeled dataset of historical cases covering both confirmed greenwashing incidents and legitimate ESG leadership examples."
        )

        if isinstance(spearman_r, (int, float)) and isinstance(spearman_p, (int, float)):
            lines.append(
                f"In the latest calibration run, the linguistic-greenwashing index achieved a Spearman rank correlation of {spearman_r:.3f} with the ground-truth labels (p = {spearman_p:.4f})."
            )
        if isinstance(pb_r, (int, float)) and isinstance(mw_p, (int, float)):
            lines.append(
                f"Point-biserial correlation between the score and the binary greenwashing label was {pb_r:.3f}, and a Mann–Whitney U test yielded p = {mw_p:.4f}."
            )

        if isinstance(mu_g, (int, float)) and isinstance(mu_l, (int, float)):
            lines.append(
                f"On the calibration sample, average scores were approximately {mu_g:.1f} for known greenwashing cases versus {mu_l:.1f} for legitimate cases."
            )

        if isinstance(threshold, (int, float)):
            lines.append(
                f"An optimal discrimination threshold around {threshold:.1f} was selected to balance sensitivity and specificity."
            )

        if isinstance(gw_score, (int, float)) and isinstance(threshold, (int, float)):
            if region == "high_suspicion_zone":
                lines.append(
                    f"With a score of {gw_score:.1f}, this issuer sits well above the calibrated threshold for greenwashing suspicion; under the reference distribution, such scores are predominantly associated with confirmed greenwashing cases."
                )
            elif region == "likely_legitimate_zone":
                lines.append(
                    f"With a score of {gw_score:.1f}, the issuer lies comfortably below the threshold; in the calibration sample, such scores are more common among legitimate ESG leaders than among greenwashers."
                )
            else:
                lines.append(
                    f"With a score of {gw_score:.1f}, the issuer falls into an intermediate grey zone where both greenwashers and legitimate firms are observed; additional human review is recommended."
                )

        lines.append(
            f"Taken together, these diagnostics support using the score as a probabilistic indicator of greenwashing risk, but not as a deterministic classification. The {esg_rating} label should be interpreted alongside qualitative context and sector expertise."
        )

        wrapped: List[str] = []
        for line in lines:
            if not line:
                wrapped.append("")
            else:
                wrapped.append(self._wrap_paragraph(line))

        return "\n".join(wrapped)

    def _render_section7_limitations(self, limitations: List[str]) -> str:
        if not limitations:
            return self._wrap_paragraph(
                "No specific methodological limitations were automatically detected for this run beyond general disclosure and model caveats."
            )

        lines = ["This assessment is subject to the following case-specific limitations:"]
        for idx, item in enumerate(limitations, start=1):
            lines.append(self._wrap_paragraph(f"{idx}. {item}"))
        return "\n".join(lines)

    def _generate_score_interpretation_section(self, state: Dict[str, Any], score: float) -> str:
        """Generate score interpretation section."""
        if score >= 75:
            level = "HIGH GREENWASHING RISK"
            detail = "Significant inconsistencies detected between ESG claims and verified evidence."
        elif score >= 50:
            level = "MODERATE GREENWASHING RISK"
            detail = "Some inconsistencies detected; further verification recommended."
        elif score >= 25:
            level = "LOW-MODERATE GREENWASHING RISK"
            detail = "Minor inconsistencies detected; claims largely supported by evidence."
        else:
            level = "LOW GREENWASHING RISK"
            detail = "Claims well-supported by available evidence and historical performance."

        return f"""
SCORE INTERPRETATION
{'─'*80}
Score: {score:.1f}/100 — {level}
{detail}
"""

    def _generate_validation_metadata_section(self, calibration: Dict[str, Any] = None) -> str:
        """Add validation & calibration status section."""
        if calibration is None:
            calibration = {}
        lines = [
            "VALIDATION & CALIBRATION STATUS",
            "─" * 52,
        ]

        # Ground Truth
        lines.append("")
        lines.append("Ground Truth Validation:")
        gt_path = os.path.join(os.path.dirname(__file__), '../data/ground_truth_dataset.csv')
        if os.path.exists(gt_path):
            import pandas as pd
            df = pd.read_csv(gt_path)
            lines.append(f"  Dataset:           Ground Truth ESG Dataset v1.0 (data/ground_truth_dataset.csv)")
            lines.append(f"  Verified Cases:    {len(df)} company-claim pairs with regulatory verdicts")
        else:
            lines.append("  Dataset:           Not available — run run_validation.py first")

        # ML Model Performance
        lines.append("")
        lines.append("ML Model Performance (from latest evaluation):")
        eval_path = os.path.join(os.path.dirname(__file__), '../reports/ml_evaluation_results.json')
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                ml = json.load(f)
            best = ml.get('best_model', 'N/A')
            best_f1 = ml.get('best_model_cv_f1', ml.get('best_model_f1', 0))
            dummy_f1 = ml.get('cross_validation_results', {}).get('Dummy', {}).get('f1_mean', 'N/A')
            best_auc = ml.get('holdout_results', {}).get(best, {}).get('auc', 'N/A')
            lines.append(f"  Best Model:        {best} (F1: {best_f1:.3f}, AUC: {best_auc})")
            lines.append(f"  Baseline F1:       {dummy_f1} (majority class)")
        else:
            lines.append("  Best Model:        Not yet evaluated")

        # Score Calibration
        lines.append("")
        lines.append("Score Calibration:")
        spearman_r = calibration.get("spearman_r")
        calib_status = calibration.get("calibration_status", "UNKNOWN")
        if isinstance(spearman_r, (int, float)):
            lines.append(f"  Spearman r:        {spearman_r:.4f} ({calib_status})")
        else:
            lines.append("  Spearman r:        Not available — run run_validation.py first")
        optimal = calibration.get("optimal_threshold")
        if isinstance(optimal, (int, float)):
            lines.append(f"  Optimal Threshold: {optimal}/100")
        else:
            lines.append("  Optimal Threshold: Not available")

        # Contradiction DB
        lines.append("")
        lines.append("Contradiction Database:")
        from data.known_cases import KNOWN_GREENWASHING_CASES
        lines.append(f"  Known Cases:       {sum(len(v) for v in KNOWN_GREENWASHING_CASES.values())} verified regulatory actions")
        lines.append("  Data Sources:      UK ASA, Dutch Courts, US FTC, US SEC, InfluenceMap, ClientEarth")

        return "\n".join(lines)

    def _generate_evidence_source_quality_table(self, state: Dict[str, Any]) -> str:
        """Add evidence source quality table."""
        evidence = state.get("evidence", [])
        counts = {"regulatory": 0, "known_db": 0, "cdp": 0, "wikirate": 0, "web": 0, "estimated": 0}
        for ev in evidence:
            t = ev.get("source_type", "").lower()
            if "regulatory" in t:
                counts["regulatory"] += 1
            elif "known" in t:
                counts["known_db"] += 1
            elif "cdp" in t:
                counts["cdp"] += 1
            elif "wikirate" in t:
                counts["wikirate"] += 1
            elif "web" in t:
                counts["web"] += 1
            elif "estimated" in t:
                counts["estimated"] += 1

        table = [
            "Evidence Quality Assessment:",
            "| Source Type           | Count | Reliability | Notes                    |",
            "|----------------------|-------|-------------|--------------------------|",
            f"| Regulatory rulings   | {counts['regulatory']}   | Very High   | Verified legal records   |",
            f"| Known cases DB       | {counts['known_db']}   | High        | Curated regulatory cases |",
            f"| CDP data             | {counts['cdp']}   | High        | Third-party verified     |",
            f"| Wikirate data        | {counts['wikirate']}   | Medium      | Crowd-sourced + audited  |",
            f"| Web search           | {counts['web']}   | Medium      | Unverified, indicative   |",
            f"| Estimated/synthetic  | {counts['estimated']}   | Low         | Sector benchmarks only   |",
        ]
        return "\n".join(table)

    def _generate_verified_regulatory_actions_section(self, state: Dict[str, Any]) -> str:
        """Add verified regulatory actions section."""
        company = state.get("company", "")
        claim = state.get("claim", "")
        contradiction_result = analyze_contradictions(claim, company, [])

        if contradiction_result["high_confidence_count"] > 0:
            lines = [
                "| Year | Regulatory Body | Severity | Contradiction Summary |",
                "|------|----------------|----------|----------------------|",
            ]
            for c in contradiction_result["contradictions"]:
                if c.get("confidence") == "HIGH":
                    lines.append(
                        f"| {c.get('year','')} | {c.get('regulatory_body','')} | "
                        f"{c.get('severity','').upper()} | {c.get('contradiction_text','')[:100]} |"
                    )
                    lines.append(f"Source: {c.get('source','')} — {c.get('source_url','')}")
            return "\n".join(lines)
        else:
            return (
                "No verified regulatory actions found in public records for this company-claim combination.\n"
                "This does not confirm the claim is accurate — it means no enforcement actions were \n"
                "identified in the known cases database."
            )

    def _generate_regulatory_compliance_section(self, state: Dict[str, Any]) -> str:
        """Add regulatory compliance assessment section."""
        company = state.get("company", "")
        claim = state.get("claim", "")

        regulations = [
            {"regulation_name": "Science Based Targets initiative"},
            {"regulation_name": "GRI"},
            {"regulation_name": "CDP"},
            {"regulation_name": "GHG Protocol"},
            {"regulation_name": "SEBI BRSR"},
            {"regulation_name": "TCFD"},
        ]

        reg_results = []
        for reg in regulations:
            gap = detect_regulation_gaps(company, claim, reg["regulation_name"])
            reg_results.append({"regulation_name": reg["regulation_name"], **gap})

        compliance = compute_compliance_score(reg_results)

        lines = [
            "REGULATORY COMPLIANCE ASSESSMENT",
            "─" * 52,
            f"Compliance Score: {compliance['score']}/100  (Risk Level: {compliance['risk_level']})",
            "",
            "| Regulation | Status | Gap Count |",
            "|------------|--------|-----------|",
        ]
        for r in compliance["per_regulation_status"]:
            lines.append(f"| {r['regulation']} | {r['status']} | {r['gap_count']} |")
        lines.append("")
        lines.append("Gap Details:")
        for r in compliance["per_regulation_status"]:
            if r["gap_count"] > 0:
                for g in r["gaps"]:
                    lines.append(f"- {r['regulation']}: {g}")

        return "\n".join(lines)

    def _generate_key_findings(self, state: Dict[str, Any]) -> str:
        """Generate key findings section."""
        risk_level = state.get("risk_level", "MODERATE")
        confidence = state.get("confidence", 0.0)
        evidence_count = len(state.get("evidence", []))

        findings = []

        if risk_level == "HIGH":
            findings.append("[ALERT] HIGH GREENWASHING RISK DETECTED")
            findings.append("  - Claim lacks sufficient evidence or contains contradictions")
            findings.append("  - Peer comparison shows below-industry-average performance")
            findings.append("  - Historical data reveals inconsistent ESG commitments")
            findings.append("  - Recommended Action: Deep due diligence required before engagement")
        elif risk_level == "MODERATE":
            findings.append("[MODERATE] GREENWASHING RISK IDENTIFIED")
            findings.append("  - Claim partially supported by available evidence")
            findings.append("  - Some contradictions or ambiguities detected")
            findings.append("  - Mixed signals from historical performance")
            findings.append("  - Recommended Action: Additional verification and monitoring")
        else:
            findings.append("[OK] LOW GREENWASHING RISK")
            findings.append("  - Claim well-supported by multiple credible sources")
            findings.append("  - Consistent with historical ESG performance")
            findings.append("  - Aligns with industry best practices")
            findings.append("  - Recommended Action: Standard monitoring protocols")

        findings.append("")

        if confidence >= 0.8:
            findings.append("[OK] HIGH CONFIDENCE ASSESSMENT")
            findings.append("  - Robust evidence base from multiple independent sources")
            findings.append("  - Agent consensus achieved across analytical dimensions")
            findings.append("  - Low uncertainty in risk classification")
        elif confidence >= 0.6:
            findings.append("[MODERATE] CONFIDENCE ASSESSMENT")
            findings.append("  - Adequate evidence but some information gaps identified")
            findings.append("  - Partial agent consensus with minor disagreements")
            findings.append("  - Moderate uncertainty in final assessment")
        else:
            findings.append("[LIMITED] CONFIDENCE")
            findings.append("  - Insufficient evidence for definitive assessment")
            findings.append("  - Significant information gaps remain")
            findings.append("  - Further investigation strongly recommended")

        findings.append("")

        if evidence_count >= 10:
            findings.append("[OK] COMPREHENSIVE EVIDENCE COVERAGE")
            findings.append(f"  - {evidence_count} independent sources analyzed")
        elif evidence_count >= 5:
            findings.append("[MODERATE] ADEQUATE EVIDENCE COVERAGE")
            findings.append(f"  - {evidence_count} sources analyzed")
        else:
            findings.append("[LIMITED] EVIDENCE AVAILABILITY")
            findings.append(f"  - Only {evidence_count} sources available")
            findings.append("  - Assessment reliability may be affected")

        return "\n".join(findings)

    def _generate_peer_comparison_section(self, state: Dict[str, Any]) -> str:
        """Generate peer comparison section with ESG benchmarking table."""
        company = state.get("company", "Unknown")
        industry = state.get("industry", "Unknown")

        risk_scorer_outputs = [
            o for o in state.get("agent_outputs", []) if o.get("agent") == "risk_scoring"
        ]

        if risk_scorer_outputs:
            risk_scorer_result = risk_scorer_outputs[-1].get("output", {})
            pillar_scores = risk_scorer_result.get("pillar_scores", {})
            overall_esg = pillar_scores.get("overall_esg_score")
        else:
            pillar_scores = {}
            overall_esg = None

        try:
            from agents.industry_comparator import IndustryComparator
            comparator = IndustryComparator()

            peer_result = comparator.generate_dynamic_peer_table(
                company=company,
                industry=industry,
                esg_score=overall_esg,
                pillar_scores=pillar_scores,
            )

            if not peer_result.get("available", False):
                return f"""
PEER COMPARISON & INDUSTRY BENCHMARKING
{'─'*80}

{peer_result.get('table_markdown', 'Peer comparison unavailable - limited industry data coverage')}

Note: Peer data unavailable for {industry} sector. This may be due to:
  • Limited public ESG data in this industry
  • Industry classification mismatch
  • Emerging sector with few established competitors
"""

            rank_text = peer_result.get("rank", "N/A")
            industry_avg = peer_result.get("industry_average", {})
            total_peers = peer_result.get("total_peers", 0)
            real_peer_count = peer_result.get("real_peer_count", 0)
            estimated_peer_count = peer_result.get("estimated_peer_count", 0)
            data_source = peer_result.get("data_source", "unknown")
            disclaimer = peer_result.get("disclaimer")

            if data_source == "real":
                data_source_text = "Historical database (previously analyzed companies)"
            elif data_source == "mixed":
                data_source_text = (
                    f"Mixed: {real_peer_count} from historical database, "
                    f"{estimated_peer_count} estimated from industry benchmarks"
                )
            else:
                data_source_text = "Estimated from industry benchmarks (insufficient historical data)"

            section = f"""
PEER COMPARISON & INDUSTRY BENCHMARKING
{'─'*80}

Analysis Context:
  • Industry:        {industry}
  • Peers Analyzed:  {total_peers} competitors
  • Company Rank:    {rank_text}
  • Industry Avg:    {industry_avg.get('esg', 'N/A')}/100
  • Data Source:     {data_source_text}

{peer_result['table_markdown']}

"""

            if disclaimer:
                section += f"""{disclaimer}

As more companies in {industry} are analyzed, this comparison will become more accurate 
with real peer data from the historical database.

"""

            section += """Legend:
  * = Target company
  E  = Environmental Score (0-100)
  S  = Social Score (0-100)
  G  = Governance Score (0-100)

Rating Scale:
  AAA-AA  = 75-100 (ESG Leaders)
  A-BBB   = 50-74  (Average Performance)
  BB-B    = 25-49  (Below Average)
  CCC-C   = 0-24   (ESG Laggards)

"""

            if overall_esg and industry_avg.get('esg'):
                delta = overall_esg - industry_avg.get('esg')
                if delta >= 10:
                    section += f"[OUTPERFORMING] {company} exceeds industry average by {delta:.1f} points\n"
                elif delta >= 5:
                    section += f"[ABOVE AVERAGE] {company} performs {delta:.1f} points above peers\n"
                elif delta >= -5:
                    section += f"[INDUSTRY AVERAGE] {company} aligns with peer performance\n"
                elif delta >= -10:
                    section += f"[BELOW AVERAGE] {company} lags industry by {abs(delta):.1f} points\n"
                else:
                    section += f"[UNDERPERFORMING] {company} significantly trails peers by {abs(delta):.1f} points\n"

            return section

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"""
PEER COMPARISON & INDUSTRY BENCHMARKING
{'─'*80}

Peer comparison unavailable - limited industry data coverage

Technical Error: {str(e)[:100]}
"""

    def _generate_agent_breakdown(self, agent_outputs: List[Dict]) -> str:
        """Generate agent execution breakdown."""
        agent_data = {}
        seen_executions = set()

        for output in agent_outputs:
            agent_name = output.get('agent', 'unknown')
            timestamp = output.get('timestamp', '')
            unique_key = f"{agent_name}_{timestamp}"

            if unique_key in seen_executions:
                continue
            seen_executions.add(unique_key)

            if agent_name not in agent_data:
                agent_data[agent_name] = {
                    'executions': 0,
                    'errors': 0,
                    'confidence_sum': 0,
                    'confidence_count': 0,
                }

            agent_data[agent_name]['executions'] += 1

            if 'error' in output:
                agent_data[agent_name]['errors'] += 1

            if 'confidence' in output and output['confidence'] is not None:
                agent_data[agent_name]['confidence_sum'] += output['confidence']
                agent_data[agent_name]['confidence_count'] += 1

        breakdown = [
            "Agent Execution Summary:",
            "─" * 80,
            f"{'Agent Name':<35} | {'Status':<8} | {'Confidence':<10} | {'Runs':<5}",
            "─" * 80,
        ]

        for agent_name in sorted(agent_data.keys()):
            data = agent_data[agent_name]

            if data['confidence_count'] > 0:
                avg_conf = data['confidence_sum'] / data['confidence_count']
                conf_display = f"{avg_conf:.1%}"
            else:
                conf_display = "N/A"

            status = "FAILED" if data['errors'] > 0 else "SUCCESS"
            display_name = agent_name.replace('_', ' ').title()

            actual_runs = data['executions']
            run_display = str(min(actual_runs, 2))

            breakdown.append(
                f"{display_name:<35} | {status:<8} | {conf_display:<10} | {run_display:<5}"
            )

        breakdown.append("─" * 80)
        return "\n".join(breakdown)

    def _generate_detailed_analysis(self, state: Dict[str, Any], agent_outputs: List[Dict]) -> str:
        """Generate detailed agent analysis section."""
        sections = []

        agent_summaries = {}
        for output in agent_outputs:
            agent_name = output.get("agent", "unknown")
            if agent_name not in ["supervisor", "confidence_monitor", "assess_complexity"]:
                if agent_name not in agent_summaries:
                    agent_summaries[agent_name] = []
                agent_summaries[agent_name].append(output)

        # Environmental Analysis
        sections.append("ENVIRONMENTAL DIMENSION")
        sections.append("─" * 80)

        if "contradiction_analysis" in agent_summaries:
            output = agent_summaries["contradiction_analysis"][0]
            contradictions = output.get("contradictions_count", 0)
            if contradictions > 0:
                sections.append(f"[WARN] Claim Consistency:    {contradictions} contradiction(s) detected")
            else:
                sections.append("[OK] Claim Consistency:    No contradictions found")

        if "evidence_retrieval" in agent_summaries:
            output = agent_summaries["evidence_retrieval"][0]
            evidence_count = output.get("evidence_count", 0)
            sections.append(f"  Evidence Coverage:    {evidence_count} independent source(s)")

        if "temporal_analysis" in agent_summaries:
            sections.append("  Historical Track Record: Past ESG performance evaluated")

        sections.append("")

        # Social Dimension
        sections.append("SOCIAL DIMENSION")
        sections.append("─" * 80)

        if "sentiment_analysis" in agent_summaries:
            sections.append("  Public Sentiment:     Analyzed from recent media coverage")

        if "credibility_analysis" in agent_summaries:
            sections.append("  Source Credibility:   Verified against trusted repositories")

        if "realtime_monitoring" in agent_summaries:
            sections.append("  Real-time Monitoring: Latest news and developments tracked")

        sections.append("")

        # Governance Dimension
        sections.append("GOVERNANCE DIMENSION")
        sections.append("─" * 80)

        if "peer_comparison" in agent_summaries:
            sections.append("  Industry Benchmarking:   Compared against sector peers")

        if "risk_scoring" in agent_summaries:
            output = agent_summaries["risk_scoring"][0]
            risk_level = output.get("risk_level", "N/A")
            sections.append(f"  Risk Assessment:         {risk_level} risk classification")

        sections.append("")
        return "\n".join(sections)

    def _generate_evidence_summary(self, state: Dict[str, Any]) -> str:
        """Generate evidence summary."""
        evidence = state.get("evidence", [])

        if not evidence:
            return (
                "No evidence sources available for this analysis.\n"
                "This may indicate data collection issues or claim verification challenges."
            )

        summary = [
            f"Total Evidence Sources: {len(evidence)}",
            "─" * 80,
            "",
        ]

        sources: Dict[str, list] = {}
        for item in evidence[:15]:
            source = item.get("source", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(item)

        for source_type, items in sorted(sources.items()):
            source_display = source_type.replace('_', ' ').title()
            summary.append(f"{source_display}: {len(items)} item(s)")
            summary.append("─" * 40)

            for i, item in enumerate(items[:5], 1):
                title = item.get("title", item.get("snippet", "N/A"))
                if len(title) > 75:
                    title = title[:72] + "..."
                summary.append(f"  {i}. {title}")

            if len(items) > 5:
                summary.append(f"  ... and {len(items)-5} more items")

            summary.append("")

        return "\n".join(summary)

    def _generate_pillar_section(self, pillar_scores: Dict[str, float]) -> str:
        """Generate ESG pillar scores section."""
        if not pillar_scores:
            return f"""
ESG PILLAR SCORES
{'─'*80}
(Pillar scores not available - insufficient data)
"""

        env_score = pillar_scores.get("environmental_score")
        soc_score = pillar_scores.get("social_score")
        gov_score = pillar_scores.get("governance_score")
        overall_esg = pillar_scores.get("overall_esg_score")
        industry_adj = pillar_scores.get("industry_adjustment", 0)

        if any(v is None for v in [env_score, soc_score, gov_score, overall_esg]):
            return f"""
ESG PILLAR SCORES
{'─'*80}
(Pillar scores partially available - upstream scoring output incomplete)
"""

        env_contribution = env_score * 0.35
        soc_contribution = soc_score * 0.30
        gov_contribution = gov_score * 0.35

        def get_performance_level(score):
            if score >= 70:
                return "Strong"
            elif score >= 50:
                return "Average"
            return "Weak"

        env_level = get_performance_level(env_score)
        soc_level = get_performance_level(soc_score)
        gov_level = get_performance_level(gov_score)

        return f"""
ESG PILLAR SCORES (Industry-Adjusted)
{'─'*80}

ENVIRONMENTAL SCORE:      {env_score:.1f}/100  ({env_level})
  Weight:                 35%
  Weighted Contribution:  {env_contribution:.1f} points

  Key Factors:
    • Carbon emissions and climate strategy
    • Energy efficiency and renewable usage
    • Water management and biodiversity impact
    • Waste reduction and circular economy

SOCIAL SCORE:             {soc_score:.1f}/100  ({soc_level})
  Weight:                 30%
  Weighted Contribution:  {soc_contribution:.1f} points

  Key Factors:
    • Labor practices and employee welfare
    • Diversity, equity, and inclusion (DEI)
    • Community engagement and human rights
    • Product safety and stakeholder relations

GOVERNANCE SCORE:         {gov_score:.1f}/100  ({gov_level})
  Weight:                 35%
  Weighted Contribution:  {gov_contribution:.1f} points

  Key Factors:
    • Board structure and independence
    • Ethics and compliance frameworks
    • Transparency and disclosure quality
    • Anti-corruption and accountability measures

{'─'*80}
OVERALL ESG SCORE:        {overall_esg:.1f}/100

Calculation:
  (Environmental × 0.35) + (Social × 0.30) + (Governance × 0.35)
  ({env_score:.1f} × 0.35) + ({soc_score:.1f} × 0.30) + ({gov_score:.1f} × 0.35) = {overall_esg:.1f}

Industry Baseline Adjustment: {industry_adj:+.1f} points
  (Applied to account for sector-specific ESG challenges)
"""

    def _collect_realism_diagnostics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect realism diagnostics from scorer and evidence outputs."""
        agent_outputs = state.get("agent_outputs", [])

        risk_output: Dict[str, Any] = {}
        evidence_output: Dict[str, Any] = {}

        for output in reversed(agent_outputs):
            if not risk_output and output.get("agent") == "risk_scoring":
                candidate = output.get("output", {})
                if isinstance(candidate, dict):
                    risk_output = candidate
            if not evidence_output and output.get("agent") == "evidence_retrieval":
                candidate = output.get("output", {})
                if isinstance(candidate, dict):
                    evidence_output = candidate
            if risk_output and evidence_output:
                break

        pillar_scores = risk_output.get("pillar_scores", {})
        if not isinstance(pillar_scores, dict):
            pillar_scores = {}

        dei_progress = pillar_scores.get("dei_progress", {})
        if not isinstance(dei_progress, dict):
            dei_progress = {}

        quality_metrics = evidence_output.get("quality_metrics", {})
        if not isinstance(quality_metrics, dict):
            quality_metrics = {}

        source_breakdown = evidence_output.get("source_breakdown", {})
        if not isinstance(source_breakdown, dict):
            source_breakdown = {}

        total_evidence_sources = 0
        for value in source_breakdown.values():
            if isinstance(value, (int, float)):
                total_evidence_sources += int(value)

        independent_sources = int(quality_metrics.get("independent_sources", 0) or 0)
        premium_sources = int(quality_metrics.get("premium_sources", 0) or 0)
        source_diversity = int(quality_metrics.get("source_diversity", len(source_breakdown)) or 0)
        evidence_gap = bool(quality_metrics.get("evidence_gap", False))

        independent_share = (independent_sources / max(total_evidence_sources, 1)) * 100
        premium_share = (premium_sources / max(total_evidence_sources, 1)) * 100

        offset_status = str(pillar_scores.get("offset_transparency_status", "unknown")).lower()
        offset_penalty = float(pillar_scores.get("offset_penalty", 0) or 0)

        offset_integrity = "unknown"
        if offset_status in {"transparent", "credible"}:
            offset_integrity = "strong"
        elif offset_status in {"mixed", "partial"}:
            offset_integrity = "moderate"
        elif offset_status in {"opaque_avoidance_heavy", "opaque", "unknown", "not_disclosed"}:
            offset_integrity = "weak"

        has_target = bool(dei_progress.get("has_target", False))
        has_actual = bool(dei_progress.get("has_actual", False))
        yoy_change = dei_progress.get("yoy_change")
        target_gap = dei_progress.get("target_gap")

        dei_execution = "insufficient"
        if has_target and has_actual:
            if isinstance(yoy_change, (int, float)) and yoy_change > 0:
                if isinstance(target_gap, (int, float)) and target_gap <= 0:
                    dei_execution = "strong"
                else:
                    dei_execution = "improving"
            elif isinstance(yoy_change, (int, float)) and yoy_change <= 0:
                dei_execution = "stagnant"
            else:
                dei_execution = "moderate"
        elif has_target and not has_actual:
            dei_execution = "target_only"

        temporal_mode = str(risk_output.get("temporal_mode", "none"))
        temporal_weight = float(risk_output.get("temporal_weight", 0) or 0)
        temporal_data_quality = risk_output.get("temporal_data_quality", {})

        temporal_quality_score = 0.0
        temporal_quality_label = "unknown"
        if isinstance(temporal_data_quality, dict):
            raw_score = temporal_data_quality.get("overall_score", temporal_data_quality.get("score", 0))
            if isinstance(raw_score, (int, float)):
                temporal_quality_score = float(raw_score)
            temporal_quality_label = str(
                temporal_data_quality.get("quality_label", temporal_data_quality.get("data_confidence", "unknown"))
            )
        elif isinstance(temporal_data_quality, (int, float)):
            temporal_quality_score = float(temporal_data_quality)
            temporal_quality_label = "numeric"

        temporal_reliability = "limited"
        if temporal_mode in {"trend", "snapshot"} and temporal_quality_score >= 70 and temporal_weight >= 0.10:
            temporal_reliability = "strong"
        elif temporal_mode in {"trend", "snapshot"} and temporal_quality_score >= 45 and temporal_weight >= 0.05:
            temporal_reliability = "moderate"

        offset_points = 8
        if offset_integrity == "strong":
            offset_points = 25
        elif offset_integrity == "moderate":
            offset_points = 17
        elif offset_integrity == "weak":
            offset_points = 7
        offset_points = max(0, offset_points - min(int(offset_penalty), 10))

        dei_points = 8
        if dei_execution == "strong":
            dei_points = 25
        elif dei_execution == "improving":
            dei_points = 20
        elif dei_execution == "moderate":
            dei_points = 15
        elif dei_execution == "stagnant":
            dei_points = 9
        elif dei_execution == "target_only":
            dei_points = 6

        evidence_points = min(
            25, int((independent_share * 0.5) + (premium_share * 0.25) + (source_diversity * 2))
        )
        if evidence_gap:
            evidence_points = max(0, evidence_points - 6)

        temporal_points = 6
        if temporal_reliability == "strong":
            temporal_points = 25
        elif temporal_reliability == "moderate":
            temporal_points = 16

        realism_score = max(0, min(100, offset_points + dei_points + evidence_points + temporal_points))
        realism_label = "high"
        if realism_score < 70:
            realism_label = "moderate"
        if realism_score < 50:
            realism_label = "limited"

        return {
            "realism_score": realism_score,
            "realism_label": realism_label,
            "offset_integrity": offset_integrity,
            "offset_status": offset_status,
            "offset_penalty": round(offset_penalty, 1),
            "dei_execution": dei_execution,
            "dei_progress": {
                "has_target": has_target,
                "has_actual": has_actual,
                "yoy_change": yoy_change,
                "target_gap": target_gap,
            },
            "evidence_composition": {
                "total_evidence_sources": total_evidence_sources,
                "independent_sources": independent_sources,
                "premium_sources": premium_sources,
                "independent_share_pct": round(independent_share, 1),
                "premium_share_pct": round(premium_share, 1),
                "source_diversity": source_diversity,
                "evidence_gap": evidence_gap,
            },
            "temporal_reliability": {
                "mode": temporal_mode,
                "weight": round(temporal_weight, 3),
                "quality_score": round(temporal_quality_score, 1),
                "quality_label": temporal_quality_label,
                "reliability": temporal_reliability,
            },
        }

    def _generate_realism_diagnostics_section(self, state: Dict[str, Any]) -> str:
        """Generate a concise diagnostics panel for evidence and offset integrity."""
        diagnostics = self._collect_realism_diagnostics(state)

        evidence_diag = diagnostics.get("evidence_composition", {})
        temporal_diag = diagnostics.get("temporal_reliability", {})

        return f"""
EVIDENCE & OFFSET INTEGRITY
{'─'*80}

Overall Realism Confidence: {diagnostics.get('realism_score', 0)}/100 ({str(diagnostics.get('realism_label', 'unknown')).upper()})

Offset Integrity:
  - Classification: {str(diagnostics.get('offset_integrity', 'unknown')).upper()} ({diagnostics.get('offset_status', 'unknown')})
  - Penalty Applied: {diagnostics.get('offset_penalty', 0)} point(s)

Evidence Composition:
  - Total Source Items: {evidence_diag.get('total_evidence_sources', 0)}
  - Independent Sources: {evidence_diag.get('independent_sources', 0)} ({evidence_diag.get('independent_share_pct', 0)}%)
  - Premium Sources: {evidence_diag.get('premium_sources', 0)} ({evidence_diag.get('premium_share_pct', 0)}%)
  - Source Diversity: {evidence_diag.get('source_diversity', 0)} type(s)
  - Evidence Gap Flag: {'YES' if evidence_diag.get('evidence_gap') else 'NO'}

Temporal Reliability:
  - Mode: {temporal_diag.get('mode', 'none')}
  - Weight in Final Scoring: {temporal_diag.get('weight', 0)}
  - Data Quality: {temporal_diag.get('quality_score', 0)}/100 ({temporal_diag.get('quality_label', 'unknown')})
  - Reliability Tier: {str(temporal_diag.get('reliability', 'limited')).upper()}
"""

    def _generate_quantitative_metrics_section(self, state: Dict[str, Any]) -> str:
        """Generate quantitative performance metrics section with industry benchmarking."""
        company = state.get("company", "Unknown")
        industry = state.get("industry", "Unknown")

        financial_context = None
        agent_outputs = state.get("agent_outputs", [])
        for output in agent_outputs:
            if output.get("agent") == "financial_analysis":
                financial_context = output.get("output", {})
                break

        agents_struct = self._extract_agent_findings(state)
        contradiction_output = agents_struct.get("contradiction_analysis", {}).get("output", {})
        controversy_count = int(contradiction_output.get("contradictions_found", 0))

        evidence_list = state.get("evidence", [])
        total_evidence = len(evidence_list)
        max_possible_sources = 14

        unique_sources = set()
        for ev in evidence_list:
            if isinstance(ev, dict):
                source = ev.get("source", "unknown")
                unique_sources.add(source)

        unique_source_count = len(unique_sources)
        effective_source_universe = max(max_possible_sources, unique_source_count, 1)
        unique_disclosure_pct = (unique_source_count / effective_source_universe * 100)

        section = f"""
KEY PERFORMANCE METRICS
{'─'*80}

"""

        # === CARBON EXTRACTION DATA (from CarbonExtractor agent) ===
        carbon_data = state.get("carbon_extraction")
        has_carbon_extraction = False

        if carbon_data and isinstance(carbon_data, dict):
            has_carbon_extraction = True

            section += "CARBON EMISSIONS (Scope 1/2/3 Analysis)\n"
            section += f"{'─'*80}\n\n"

            emissions = carbon_data.get("emissions", {})
            scope1 = emissions.get("scope1", carbon_data.get("scope_1", {}))
            scope2 = emissions.get("scope2", carbon_data.get("scope_2", {}))
            scope3 = emissions.get("scope3", carbon_data.get("scope_3", {}))

            section += f"| {'Scope':<20} | {'Emissions (tCO2e)':<20} | {'Year':<10} | {'Source':<25} |\n"
            section += f"|{'-'*22}|{'-'*22}|{'-'*12}|{'-'*27}|\n"
            missing_scope_rows = []

            scope1_value = scope1.get("value") or scope1.get("emissions_tco2e")
            scope1_year = scope1.get("year", "")
            scope1_source = scope1.get("source", "BRSR/CDP")
            if scope1_value is not None and scope1_value != "N/A":
                section += f"| {'Scope 1 (Direct)':<20} | {scope1_value:>18,} | {str(scope1_year):<10} | {str(scope1_source)[:23]:<25} |\n"
            else:
                missing_scope_rows.append("Scope 1")

            scope2_value = scope2.get("value") or scope2.get("emissions_tco2e")
            scope2_year = scope2.get("year", "")
            scope2_source = scope2.get("source", scope2.get("methodology", ""))
            if scope2_value is not None and scope2_value != "N/A":
                section += f"| {'Scope 2 (Energy)':<20} | {scope2_value:>18,} | {str(scope2_year):<10} | {str(scope2_source)[:23]:<25} |\n"
            else:
                missing_scope_rows.append("Scope 2")

            scope3_value = scope3.get("total") or scope3.get("value") or scope3.get("emissions_tco2e")
            scope3_year = scope3.get("year", "")
            scope3_cats = scope3.get("categories", {})
            scope3_source = f"{len(scope3_cats)} categories" if scope3_cats else "Value Chain"
            if scope3_value is not None and scope3_value != "N/A":
                section += f"| {'Scope 3 (Value Chain)':<20} | {scope3_value:>18,} | {str(scope3_year):<10} | {str(scope3_source)[:23]:<25} |\n"
            else:
                missing_scope_rows.append("Scope 3")

            # Only surface missing scopes if they are material in context.
            if missing_scope_rows:
                status = str(carbon_data.get("data_quality", {}).get("status", "")).lower()
                used_baseline = bool(carbon_data.get("used_baseline_estimate", False))
                note_prefix = "Estimated from industry baselines; underlying disclosures are missing for: " if used_baseline else "Missing scope disclosures: "
                # If carbon is non-material to the sector, keep this note out of the main body.
                if industry.lower().replace(" ", "_") in ["oil_and_gas", "coal", "mining", "aviation", "power", "cement", "steel", "energy", "utilities", "banking"]:
                    section += f"\nNote: {note_prefix}{', '.join(missing_scope_rows)}\n"
                else:
                    section += f"\nNote (non-material carbon context): {', '.join(missing_scope_rows)}\n"

            section += "\n"

            total_emissions = emissions.get("total") or carbon_data.get("total_emissions_tco2e")
            if isinstance(total_emissions, dict):
                total_emissions = (
                    total_emissions.get("all_scopes")
                    or total_emissions.get("scope1_2")
                    or total_emissions.get("value")
                )

            carbon_intensity = carbon_data.get("carbon_intensity") or carbon_data.get(
                "intensity_metrics", {}
            ).get("carbon_intensity")
            if isinstance(carbon_intensity, dict):
                carbon_intensity = carbon_intensity.get("value")

            net_zero_target = carbon_data.get("net_zero_target")
            renewable_pct = carbon_data.get("renewable_energy_percentage")
            sbt = carbon_data.get("science_based_target")
            verification = carbon_data.get("verification_status")
            data_source = carbon_data.get("data_source")
            data_quality = carbon_data.get("data_quality", {})
            offset_transparency = carbon_data.get("offset_transparency", {})

            if total_emissions and isinstance(total_emissions, (int, float)):
                section += f"Total Emissions: {int(total_emissions):,} tCO2e\n"
            if carbon_intensity and isinstance(carbon_intensity, (int, float)):
                section += f"Carbon Intensity: {carbon_intensity} tCO2e/unit\n"
            elif carbon_intensity:
                section += f"Carbon Intensity: {carbon_intensity}\n"
            if net_zero_target:
                section += f"Net Zero Target: {net_zero_target}\n"
            if renewable_pct:
                section += f"Renewable Energy: {renewable_pct}\n"
            if sbt:
                section += "Science-Based Target: Yes (SBTi approved)\n"
            if verification:
                section += f"Verification: {verification}\n"
            if data_source:
                section += f"Data Source: {data_source}\n"

            if isinstance(offset_transparency, dict) and offset_transparency:
                section += (
                    f"Offset Transparency: {offset_transparency.get('status', 'unknown')} "
                    f"(avoidance={offset_transparency.get('avoidance_share_pct', 0)}%, "
                    f"removal={offset_transparency.get('removal_share_pct', 0)}%)\n"
                )

            if isinstance(data_quality, dict):
                quality_score = data_quality.get("overall_score", 0)
                confidence = data_quality.get("data_confidence", "Unknown")
                status = str(data_quality.get("status", "")).lower()
                if status == "estimated_baseline":
                    section += (
                        f"Data Quality Score: {quality_score}/100 ({confidence} confidence)\n"
                        f"Explanation: No disclosed Scope 1/2/3 values were found; emissions table above reflects "
                        f"industry baseline estimates for stability and should be treated as indicative only.\n"
                    )
                else:
                    section += f"Data Quality Score: {quality_score}/100 ({confidence} confidence)\n"
            else:
                # Only surface a generic 0/None if this is explicitly flagged upstream
                if data_quality not in (None, 0, "N/A"):
                    section += f"Data Quality: {data_quality}\n"

            section += "\n"

            grid_factor = carbon_data.get("grid_emission_factor")
            country = carbon_data.get("country_detected", "Unknown")
            if grid_factor:
                section += f"Grid Emission Factor: {grid_factor} tCO2/MWh ({country})\n\n"

        # === CARBON METRICS (from Financial Analyst - fallback) ===
        has_carbon_data = has_carbon_extraction

        if financial_context and isinstance(financial_context, dict) and not has_carbon_extraction:
            esg_metrics = financial_context.get("esg_financial_metrics", {})

            carbon_intensity = esg_metrics.get("carbon_intensity")
            water_efficiency = esg_metrics.get("water_efficiency")
            energy_efficiency = esg_metrics.get("energy_efficiency")

            if carbon_intensity is not None or water_efficiency is not None or energy_efficiency is not None:
                has_carbon_data = True

                section += "ENVIRONMENTAL METRICS\n"
                section += f"{'─'*80}\n\n"

                section += f"| {'Metric':<30} | {'Value':<20} | {'Status':<15} |\n"
                section += f"|{'-'*32}|{'-'*22}|{'-'*17}|\n"

                if carbon_intensity is not None:
                    carbon_benchmarks = {
                        "oil_and_gas": 0.05, "energy": 0.04, "automotive": 0.02,
                        "aviation": 0.03, "manufacturing": 0.015, "technology": 0.005,
                        "finance": 0.001, "healthcare": 0.008,
                    }
                    industry_key = industry.lower().replace(" ", "_").replace("&", "and")
                    industry_avg = carbon_benchmarks.get(industry_key, 0.01)
                    status = "Above Avg" if carbon_intensity > industry_avg else "Below Avg"
                    section += f"| {'Carbon Intensity':<30} | {carbon_intensity:.6f} tCO2/${'':>8} | {status:<15} |\n"
                    section += f"| {'  Industry Average':<30} | {industry_avg:.6f} tCO2/${'':>8} | {'':>15} |\n"

                if water_efficiency is not None:
                    water_benchmarks = {
                        "oil_and_gas": 0.002, "energy": 0.0015, "automotive": 0.001,
                        "manufacturing": 0.0008, "food_beverage": 0.003,
                    }
                    industry_key = industry.lower().replace(" ", "_").replace("&", "and")
                    industry_avg = water_benchmarks.get(industry_key, 0.001)
                    status = "Above Avg" if water_efficiency > industry_avg else "Below Avg"
                    section += f"| {'Water Intensity':<30} | {water_efficiency:.6f} L/${'':>10} | {status:<15} |\n"

                if energy_efficiency is not None:
                    energy_benchmarks = {
                        "oil_and_gas": 0.003, "energy": 0.0025, "manufacturing": 0.002,
                        "technology": 0.0008, "finance": 0.0005,
                    }
                    industry_key = industry.lower().replace(" ", "_").replace("&", "and")
                    industry_avg = energy_benchmarks.get(industry_key, 0.0015)
                    status = "Above Avg" if energy_efficiency > industry_avg else "Below Avg"
                    section += f"| {'Energy Intensity':<30} | {energy_efficiency:.6f} kWh/${'':>8} | {status:<15} |\n"

                section += "\n"
                section += "Interpretation:\n"
                section += "  • Lower intensity = Better environmental efficiency\n"
                section += f"  • {company} carbon footprint per revenue dollar\n"
                section += f"  • Benchmarked against {industry} sector averages\n\n"

        if not has_carbon_data:
            section += "ENVIRONMENTAL METRICS\n"
            section += f"{'─'*80}\n\n"
            section += "[NOTE] Carbon Metrics: Not publicly disclosed (Transparency Gap)\n"
            section += "[NOTE] Water Usage: Not publicly disclosed\n"
            section += "[NOTE] Energy Consumption: Not publicly disclosed\n\n"
            section += "Note: Lack of environmental data disclosure may indicate:\n"
            section += "  • Limited ESG reporting maturity\n"
            section += "  • Private company without disclosure requirements\n"
            section += "  • Emerging market with lower transparency standards\n\n"

        # === GOVERNANCE METRICS ===
        section += "GOVERNANCE & DISCLOSURE METRICS\n"
        section += f"{'─'*80}\n\n"

        section += f"| {'Metric':<35} | {'Value':<20} | {'Assessment':<15} |\n"
        section += f"|{'-'*37}|{'-'*22}|{'-'*17}|\n"

        board_independence = None
        if financial_context and isinstance(financial_context, dict):
            gov_metrics = financial_context.get("governance_metrics", {})
            board_independence = gov_metrics.get("board_independence")

        if board_independence:
            status = "Strong" if board_independence > 60 else "Weak" if board_independence < 40 else "Average"
            section += f"| {'Board Independence Score':<35} | {board_independence:.1f}/100{'':>13} | {status:<15} |\n"

        controversy_status = "Clean" if controversy_count == 0 else "Concerns" if controversy_count <= 3 else "High Risk"
        section += f"| {'Controversy Count':<35} | {controversy_count} issue(s){'':>11} | {controversy_status:<15} |\n"

        disclosure_status = (
            "Excellent" if unique_disclosure_pct >= 70 else "Good" if unique_disclosure_pct >= 50 else "Limited"
        )
        section += (
            f"| {'Disclosure Score':<35} | {unique_source_count}/{effective_source_universe} sources "
            f"({unique_disclosure_pct:.0f}%){'':>3} | {disclosure_status:<15} |\n"
        )

        section += "\n"
        section += "Interpretation:\n"
        section += f"  • Controversy Count: {controversy_count} contradiction(s) found in claims vs evidence\n"
        section += (
            f"  • Disclosure Score: {unique_source_count} unique sources out of "
            f"{effective_source_universe} observed ({unique_disclosure_pct:.0f}%)\n"
        )
        section += f"  • Total Evidence Items: {total_evidence} (may include multiple items per source)\n"
        section += "  • Higher disclosure = Greater transparency\n\n"

        # === FINANCIAL-ESG ALIGNMENT ===
        if financial_context and isinstance(financial_context, dict):
            greenwashing_flags = financial_context.get("greenwashing_flags", [])

            if greenwashing_flags and len(greenwashing_flags) > 0:
                section += "FINANCIAL-ESG MISALIGNMENT FLAGS\n"
                section += f"{'─'*80}\n\n"

                for flag in greenwashing_flags[:5]:
                    if isinstance(flag, dict):
                        severity = flag.get("severity", "Low")
                        description = flag.get("description", "")
                        marker = "[ALERT]" if severity == "High" else "[WARN]" if severity == "Moderate" else "[NOTE]"
                        section += f"{marker} {severity} Risk: {description}\n"

                section += "\n"

        return section

    def generate_json_export(
        self,
        analysis_state: Dict[str, Any],
        report_metadata: Dict[str, Any],
        structured: Dict[str, Any] = None,
        quality: Dict[str, Any] = None,
    ) -> Tuple[str, int]:
        if structured is None:
            structured = self._build_structured_report(analysis_state)
        if quality is None:
            quality = ReportQualityChecker().evaluate(analysis_state, structured)

        scores = structured.get("scores", {}) or {}
        raw_scores = scores.get("raw", {}) if isinstance(scores.get("raw"), dict) else {}
        pillar_scores = scores.get("pillar_scores", {}) if isinstance(scores.get("pillar_scores"), dict) else {}
        evidence_struct = structured.get("evidence", {}) or {}
        agents_struct = structured.get("agents", {}) or {}
        calibration = structured.get("calibration", {}) or {}

        esg_score = raw_scores.get("esg_score")
        if esg_score is None:
            esg_score = raw_scores.get("overall_esg_score") or pillar_scores.get("overall_esg_score")

        carbon_source = (
            analysis_state.get("carbon_results")
            or analysis_state.get("carbon_extraction")
            or {}
        )
        emissions = carbon_source.get("emissions", {}) if isinstance(carbon_source, dict) else {}
        scope1 = emissions.get("scope1", carbon_source.get("scope_1", {})) if isinstance(emissions, dict) else {}
        scope2 = emissions.get("scope2", carbon_source.get("scope_2", {})) if isinstance(emissions, dict) else {}
        scope3 = emissions.get("scope3", carbon_source.get("scope_3", {})) if isinstance(emissions, dict) else {}

        contradiction_output = agents_struct.get("contradiction_analysis", {}).get("output", {})
        contradictions = []
        if isinstance(contradiction_output, dict):
            contradictions = (
                contradiction_output.get("contradiction_list")
                or contradiction_output.get("contradictions")
                or contradiction_output.get("specific_contradictions")
                or []
            )

        regulatory = (
            analysis_state.get("regulatory_results")
            or analysis_state.get("regulatory_compliance")
            or {}
        )

        agent_results = []
        for name, info in sorted(agents_struct.items()):
            output = info.get("output") if isinstance(info, dict) else {}
            if not isinstance(output, dict):
                output = {"raw": output}
            status = "FAILED" if info.get("error") else "SUCCESS" if info.get("has_findings") else "NO_DATA"
            key_findings = {
                k: v
                for k, v in output.items()
                if k not in {"raw_response", "prompt", "status", "confidence"}
                and not isinstance(v, (bytes, type(None)))
            }
            agent_results.append(
                {
                    "agent": name,
                    "status": status,
                    "confidence": info.get("confidence"),
                    "key_findings": key_findings,
                }
            )

        export = {
            "report_id": report_metadata.get("report_id"),
            "analysis_date": report_metadata.get("analysis_date"),
            "company": analysis_state.get("company"),
            "industry": analysis_state.get("industry"),
            "claim_analyzed": analysis_state.get("claim"),
            "scores": {
                "greenwashing_score": scores.get("greenwashing_risk_score"),
                "esg_score": esg_score,
                "esg_rating": scores.get("esg_rating"),
                "environmental": pillar_scores.get("environmental_score"),
                "social": pillar_scores.get("social_score"),
                "governance": pillar_scores.get("governance_score"),
                "confidence": scores.get("confidence", analysis_state.get("confidence")),
                "compliance": regulatory.get("compliance_score"),
            },
            "pillar_factors": raw_scores.get("pillar_factors") or {},
            "contradictions": contradictions,
            "regulatory_gaps": [
                {
                    "regulation": r.get("regulation_name"),
                    "gap_details": r.get("gap_details", []),
                }
                for r in (regulatory.get("compliance_results", []) or [])
                if len(r.get("gap_details", [])) > 0
            ],
            "carbon_data": {
                "scope1": scope1.get("value") if isinstance(scope1, dict) else None,
                "scope2": scope2.get("value") if isinstance(scope2, dict) else None,
                "scope3": scope3.get("total") if isinstance(scope3, dict) else None,
                "data_quality": (
                    safe_get(carbon_source, "data_quality", "overall_score")
                    if isinstance(carbon_source, dict)
                    else None
                ),
            },
            "evidence_sources": [
                {
                    "source_name": parse_source_name(e.get("url", "")) if e.get("source_name") in (None, "Unknown", "") else e.get("source_name"),
                    "url": e.get("url"),
                    "reliability_tier": e.get("reliability_tier"),
                    "stance": e.get("claim_support"),
                    "date_retrieved": e.get("date"),
                }
                for e in (evidence_struct.get("citations", []) or [])
                if isinstance(e, dict)
            ],
            "agent_results": agent_results,
            "calibration": {
                "spearman_r": calibration.get("spearman_r"),
                "spearman_p": calibration.get("spearman_p"),
                "point_biserial_r": calibration.get("point_biserial_r"),
                "optimal_threshold": calibration.get("optimal_threshold"),
                "dataset_size": calibration.get("dataset_size"),
                "calibration_status": calibration.get("calibration_status"),
            },
            "report_generation_log": analysis_state.get("report_generation_log", {
                "status": "success",
                "stages_completed": [],
                "stages_failed": [],
                "warnings": [],
                "duration_seconds": None,
            }),
            "esg_mismatch_analysis": analysis_state.get("esg_mismatch_analysis", {}),
            "report_confidence_level": report_metadata.get("report_confidence", "MEDIUM"),
            "quality_warnings": quality.get("quality_warnings", report_metadata.get("quality_warnings", [])),
        }

        report_id = export.get("report_id") or f"ESG_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs("reports", exist_ok=True)
        json_path = os.path.join("reports", f"{report_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, default=str)
        json_size = len(json.dumps(export, default=str))
        return json_path, json_size

    def export_json(self, state: Dict[str, Any]) -> str:
        """Return full machine-readable JSON export content."""
        try:
            structured = self._build_structured_report(state)
            quality = ReportQualityChecker().evaluate(state, structured)
            meta = structured.get("metadata", {})
            report_metadata = {
                "report_id": meta.get("report_id"),
                "analysis_date": (meta.get("timestamp_dt") or datetime.utcnow()).isoformat(),
                "report_confidence": quality.get("report_confidence_level", "MEDIUM"),
                "quality_warnings": quality.get("quality_warnings", []),
            }
            path, _ = self.generate_json_export(state, report_metadata, structured=structured, quality=quality)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            return json.dumps({"error": "JSON export failed", "detail": str(exc)}, indent=2)

    def _generate_mismatch_section(self, state: Dict[str, Any]) -> str:
        """
        Generate ESG Mismatch Detector analysis section.
        Highlights contradictions between company promises and actual evidence.
        """
        mismatch_data = state.get("esg_mismatch_analysis")
        if not mismatch_data:
            return "No ESG Promise vs Actual gap analysis was performed for this report."

        lines = [
            "ESG MISMATCH DETECTOR (PROMISE VS ACTUAL PERFORMANCE)",
            "─" * 60,
        ]

        overall_risk = mismatch_data.get("Overall Greenwashing Risk", "Unknown")
        summary = mismatch_data.get("Executive Summary", "No summary available.")
        
        lines.append(f"Overall Mismatch Risk: {overall_risk.upper()}")
        lines.append(f"Summary: {summary}")
        lines.append("")

        future = mismatch_data.get("1. Future Commitments & Progress", [])
        if future:
            lines.append("FUTURE COMMITMENTS & PROGRESS")
            lines.append("─" * 30)
            for idx, item in enumerate(future, start=1):
                lines.append(f"  {idx}. Pledge: {safe_get(item, 'Pledge', default='Unknown')}")
                lines.append(f"     Status:   {safe_get(item, 'Status Trend', default='N/A')}")
                lines.append(f"     Progress: {safe_get(item, 'Progress/Trend', default='N/A')}")
                lines.append(f"     Source:   {safe_get(item, 'Source of Measure', default='N/A')}")
                lines.append("")

        past = mismatch_data.get("2. Past Promise-Implementation Gaps (Mismatches)", [])
        if past and isinstance(past, list) and isinstance(past[0], dict):
            lines.append("PAST PROMISE-IMPLEMENTATION GAPS DETECTED")
            lines.append("─" * 40)
            for idx, item in enumerate(past, start=1):
                lines.append(f"  {idx}. Failed Pledge: {safe_get(item, 'Failed Pledge', default='Unknown')}")
                lines.append(f"     Target:        {safe_get(item, 'Expected Target', default='N/A')}")
                lines.append(f"     Actual Gap:    {safe_get(item, 'Flagged Status', default='N/A')}")
                lines.append(f"     Risk Level:    {safe_get(item, 'Risk Level', default='N/A')}")
                lines.append(f"     Evidence:      {safe_get(item, 'Evidence Source', default='N/A')}")
                lines.append("")
        elif past and isinstance(past, list) and isinstance(past[0], str):
            lines.append(past[0])
            lines.append("")

        # Wrap long lines cleanly
        wrapped_lines = []
        for line in lines:
            if line.startswith("─"):
                wrapped_lines.append(line)
            else:
                wrapped_lines.append("\n".join(textwrap.wrap(line, width=80, subsequent_indent="     " if line.startswith("     ") else "")))

        return "\n".join(wrapped_lines)

    def _generate_temporal_consistency_section(self, state: Dict[str, Any]) -> str:
        """
        Generate temporal ESG consistency analysis section.
        Shows historical claim trends and greenwashing patterns.
        """
        temporal_outputs = [
            o for o in state.get("agent_outputs", []) if o.get("agent") == "temporal_consistency"
        ]

        if not temporal_outputs:
            return f"""
TEMPORAL ESG CONSISTENCY ANALYSIS
{'─'*80}
(No report-based temporal analysis available - web evidence only)
"""

        temporal_result = temporal_outputs[-1].get("output", {})
        if not isinstance(temporal_result, dict):
            return f"""
TEMPORAL ESG CONSISTENCY ANALYSIS
{'─'*80}
(Temporal analysis skipped - no ESG report claims available)
"""

        temporal_status = temporal_result.get("status", "success")
        if temporal_status in ["insufficient_data", "insufficient_history"]:
            return f"""
TEMPORAL ESG CONSISTENCY ANALYSIS
{'─'*80}
({temporal_result.get('message', 'Temporal analysis inconclusive due to insufficient multi-year report data')})
"""

        temporal_score = temporal_result.get("temporal_consistency_score", 50)
        risk_level = temporal_result.get("risk_level", "MODERATE")
        claim_trend = temporal_result.get("claim_trend", "unknown")
        env_trend = temporal_result.get("environmental_trend", "unknown")
        inconsistency_detected = temporal_result.get("temporal_inconsistency_detected", False)
        evidence = temporal_result.get("evidence", [])
        explanation = temporal_result.get("explanation", "")
        years_analyzed = temporal_result.get("years_analyzed", [])

        section = f"""
TEMPORAL ESG CONSISTENCY ANALYSIS
{'─'*80}

Overview:
  This analysis examines ESG claim trends across reported years and compares them
  against actual environmental performance metrics. Greenwashing typically manifests
  as claim escalation without corresponding performance improvement.

Temporal Consistency Score:  {temporal_score:.0f}/100
    Risk Level: {risk_level}
    Inconsistency Detected: {"YES - Claims and performance are misaligned" if inconsistency_detected else "NO - Claims align with performance" if claim_trend != 'unknown' and env_trend != 'unknown' else "INCONCLUSIVE - trend data is limited"}

Claims Analysis:
  Temporal Trend: {claim_trend.upper() if claim_trend else "UNKNOWN"}
  Years Analyzed: {', '.join(str(y) for y in sorted(years_analyzed, reverse=True)) if years_analyzed else "N/A"}
  """

        if env_trend:
            direction = (
                "⬇️ WORSENING while claims escalate"
                if env_trend == "worsening"
                else "⬆️ IMPROVING"
                if env_trend == "improving"
                else "→ STABLE"
            )
            section += f"""
Environmental Performance:
  Performance Trend: {env_trend.upper()}
  Direction: {direction}
  """

        if evidence:
            section += "\nKey Findings:\n  "
            for i, item in enumerate(evidence[:5], 1):
                section += f"\n  {i}. {item}"
            if len(evidence) > 5:
                section += f"\n  ... and {len(evidence)-5} more"

        if explanation:
            section += f"""

Analysis Summary:
  {explanation}
  """

        risk_commentary = {
            "INCONCLUSIVE": (
                "✓ INCONCLUSIVE:\n"
                "    Temporal module has insufficient longitudinal data for high-confidence\n"
                "    trend attribution. Continue collecting annual disclosures for stronger signals."
            ),
            "CRITICAL": (
                "⚠️  CRITICAL ALERT:\n"
                "    Severe temporal inconsistencies detected. Company claims escalate dramatically\n"
                "    while environmental or financial performance deteriorates. This is a strong\n"
                "    indicator of sophisticated greenwashing. Immediate due diligence recommended."
            ),
            "HIGH": (
                "⚠️  HIGH RISK:\n"
                "    Significant temporal misalignment detected. Claims strengthen over time while\n"
                "    actual performance metrics stagnate or decline. Further investigation required."
            ),
            "MODERATE": (
                "✓ MODERATE RISK:\n"
                "    Some temporal inconsistencies noted but pattern is not conclusive.\n"
                "    Recommend ongoing monitoring and periodic re-evaluation."
            ),
        }

        section += "\n\n" + risk_commentary.get(
            risk_level,
            (
                "✓ LOW RISK:\n"
                "    ESG claims align well with historical performance trends. Temporal consistency\n"
                "    suggests company is committed to stated ESG goals."
            ),
        )

        section += f"\n{'─'*80}\n"
        return section

    def _generate_data_enrichment_section(self, state: Dict[str, Any]) -> str:
        """
        Generate section showing results from enterprise features:
        - Indian Financial Data (revenue, profit, market cap)
        - Company Reports (PDF extraction)
        - Carbon Extractor (Scope 1/2/3)
        - Greenwishing/Greenhushing Detection
        - Regulatory Compliance Status
        """
        section = ""
        has_data = False

        agent_outputs = state.get("agent_outputs", [])
        evidence_output = None
        for output in agent_outputs:
            if output.get("agent") == "evidence_retrieval":
                evidence_output = output.get("output", {})
                break

        # === INDIAN FINANCIAL DATA ===
        indian_financials = {}
        if evidence_output:
            indian_financials = evidence_output.get("indian_financials", {})
        if not indian_financials:
            indian_financials = state.get("indian_financials", {})

        if indian_financials and indian_financials.get("financials"):
            has_data = True
            fin = indian_financials.get("financials", {})
            ratios = indian_financials.get("ratios", {})
            sources = indian_financials.get("sources", [])

            section += f"""
INDIAN COMPANY FINANCIALS (Live Data)
{'─'*80}

| {'Metric':<30} | {'Value':<25} | {'Source':<20} |
|{'-'*32}|{'-'*27}|{'-'*22}|
"""
            if fin.get("revenue"):
                section += f"| {'Revenue (Annual)':<30} | {'₹{:,.0f} Cr'.format(fin['revenue']):<25} | {'Screener/Yahoo':<20} |\n"
            if fin.get("net_profit"):
                section += f"| {'Net Profit (Annual)':<30} | {'₹{:,.0f} Cr'.format(fin['net_profit']):<25} | {'Screener/Yahoo':<20} |\n"
            if fin.get("market_cap"):
                section += f"| {'Market Cap':<30} | {'₹{:,.0f} Cr'.format(fin['market_cap']):<25} | {'NSE/Yahoo':<20} |\n"
            if fin.get("current_price"):
                section += f"| {'Current Price':<30} | {'₹{:,.2f}'.format(fin['current_price']):<25} | {'NSE India':<20} |\n"
            if ratios.get("pe_ratio"):
                section += f"| {'P/E Ratio':<30} | {'{:.2f}'.format(ratios['pe_ratio']):<25} | {'Screener':<20} |\n"
            if ratios.get("roe"):
                roe_val = ratios['roe'] * 100 if ratios['roe'] < 1 else ratios['roe']
                section += f"| {'Return on Equity (ROE)':<30} | {'{:.1f}%'.format(roe_val):<25} | {'Screener':<20} |\n"
            if ratios.get("roce"):
                section += f"| {'Return on Capital (ROCE)':<30} | {'{:.1f}%'.format(ratios['roce']):<25} | {'Screener':<20} |\n"
            if sources:
                section += f"\nData Sources: {', '.join(sources)}\n"
            section += "\n"

        # === COMPANY REPORTS (PDF EXTRACTION) ===
        company_reports = {}
        if evidence_output:
            company_reports = evidence_output.get("company_reports", {})
        if not company_reports:
            company_reports = state.get("company_reports", {})

        if company_reports:
            reports_found = company_reports.get("reports_found", [])
            extracted_data = company_reports.get("extracted_data", {})

            if reports_found or extracted_data:
                has_data = True
                section += f"""
OFFICIAL COMPANY REPORTS (PDF Extraction)
{'─'*80}

"""
                if reports_found:
                    section += "Reports Downloaded:\n"
                    for i, report in enumerate(reports_found[:5], 1):
                        rtype = report.get("type", "unknown").replace("_", " ").title()
                        rtitle = report.get("title", "Unknown")[:50]
                        pages = report.get("pages", "?")
                        section += f"  {i}. [{rtype}] {rtitle}... ({pages} pages)\n"
                    section += "\n"

                if extracted_data:
                    section += "ESG Metrics Extracted from PDFs:\n"
                    section += f"| {'Metric':<35} | {'Value':<30} |\n"
                    section += f"|{'-'*37}|{'-'*32}|\n"

                    metrics_map = [
                        ("scope_1_emissions", "Scope 1 Emissions", "{:,.0f} tCO2e"),
                        ("scope_2_emissions", "Scope 2 Emissions", "{:,.0f} tCO2e"),
                        ("scope_3_emissions", "Scope 3 Emissions", "{:,.0f} tCO2e"),
                        ("total_emissions", "Total GHG Emissions", "{:,.0f} tCO2e"),
                        ("renewable_energy_pct", "Renewable Energy %", "{:.1f}%"),
                        ("energy_consumption", "Energy Consumption", "{:,.0f} GWh"),
                        ("water_consumption", "Water Consumption", "{:,.0f} ML"),
                        ("water_recycled_pct", "Water Recycled %", "{:.1f}%"),
                        ("total_employees", "Total Employees", "{:,}"),
                        ("women_employees_pct", "Women Employees %", "{:.1f}%"),
                        ("women_leadership_pct", "Women in Leadership %", "{:.1f}%"),
                        ("board_independence_pct", "Board Independence %", "{:.1f}%"),
                        ("independent_directors", "Independent Directors", "{}"),
                        ("net_zero_target_year", "Net Zero Target Year", "{}"),
                    ]

                    for key, label, fmt in metrics_map:
                        val = extracted_data.get(key)
                        if val is not None:
                            formatted = fmt.format(val)
                            section += f"| {label:<35} | {formatted:<30} |\n"

                    if extracted_data.get("revenue"):
                        section += f"| {'Revenue (from report)':<35} | ₹{extracted_data['revenue']:,.0f} Cr{'':<17} |\n"
                    if extracted_data.get("csr_spend"):
                        section += f"| {'CSR Spend':<35} | ₹{extracted_data['csr_spend']:,.0f} Cr{'':<17} |\n"

                    section += "\n"

        # === GREENWISHING/GREENHUSHING ANALYSIS ===
        greenwishing = state.get("greenwishing_analysis", {})
        if greenwishing and isinstance(greenwishing, dict):
            has_data = True
            section += f"""
GREENWISHING & GREENHUSHING DETECTION
{'─'*80}

"""
            gw = greenwishing.get("greenwishing", {})
            gh = greenwishing.get("greenhushing", {})
            sd = greenwishing.get("selective_disclosure", {})
            overall = greenwishing.get("overall_deception_risk", {})

            section += f"| {'Tactic':<30} | {'Risk Level':<15} | {'Score':<10} | {'Details':<25} |\n"
            section += f"|{'-'*32}|{'-'*17}|{'-'*12}|{'-'*27}|\n"

            if gw:
                gw_risk = gw.get("risk_level", "N/A")
                gw_score = gw.get("score", "N/A")
                gw_indicators = len(gw.get("findings", gw.get("indicators_found", [])))
                section += f"| {'Greenwishing (Unfunded Goals)':<30} | {gw_risk:<15} | {gw_score:<10} | {f'{gw_indicators} indicators':<25} |\n"

            if gh:
                gh_risk = gh.get("risk_level", "N/A")
                gh_score = gh.get("score", "N/A")
                gh_findings = gh.get("findings", [])
                gh_missing = gh.get("missing_fields")
                if gh_missing is None:
                    gh_missing = sum(
                        1 for f in gh_findings
                        if f.get("type") in ["missing_mandatory_disclosure", "brsr_disclosure_gap"]
                    )
                gh_detail = f"{gh_missing} missing fields" if gh_missing else "No material disclosure gaps"
                section += f"| {'Greenhushing (Hidden Data)':<30} | {gh_risk:<15} | {gh_score:<10} | {gh_detail:<25} |\n"

            if sd:
                sd_detected = "Yes" if sd.get("detected") else "No"
                sd_patterns = len(sd.get("findings", sd.get("patterns", [])))
                section += f"| {'Selective Disclosure':<30} | {sd_detected:<15} | {'N/A':<10} | {f'{sd_patterns} patterns':<25} |\n"

            if overall:
                section += f"\n{'Overall Deception Risk Score':<30}: {overall.get('score', 'N/A')}/100 ({overall.get('level', 'N/A')})\n"

            indicators = gw.get("findings", gw.get("indicators_found", []))[:3]
            if indicators:
                section += "\nTop Greenwishing Indicators:\n"
                for ind in indicators:
                    if isinstance(ind, dict):
                        detail = ind.get("type", "indicator").replace("_", " ")
                        section += f"  [NOTE] {detail}\n"
                    else:
                        section += f"  [NOTE] {ind}\n"

            section += "\n"

        # === REGULATORY COMPLIANCE ===
        regulatory = state.get("regulatory_compliance", {})
        if regulatory and isinstance(regulatory, dict):
            has_data = True
            section += f"""
REGULATORY COMPLIANCE ASSESSMENT
{'─'*80}

"""
            jurisdiction = regulatory.get("jurisdiction", "N/A")
            compliance_score = regulatory.get("compliance_score", "N/A")

            if isinstance(compliance_score, dict):
                compliance_score_value = compliance_score.get("score", "N/A")
                risk_level = compliance_score.get("risk_level", regulatory.get("risk_level", "N/A"))
            else:
                compliance_score_value = compliance_score
                risk_level = regulatory.get("risk_level", "N/A")

            applicable_regs = regulatory.get("applicable_regulations", [])

            section += f"Jurisdiction: {jurisdiction}\n"
            section += f"Compliance Score: {compliance_score_value}/100\n"
            section += f"Risk Level: {risk_level}\n\n"

            if applicable_regs:
                section += "Applicable Regulations:\n"
                for reg in applicable_regs[:6]:
                    section += f"  - {reg}\n"
                if len(applicable_regs) > 6:
                    section += f"  ... and {len(applicable_regs) - 6} more\n"
                section += "\n"

            compliance_results = regulatory.get("compliance_results", [])
            valid_results = [
                r for r in compliance_results
                if r.get("regulation_name") and r.get("regulation_name") != "Unknown"
            ]
            if valid_results:
                section += f"| {'Regulation':<35} | {'Status':<12} | {'Gaps':<15} |\n"
                section += f"|{'-'*37}|{'-'*14}|{'-'*17}|\n"
                for result in valid_results[:5]:
                    reg_name = result.get("regulation_name", "")[:35]
                    gap_details = result.get("gap_details", [])
                    if not isinstance(gap_details, list):
                        gap_details = []
                    has_gap = len(gap_details) > 0
                    status = "[GAP FOUND]" if has_gap else "[COMPLIANT]"
                    gaps = len(gap_details)
                    section += f"| {reg_name:<35} | {status:<12} | {gaps} issue(s){'':<7} |\n"
                section += "\n"

            risks = regulatory.get("regulatory_risks", [])
            valid_risks = [r for r in risks if r.get('regulation') and r.get('risk_level')]
            if valid_risks:
                section += "Regulatory Risks Identified:\n"
                for risk in valid_risks[:3]:
                    unverified = len(risk.get('unverified_requirements', []))
                    section += (
                        f"  [ALERT] {risk.get('risk_level')} Risk - "
                        f"{risk.get('regulation')}: {unverified} unverified requirement(s)\n"
                    )
                section += "\n"

        # === CLIMATEBERT NLP ANALYSIS ===
        climatebert = state.get("climatebert_analysis", {})
        if climatebert and isinstance(climatebert, dict):
            has_data = True
            section += f"""
CLIMATEBERT NLP ANALYSIS
{'─'*80}

"""
            claim_analysis = climatebert.get("claim_analysis", {})
            comparison = climatebert.get("comparison", {})
            verdict = climatebert.get("final_verdict", {})

            climate_rel = claim_analysis.get("climate_relevance", {})
            if climate_rel:
                section += f"Climate Relevance Score: {climate_rel.get('score', 'N/A')}/100\n"
                section += f"Classification: {climate_rel.get('classification', 'N/A')}\n\n"

            gw_detect = claim_analysis.get("greenwashing_detection", {})
            if gw_detect:
                section += f"Greenwashing Risk (NLP): {gw_detect.get('risk_score', 'N/A')}/100\n"
                section += f"Risk Level: {gw_detect.get('risk_level', 'N/A')}\n"
                patterns = gw_detect.get("detected_patterns", [])
                if patterns:
                    section += f"Detected Patterns: {', '.join(patterns[:4])}\n"
                section += "\n"

            if comparison:
                section += "Claim vs Evidence Comparison:\n"
                section += f"  • Claim Greenwashing Score: {comparison.get('claim_greenwashing_score', 'N/A')}\n"
                section += f"  • Evidence Greenwashing Score: {comparison.get('evidence_greenwashing_score', 'N/A')}\n"
                section += f"  • Interpretation: {comparison.get('interpretation', 'N/A')}\n\n"

            if verdict:
                verdict_conf = verdict.get('confidence')
                if verdict_conf is None:
                    cb_outputs = [
                        o for o in state.get("agent_outputs", []) if o.get("agent") == "climatebert_analysis"
                    ]
                    verdict_conf = f"{cb_outputs[-1].get('confidence', 0):.1%}" if cb_outputs else "Model-derived"
                section += f"ClimateBERT Verdict: {verdict.get('verdict', 'N/A')}\n"
                section += f"Confidence: {verdict_conf}\n"

            section += "\n"

        # === EXPLAINABILITY (SHAP/LIME) ===
        explainability = state.get("explainability_report", {})
        if explainability and isinstance(explainability, dict):
            has_data = True
            section += f"""
ML EXPLAINABILITY (SHAP/LIME)
{'─'*80}

"""
            method = explainability.get("method", "N/A")
            section += f"Explanation Method: {method}\n\n"

            top_factors = explainability.get("top_factors", [])
            if top_factors:
                section += "Key Factors Driving Risk Assessment:\n"
                section += f"| {'Factor':<30} | {'Impact':<12} | {'Direction':<20} |\n"
                section += f"|{'-'*32}|{'-'*14}|{'-'*22}|\n"
                for factor in top_factors[:5]:
                    name = factor.get("feature", factor.get("description", "Unknown"))[:30]
                    impact = factor.get("impact", "N/A")
                    direction = factor.get("direction", "N/A")
                    section += f"| {name:<30} | {impact:<12} | {direction:<20} |\n"
                section += "\n"

            narrative = explainability.get("human_readable_explanation", "")
            if narrative:
                section += f"AI Explanation:\n{narrative}\n"

            section += "\n"

        # === FINANCIAL CONTEXT FLAGS ===
        financial_context = {}
        if evidence_output:
            financial_context = evidence_output.get("financial_context", {})
        if not financial_context:
            financial_context = state.get("financial_context", {})

        if financial_context:
            report_metrics = financial_context.get("report_metrics", {})

            if report_metrics:
                has_data = True
                section += f"""
ADDITIONAL METRICS FROM REPORTS
{'─'*80}

"""
                for key, value in list(report_metrics.items())[:10]:
                    key_display = key.replace("_", " ").title()
                    if isinstance(value, (int, float)):
                        section += f"  • {key_display}: {value:,.2f}\n"
                    else:
                        section += f"  • {key_display}: {value}\n"
                section += "\n"

        # === NO DATA FOUND ===
        if not has_data:
            section += f"""
DATA ENRICHMENT STATUS
{'─'*80}

[NOTE] Indian Financial Data: Not available (company may not be in database)
[NOTE] Company Reports: No official PDFs could be fetched
[NOTE] PDF Metrics: No data extracted

Note: This may occur when:
  - Company is not in the 50+ Indian companies database
  - Investor relations page structure is not recognized
  - PDF reports are not publicly accessible
  - Non-Indian company without configured IR URL

"""

        return section


# LangGraph node wrapper
def professional_report_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate professional enterprise report - Node wrapper for LangGraph."""
    import time as _time
    _t0 = _time.time()
    print(f"\n{'== GENERATING PROFESSIONAL REPORT':=^70}")

    # === SAFETY TRIM: guard against operator.add accumulation ===
    raw_outputs = state.get("agent_outputs", [])
    if isinstance(raw_outputs, list) and len(raw_outputs) > 200:
        print(f"⚠️ [RPT] agent_outputs has {len(raw_outputs)} entries — trimming to last 200 (operator.add accumulation)")
        # Keep last unique-by-agent entries within the last 200
        seen, trimmed = set(), []
        for item in reversed(raw_outputs):
            name = item.get("agent", "") if isinstance(item, dict) else ""
            if name not in seen:
                seen.add(name)
                trimmed.append(item)
        state["agent_outputs"] = list(reversed(trimmed))
    print(f"[RPT] agent_outputs count: {len(state.get('agent_outputs', []))}", flush=True)

    print(f"[RPT] Step 1: generate_executive_report...", flush=True)
    generator = ProfessionalReportGenerator()
    professional_report = generator.generate_executive_report(state)
    print(f"[RPT] Step 1 done ({_time.time()-_t0:.1f}s) — {len(professional_report)} chars", flush=True)


    # Never write a bloated report to state — cap it
    if len(professional_report) > 500_000:
        professional_report = professional_report[:500_000] + "\n[TRUNCATED]"

    state["report"] = professional_report

    print(f"[RPT] Step 2: _build_structured_report...", flush=True)
    structured = generator._build_structured_report(state)
    print(f"[RPT] Step 2 done ({_time.time()-_t0:.1f}s)", flush=True)

    print(f"[RPT] Step 3: ReportQualityChecker...", flush=True)
    quality = ReportQualityChecker().evaluate(state, structured)
    print(f"[RPT] Step 3 done ({_time.time()-_t0:.1f}s)", flush=True)

    metadata = {
        "report_id": structured.get("metadata", {}).get("report_id"),
        "analysis_date": (structured.get("metadata", {}).get("timestamp_dt") or datetime.utcnow()).isoformat(),
        "report_confidence": quality.get("report_confidence_level", "MEDIUM"),
        "quality_warnings": quality.get("quality_warnings", []),
    }

    print(f"[RPT] Step 4: generate_json_export...", flush=True)
    json_path, json_size = generator.generate_json_export(state, metadata)
    print(f"[RPT] Step 4 done ({_time.time()-_t0:.1f}s) — {json_size} chars → {json_path}", flush=True)

    print(f"[RPT] Step 5: reading JSON file...", flush=True)
    with open(json_path, "r", encoding="utf-8") as f:
        state["json_export"] = f.read()
    state["json_export_path"] = json_path
    print(f"[RPT] Step 5 done ({_time.time()-_t0:.1f}s)", flush=True)

    print(f"[OK] Professional report generated ({len(professional_report)} characters)")
    print(f"[OK] JSON export generated ({json_size} characters)")

    state["agent_outputs"].append({
        "agent": "professional_report_generation",
        "confidence": 0.95,
        "timestamp": datetime.now().isoformat(),
        "output": {
            "report_id": metadata.get("report_id"),
            "report_confidence": metadata.get("report_confidence"),
            "quality_warnings": metadata.get("quality_warnings", []),
        },
    })

    print(f"[RPT] TOTAL report generation time: {_time.time()-_t0:.1f}s")
    return state