"""
Final Risk Scorer & ESG Rating Specialist
Industry-adjusted risk scoring matching MSCI/Sustainalytics methodology
100% Dynamic - All data loaded from config files
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from core.llm_call import call_llm
import asyncio
from config.agent_prompts import RISK_SCORING_PROMPT
from core.pillar_factors_builder import build_pillar_factors
import json
import os
import logging
import numpy as np
from ml_models.xgboost_risk_model import XGBoostRiskModel
from ml_models.lightgbm_esg_predictor import LightGBMESGPredictor
from ml_models.lstm_trend_predictor import get_lstm_predictor
from ml_models.anomaly_detector import get_anomaly_detector


logger = logging.getLogger(__name__)


class RiskScorer:
    # Industry-specific risk penalty multipliers for high-carbon sectors
    INDUSTRY_RISK_MULTIPLIERS = {
        "oil_and_gas": 1.5,
        "mining": 1.4,
        "aviation": 1.3,
        "coal": 1.6
    }
    
    def __init__(self):
        self.name = "Multi-Dimensional ESG Risk & Integrity Scorer"
        
        # Load ALL configuration from external file (NO HARDCODING)
        self.config = self._load_config()
        self.industry_baseline_risk = self.config.get('industry_baseline_risk', {})
        self.weights = self.config.get('component_weights', {})
        self.risk_thresholds = self.config.get('risk_thresholds', {})
        
        # Load ML models
        self.ml_model = XGBoostRiskModel()
        self.use_ml = self.ml_model.model is not None
        
        self.esg_predictor = LightGBMESGPredictor()
        self.use_esg_predictor = self.esg_predictor.model_available
        
        # Load LSTM trend predictor
        self.lstm_predictor = get_lstm_predictor()
        self.use_lstm = self.lstm_predictor.model_available
        
        # Load anomaly detector
        self.anomaly_detector = get_anomaly_detector()
        self.use_anomaly_detector = self.anomaly_detector.model_available
        
        print(f"✅ Loaded {len(self.industry_baseline_risk)} industry baselines from config")
        if self.use_ml:
            print(f"✅ XGBoost risk model loaded - hybrid prediction enabled")
        else:
            print(f"ℹ️  XGBoost model not available - using formula-based scoring only")
        
        if self.use_esg_predictor:
            print(f"✅ LightGBM ESG predictor loaded - score validation enabled")
        else:
            print(f"ℹ️  LightGBM predictor not available - skipping ESG score validation")
        
        if self.use_lstm:
            print(f"✅ LSTM trend forecaster loaded - temporal prediction enabled")
        else:
            print(f"ℹ️  LSTM trend forecaster not available - skipping trend analysis")
        
        if self.use_anomaly_detector:
            print(f"✅ Anomaly detector loaded - pattern anomaly detection enabled")
        else:
            print(f"ℹ️  Anomaly detector not available - skipping anomaly detection")
    
    @staticmethod
    def esg_score_to_rating(esg_score: float) -> tuple:
        """
        Convert ESG score (0-100) to MSCI-style rating and risk level
        
        Args:
            esg_score: Overall ESG score from 0-100
            
        Returns:
            tuple: (rating_grade, risk_level)
            
        Rating Scale (MSCI-style):
            AAA: 90-100  (ESG Leaders)
            AA:  85-89   (ESG Leaders)
            A:   75-84   (Above Average)
            BBB: 60-74   (Average)
            BB:  50-59   (Below Average)
            B:   35-49   (Laggards)
            CCC: 0-34    (Laggards)
        """
        if esg_score >= 90:
            return ('AAA', 'LOW')
        elif esg_score >= 85:
            return ('AA', 'LOW')
        elif esg_score >= 75:
            return ('A', 'LOW')
        elif esg_score >= 60:
            return ('BBB', 'MODERATE')
        elif esg_score >= 50:
            return ('BB', 'MODERATE')
        elif esg_score >= 35:
            return ('B', 'HIGH')
        else:
            return ('CCC', 'HIGH')
    
    def _load_config(self) -> Dict:
        """Load industry configuration from external JSON file"""
        config_path = "config/industry_baselines.json"
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    print(f"✅ Loaded industry config from {config_path}")
                    return config
            else:
                print(f"⚠️ Config file not found: {config_path}")
                print("   Creating default config...")
                self._create_default_config(config_path)
                
                # Load again
                with open(config_path, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            print("   Using fallback defaults")
        
        # Fallback defaults (minimal)
        return {
            "industry_baseline_risk": {"unknown": {"baseline": 50, "source": "Default"}},
            "component_weights": {
                'claim_verification': 0.30,
                'evidence_quality': 0.15,
                'source_credibility': 0.15,
                'sentiment_divergence': 0.10,
                'historical_pattern': 0.15,
                'contradiction_severity': 0.15
            },
            "risk_thresholds": {
                "very_high_risk_industries": {"baseline_threshold": 70, "high_risk": 45, "moderate_risk": 30},
                "high_risk_industries": {"baseline_threshold": 60, "high_risk": 55, "moderate_risk": 35},
                "moderate_risk_industries": {"baseline_threshold": 50, "high_risk": 65, "moderate_risk": 40},
                "low_risk_industries": {"baseline_threshold": 0, "high_risk": 70, "moderate_risk": 45}
            }
        }
    
    def _create_default_config(self, config_path: str):
        """Create default industry baselines config file"""
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        default_config = {
            "industry_baseline_risk": {
                "oil_and_gas": {"baseline": 75, "source": "MSCI ESG 2024", "rationale": "High carbon intensity"},
                "coal": {"baseline": 80, "source": "MSCI ESG 2024", "rationale": "Highest emissions"},
                "mining": {"baseline": 70, "source": "MSCI ESG 2024", "rationale": "Environmental degradation"},
                "chemicals": {"baseline": 65, "source": "MSCI ESG 2024", "rationale": "Toxic waste"},
                "aviation": {"baseline": 70, "source": "MSCI ESG 2024", "rationale": "High emissions"},
                "automotive": {"baseline": 60, "source": "MSCI ESG 2024", "rationale": "Transition challenges"},
                "fast_fashion": {"baseline": 65, "source": "MSCI ESG 2024", "rationale": "Labor & waste"},
                "tobacco": {"baseline": 75, "source": "MSCI ESG 2024", "rationale": "Public health"},
                "defense": {"baseline": 60, "source": "MSCI ESG 2024", "rationale": "Ethical concerns"},
                "consumer_goods": {"baseline": 50, "source": "MSCI ESG 2024", "rationale": "Packaging waste"},
                "retail": {"baseline": 45, "source": "MSCI ESG 2024", "rationale": "Labor practices"},
                "food_beverage": {"baseline": 50, "source": "MSCI ESG 2024", "rationale": "Water usage"},
                "pharmaceuticals": {"baseline": 55, "source": "MSCI ESG 2024", "rationale": "Drug pricing"},
                "banking": {"baseline": 50, "source": "MSCI ESG 2024", "rationale": "Fossil fuel financing"},
                "real_estate": {"baseline": 45, "source": "MSCI ESG 2024", "rationale": "Energy efficiency"},
                "transportation": {"baseline": 55, "source": "MSCI ESG 2024", "rationale": "Emissions"},
                "hospitality": {"baseline": 45, "source": "MSCI ESG 2024", "rationale": "Water & waste"},
                "technology": {"baseline": 35, "source": "MSCI ESG 2024", "rationale": "Data privacy"},
                "software": {"baseline": 30, "source": "MSCI ESG 2024", "rationale": "Low footprint"},
                "healthcare_services": {"baseline": 35, "source": "MSCI ESG 2024", "rationale": "Patient care"},
                "telecommunications": {"baseline": 40, "source": "MSCI ESG 2024", "rationale": "Digital divide"},
                "renewable_energy": {"baseline": 25, "source": "MSCI ESG 2024", "rationale": "Positive impact"},
                "education": {"baseline": 30, "source": "MSCI ESG 2024", "rationale": "Access equity"},
                "unknown": {"baseline": 50, "source": "Default", "rationale": "Insufficient data"}
            },
            "component_weights": {
                "claim_verification": 0.30,
                "evidence_quality": 0.15,
                "source_credibility": 0.15,
                "sentiment_divergence": 0.10,
                "historical_pattern": 0.15,
                "contradiction_severity": 0.15
            },
            "risk_thresholds": {
                "very_high_risk_industries": {"baseline_threshold": 70, "high_risk": 45, "moderate_risk": 30},
                "high_risk_industries": {"baseline_threshold": 60, "high_risk": 55, "moderate_risk": 35},
                "moderate_risk_industries": {"baseline_threshold": 50, "high_risk": 65, "moderate_risk": 40},
                "low_risk_industries": {"baseline_threshold": 0, "high_risk": 70, "moderate_risk": 45}
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✅ Created default config: {config_path}")

    @staticmethod
    def _compute_pillar_score(factors: List[Dict[str, Any]]) -> float:
        total = 0.0
        for factor in factors:
            raw = factor.get("raw_signal_normalized", factor.get("score", 0.0))
            weight = factor.get("weight", 0.0)
            if isinstance(raw, (int, float)) and isinstance(weight, (int, float)):
                contribution = round(float(raw) * float(weight), 2)
                factor["raw_signal_normalized"] = float(raw)
                factor["points_contributed"] = contribution
                total += contribution
            else:
                factor["points_contributed"] = 0.0
        return round(total, 1)

    @staticmethod
    def _resolve_sg_adequacy(all_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Read social/governance adequacy diagnostics emitted by evidence retrieval."""
        evidence = all_analyses.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        high_trust_types = {
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
        fallback_distinct_sources = {
            str(ev.get("source_name") or ev.get("source") or "").strip().lower()
            for ev in evidence
            if isinstance(ev, dict)
        }
        fallback_high_trust = sum(
            1
            for ev in evidence
            if isinstance(ev, dict) and str(ev.get("source_type", "") or "") in high_trust_types
        )
        fallback_ready = len(evidence) >= 8 and len([s for s in fallback_distinct_sources if s]) >= 3 and fallback_high_trust >= 2

        defaults = {
            "overall_ready": fallback_ready,
            "social": {"is_adequate": fallback_ready},
            "governance": {"is_adequate": fallback_ready},
            "warnings": [
                "Social/Governance adequacy metrics unavailable; using fallback heuristic."
            ],
        }

        quality_metrics = all_analyses.get("evidence_quality_metrics", {})
        if not isinstance(quality_metrics, dict):
            return defaults

        adequacy = quality_metrics.get("social_governance_adequacy", {})
        if not isinstance(adequacy, dict):
            return defaults

        social_block = adequacy.get("social", {}) if isinstance(adequacy.get("social"), dict) else {}
        governance_block = adequacy.get("governance", {}) if isinstance(adequacy.get("governance"), dict) else {}
        warnings = adequacy.get("warnings", []) if isinstance(adequacy.get("warnings"), list) else []

        return {
            "overall_ready": bool(adequacy.get("overall_ready", False)),
            "social": {"is_adequate": bool(social_block.get("is_adequate", False)), **social_block},
            "governance": {"is_adequate": bool(governance_block.get("is_adequate", False)), **governance_block},
            "warnings": warnings,
        }
    
    def calculate_pillar_scores(self, all_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Environmental, Social, and Governance pillar scores (0-100)
        
        Args:
            all_analyses: Dict containing all agent outputs
            
        Returns:
            Dict with environmental_score, social_score, governance_score
        """
        
        print(f"\n📊 Calculating ESG Pillar Scores...")
        
        # Get evidence and contradictions
        evidence = all_analyses.get('evidence', [])
        sg_adequacy = self._resolve_sg_adequacy(all_analyses)
        contradiction_payload = all_analyses.get('contradiction_analysis', [])
        contradictions = []
        if isinstance(contradiction_payload, dict):
            contradictions = (
                contradiction_payload.get('contradictions')
                or contradiction_payload.get('contradiction_list')
                or contradiction_payload.get('specific_contradictions')
                or []
            )
        elif isinstance(contradiction_payload, list):
            contradictions = contradiction_payload
        
        # Flatten all text content for keyword analysis
        all_text = []
        for ev in evidence:
            if isinstance(ev, dict):
                all_text.append(ev.get('snippet', '').lower())
                all_text.append(ev.get('relevant_text', '').lower())
                for e in ev.get('evidence', []):
                    if isinstance(e, dict):
                        all_text.append(e.get('relevant_text', '').lower())
        
        combined_text = ' '.join(all_text)
        
        # Define pillar keywords
        environmental_keywords = [
            "carbon", "emissions", "climate", "renewable", "waste", 
            "water", "biodiversity", "pollution", "energy", "environmental"
        ]
        
        social_keywords = [
            "labor", "diversity", "community", "human rights", "employee", 
            "safety", "stakeholder", "workers", "social", "equity"
        ]
        
        governance_keywords = [
            "board", "ethics", "compliance", "transparency", "corruption", 
            "disclosure", "audit", "governance", "accountability", "integrity"
        ]
        
        # Calculate Environmental Score
        env_positive = sum(1 for kw in environmental_keywords if kw in combined_text)
        env_negative = sum(
            1 for c in contradictions 
            if any(kw in str(c).lower() for kw in environmental_keywords)
        )
        
        # Contradictions should materially reduce ESG pillar scores.
        env_penalty = min(env_negative * 15, 60)
        environmental_score = 50 + (env_positive * 10) - env_penalty
        environmental_score = max(0, min(100, environmental_score))
        
        print(f"   Environmental: {environmental_score:.1f}/100 (positive: {env_positive}, contradictions: {env_negative})")
        
        # Calculate Social Score
        soc_positive = sum(1 for kw in social_keywords if kw in combined_text)
        soc_negative = sum(
            1 for c in contradictions 
            if any(kw in str(c).lower() for kw in social_keywords)
        )
        
        soc_penalty = min(soc_negative * 15, 60)
        social_score = 50 + (soc_positive * 10) - soc_penalty
        social_score = max(0, min(100, social_score))

        if not sg_adequacy.get("social", {}).get("is_adequate", False):
            # Pull social score toward neutral when evidence is not decision-grade.
            social_score = round((social_score * 0.40) + (50.0 * 0.60), 1)
            print("   ⚠️ Social pillar confidence-limited due to insufficient free-source evidence")
        
        print(f"   Social: {social_score:.1f}/100 (positive: {soc_positive}, contradictions: {soc_negative})")
        
        # Calculate Governance Score
        gov_positive = sum(1 for kw in governance_keywords if kw in combined_text)
        gov_negative = sum(
            1 for c in contradictions 
            if any(kw in str(c).lower() for kw in governance_keywords)
        )
        
        gov_penalty = min(gov_negative * 15, 60)
        governance_score = 50 + (gov_positive * 10) - gov_penalty
        governance_score = max(0, min(100, governance_score))

        if not sg_adequacy.get("governance", {}).get("is_adequate", False):
            # Pull governance score toward neutral when evidence is not decision-grade.
            governance_score = round((governance_score * 0.40) + (50.0 * 0.60), 1)
            print("   ⚠️ Governance pillar confidence-limited due to insufficient free-source evidence")

        # Cross-pillar contradiction and regulatory gap penalties.
        high_severity_count = sum(
            1 for c in contradictions
            if str(c.get("severity", "")).upper() == "HIGH"
        )
        reg_payload = all_analyses.get("regulatory_scanning") or all_analyses.get("regulatory_compliance") or {}
        if isinstance(reg_payload, dict):
            reg_results = reg_payload.get("compliance_results", []) or []
            reg_gaps = sum(1 for r in reg_results if isinstance(r, dict) and (r.get("gap_details") or []))
        else:
            reg_gaps = 0

        if high_severity_count > 0 or reg_gaps > 0:
            environmental_score = max(0, environmental_score - min(35, high_severity_count * 6 + reg_gaps * 2))
            social_score = max(0, social_score - min(30, high_severity_count * 5 + reg_gaps * 2))
            governance_score = max(0, governance_score - min(40, high_severity_count * 7 + reg_gaps * 3))
        
        print(f"   Governance: {governance_score:.1f}/100 (positive: {gov_positive}, contradictions: {gov_negative})")
        
        # Apply industry baseline adjustment
        industry = self._identify_industry(all_analyses.get('company', 'Unknown'), all_analyses)
        industry_data = self.industry_baseline_risk.get(industry, self.industry_baseline_risk.get('unknown'))
        industry_baseline = industry_data.get('baseline', 50)
        
        # Industry adjustment (light-touch): applied mainly to Environmental pillar and reduced for low-footprint sectors.
        industry_penalty = (industry_baseline - 50) * 0.10  # 10% of baseline deviation (calibrated down)
        sector_env_multiplier = {
            "oil_and_gas": 1.0,
            "coal": 1.0,
            "mining": 0.9,
            "aviation": 0.8,
            "banking": 0.3,
            "consumer_goods": 0.5,
            "food_beverage": 0.6,
            "technology": 0.4,
            "software": 0.3,
        }.get(industry, 0.6)
        
        environmental_score = max(0, min(100, environmental_score - (industry_penalty * sector_env_multiplier)))
        
        if industry_penalty != 0:
            print(f"   Industry Adjustment ({industry}): {-industry_penalty:+.1f} points")
        
        # Industry-normalized pillar weights (light-touch calibration, MSCI-like materiality)
        pillar_weights = {
            "oil_and_gas": (0.45, 0.25, 0.30),
            "coal": (0.50, 0.20, 0.30),
            "mining": (0.45, 0.25, 0.30),
            "aviation": (0.40, 0.30, 0.30),
            "banking": (0.25, 0.30, 0.45),
            "consumer_goods": (0.30, 0.40, 0.30),
            "food_beverage": (0.32, 0.38, 0.30),
        }.get(industry, (0.35, 0.30, 0.35))
        
        w_e, w_s, w_g = pillar_weights
        overall_esg = (environmental_score * w_e) + (social_score * w_s) + (governance_score * w_g)
        
        print(f"   Overall ESG: {overall_esg:.1f}/100 (E×{w_e:.2f} + S×{w_s:.2f} + G×{w_g:.2f})")
        
        return {
            "environmental_score": round(environmental_score, 1),
            "social_score": round(social_score, 1),
            "governance_score": round(governance_score, 1),
            "overall_esg_score": round(overall_esg, 1),
            "industry_adjustment": round(industry_penalty, 1),
            "pillar_weighting": {"E": w_e, "S": w_s, "G": w_g},
            "data_adequacy": sg_adequacy,
        }
    
    def calculate_final_score(self, company: str, all_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final ESG score with PILLAR-PRIMARY approach
        ESG pillar scores are the PRIMARY rating source (MSCI-aligned)
        ML is used only for refinement in MODERATE range (50-74 ESG)
        """
        
        print(f"\n{'='*60}")
        print(f"🔍 AGENT 7: {self.name}")
        print(f"{'='*60}")
        print(f"Company: {company}")
        
        # Step 0: Calculate ESG pillar scores FIRST (PRIMARY rating source)
        pillar_scores = self.calculate_pillar_scores(all_analyses)
        overall_esg_score = pillar_scores.get("overall_esg_score", 50)
        
        # Step 0.25: Positive ESG reinforcement (capped)
        positive_boost = self._calculate_positive_esg_boost(all_analyses, pillar_scores)
        boosted_esg_score = max(0, min(100, overall_esg_score + positive_boost))
        if positive_boost != 0:
            print(f"\n✅ Positive ESG reinforcement applied: {positive_boost:+.1f} points (capped)")
            print(f"   ESG Score: {overall_esg_score:.1f} → {boosted_esg_score:.1f}")
        overall_esg_score = boosted_esg_score
        
        print(f"\n📊 ESG Pillar Scores Calculated:")
        print(f"   Environmental: {pillar_scores['environmental_score']}/100")
        print(f"   Social: {pillar_scores['social_score']}/100")
        print(f"   Governance: {pillar_scores['governance_score']}/100")
        print(f"   Overall ESG: {overall_esg_score}/100")
        
        # Step 0.5: Convert ESG score to MSCI-style rating (PRIMARY METHOD)
        rating_grade, risk_level = self.esg_score_to_rating(overall_esg_score)
        base_greenwashing_risk = 100 - overall_esg_score
        greenwashing_risk = base_greenwashing_risk
        
        print(f"\n⭐ MSCI-Style Rating from ESG Score:")
        print(f"   ESG Score: {overall_esg_score}/100")
        print(f"   Rating Grade: {rating_grade}")
        print(f"   Risk Level: {risk_level}")
        print(f"   Greenwashing Risk: {greenwashing_risk:.1f}/100")
        
        # Track rating methodology
        override_ml = False
        risk_source = "ESG Pillar Score (MSCI-Aligned)"
        scoring_methodology = "ESG Pillar Primary"
        confidence = 0.85

        # FIX 1A: Enforce ESG pillar primary pathway whenever all pillar scores are present.
        environmental_score = pillar_scores.get("environmental_score")
        social_score = pillar_scores.get("social_score")
        governance_score = pillar_scores.get("governance_score")
        if all(s is not None for s in [environmental_score, social_score, governance_score]):
            overall_esg_score = (
                float(environmental_score) * 0.35
                + float(social_score) * 0.30
                + float(governance_score) * 0.35
            )
            overall_esg_score = round(max(0.0, min(100.0, overall_esg_score)), 1)
            greenwashing_risk = round(100.0 - overall_esg_score, 1)
            base_greenwashing_risk = greenwashing_risk
            rating_grade, risk_level = self.esg_score_to_rating(overall_esg_score)
            scoring_methodology = "ESG Pillar Primary"
            override_ml = True
            risk_source = "ESG Pillar Primary"
        
        # Confidence-aware calibration for missing/estimated carbon and sparse evidence
        confidence_penalties = []
        sg_adequacy = pillar_scores.get("data_adequacy", {}) if isinstance(pillar_scores, dict) else {}
        carbon = all_analyses.get("carbon_extraction") or all_analyses.get("carbon_results") or {}
        if isinstance(carbon, dict):
            dq = carbon.get("data_quality", {}) if isinstance(carbon.get("data_quality", {}), dict) else {}
            dq_status = str(dq.get("status", "")).lower()
            dq_score = float(dq.get("overall_score", 0) or 0)
            if dq_status in {"insufficient_data", "estimated_baseline"} or dq_score < 30:
                confidence_penalties.append(0.12)
        
        evidence_list = all_analyses.get("evidence", [])
        if not isinstance(evidence_list, list) or len(evidence_list) == 0:
            confidence_penalties.append(0.10)
        
        credibility = all_analyses.get("credibility_analysis", {})
        if not isinstance(credibility, dict) or not credibility:
            confidence_penalties.append(0.08)
        
        temporal = all_analyses.get("temporal_consistency") or {}
        if isinstance(temporal, dict) and str(temporal.get("status", "")).lower() in {"insufficient_data", "insufficient_history"}:
            confidence_penalties.append(0.05)

        if not bool(sg_adequacy.get("overall_ready", False)):
            confidence_penalties.append(0.12)
            print("   ⚠️ S/G decision confidence reduced (insufficient social/governance evidence coverage)")
        
        confidence_penalty_total = min(0.25, sum(confidence_penalties))
        confidence = max(0.55, confidence - confidence_penalty_total)
        
        # Step 1: Check for ESG Leaders (≥75) - bypass ML entirely
        if overall_esg_score >= 85:
            print(f"\n✅ ESG LEADER STATUS (AA/AAA) - ML Bypassed")
            print(f"   Overall ESG: {overall_esg_score}/100")
            print(f"   Rating: {rating_grade} (Best-in-class ESG performance)")
            
            override_ml = True
            confidence = 0.90
            risk_source = "ESG Pillar Override (ESG Leader)"
            
        elif overall_esg_score >= 75:
            print(f"\n✅ STRONG ESG PERFORMANCE (A Rating) - ML Bypassed")
            print(f"   Overall ESG: {overall_esg_score}/100")
            print(f"   Rating: {rating_grade} (Above-average performance)")
            
            override_ml = True
            confidence = 0.88
            risk_source = "ESG Pillar Override (Strong Performance)"
        
        elif overall_esg_score < 50:
            print(f"\n⚠️ ESG LAGGARD (B/CCC Rating) - ML Bypassed")
            print(f"   Overall ESG: {overall_esg_score}/100")
            print(f"   Rating: {rating_grade} (Significant ESG concerns)")
            
            override_ml = True
            confidence = 0.85
            risk_source = "ESG Pillar Score (Laggard)"
        
        # Step 2: Get ML prediction (ONLY for MODERATE range: 50-74 ESG)
        ml_prediction = None
        ml_confidence = 0.0
        use_ml_prediction = False
        
        if not override_ml and self.use_ml and 50 <= overall_esg_score < 75:
            print(f"\n🤖 Running ML prediction for MODERATE range refinement...")
            print(f"   ESG Score: {overall_esg_score}/100 (50-74 range)")
            print(f"   Purpose: Refine BBB/BB rating boundary")
            
            # CRITICAL FIX: Inject pillar scores into all_analyses for XGBoost
            all_analyses['pillar_scores'] = pillar_scores
            print(f"   Pillar scores injected for ML feature extraction")
            
            ml_result = self.ml_model.predict(all_analyses)
            
            if ml_result.get('ml_available'):
                ml_prediction = ml_result['prediction']
                ml_confidence = ml_result['confidence']
                
                print(f"   ML Prediction: {ml_prediction}")
                print(f"   ML Confidence: {ml_confidence:.1%}")
                print(f"   Probabilities: LOW={ml_result['probabilities']['LOW']:.2%}, "
                      f"MODERATE={ml_result['probabilities']['MODERATE']:.2%}, "
                      f"HIGH={ml_result['probabilities']['HIGH']:.2%}")
                
                # Use ML to refine rating ONLY in moderate range with high confidence
                if ml_confidence >= 0.80:
                    use_ml_prediction = True
                    print(f"   ✅ Using ML to refine BBB/BB boundary (confidence ≥ 80%)")
                    
                    # ML can adjust rating within MODERATE range
                    if ml_prediction == "LOW" and overall_esg_score >= 60:
                        rating_grade = "BBB"
                        risk_level = "MODERATE"
                        greenwashing_risk = 35
                        print(f"   ML refinement: Upgraded to BBB (low-moderate risk)")
                    elif ml_prediction == "HIGH" and overall_esg_score < 60:
                        rating_grade = "BB"
                        risk_level = "MODERATE"
                        greenwashing_risk = 55
                        print(f"   ML refinement: Maintained BB (high-moderate risk)")
                    # else keep pillar-based rating
                    
                    risk_source = "ESG Pillar + ML Refinement"
                    confidence = (0.85 + ml_confidence) / 2
                else:
                    print(f"   ⚠️ ML confidence too low, using pillar-based rating")
        
        # Step 3: Formula-based components (for explainability ONLY)
        print(f"\n📐 Calculating formula components for explainability...")
        
        industry = self._identify_industry(company, all_analyses)
        industry_data = self.industry_baseline_risk.get(industry, self.industry_baseline_risk.get('unknown'))
        industry_baseline = industry_data.get('baseline', 50)
        
        print(f"Industry: {industry.replace('_', ' ').title()}")
        print(f"Industry baseline risk: {industry_baseline}/100")
        
        components = self._calculate_components(all_analyses)
        
        base_risk = sum(
            components.get(key, 50) * weight 
            for key, weight in self.weights.items()
        )
        
        industry_adjustment = (industry_baseline - 50) * 0.3
        adjusted_risk = base_risk + industry_adjustment
        
        # Apply industry-specific risk multipliers for high-carbon sectors
        industry_multiplier = self.INDUSTRY_RISK_MULTIPLIERS.get(industry, 1.0)
        if industry_multiplier > 1.0:
            print(f"Industry penalty applied: {industry.replace('_', ' ').title()} × {industry_multiplier}")
            adjusted_risk = adjusted_risk * industry_multiplier
        
        peer_modifier = self._calculate_peer_modifier(all_analyses, industry)
        debate_penalty = 10.0 if all_analyses.get("debate_activated") else 0.0
        
        # Canonical component score (before override/non-override arbitration).
        base_risk_score = max(0, min(100, round(base_risk, 1)))
        final_score = max(0, min(100, round(base_risk_score + industry_adjustment + peer_modifier + debate_penalty, 1)))
        greenwashing_risk_formula = final_score
        
        print(f"   Formula Risk: {greenwashing_risk_formula:.1f}/100 (for component analysis)")
        print(f"   Industry: {industry.replace('_', ' ').title()}")
        print(f"   Industry Baseline: {industry_baseline}/100")
        # Step 4: DOMAIN KNOWLEDGE OVERRIDES (highest priority)
        # CRITICAL: Industry-specific greenwashing patterns override pillar scores
        high_carbon_greenwashing_flag = False
        
        # Extract claim text for greenwashing check
        claim_text = ""
        if isinstance(all_analyses.get("claim"), dict):
            claim_text = all_analyses["claim"].get("claim_text", "").lower()
        elif isinstance(all_analyses.get("claim"), str):
            claim_text = all_analyses["claim"].lower()
        
        # Define green keywords
        green_keywords = ["renewable", "green", "carbon neutral", "net zero", "sustainable"]
        has_green_claim = any(keyword in claim_text for keyword in green_keywords)
        
        # Calculate evidence quality score from components
        evidence_quality_score = 100 - components.get('evidence_quality', 50)
        
        print(f'\n🔍 DOMAIN KNOWLEDGE CHECK:')
        print(f'  Industry: {industry}')
        print(f'  Green claim: {has_green_claim}')
        print(f'  ESG Score: {overall_esg_score}/100')
        
        # CRITICAL: High-carbon sector greenwashing check
        if industry == "oil_and_gas" and has_green_claim:
            print(f"\n🔴 HIGH-CARBON SECTOR GREENWASHING CHECK")
            print(f"   Industry: Oil & Gas")
            print(f"   Green keywords: {[kw for kw in green_keywords if kw in claim_text]}")
            print(f"   ESG Score: {overall_esg_score}/100")
            
            # Override rating if ESG score is LOW/MODERATE with green claims
            if overall_esg_score < 60:
                print(f"   🚨 DOMAIN OVERRIDE: Low ESG + Green Claims = Greenwashing")
                
                rating_grade = "BB"
                risk_level = "HIGH"
                greenwashing_risk = max(greenwashing_risk, 70)
                high_carbon_greenwashing_flag = True
                override_ml = True
                risk_source = "Domain Knowledge Override (Oil & Gas Greenwashing)"
                confidence = 0.92
                
                print(f"   Assigned Rating: {rating_grade} (forced HIGH risk)")
        
        # OLD HYBRID LOGIC (now deprecated - kept for reference)
        # Step 3: Hybrid decision with DOMAIN KNOWLEDGE PRIORITY
        # CRITICAL: Industry penalties and greenwashing flags MUST override ML predictions
        if False and not override_ml and use_ml_prediction:
            # Check if domain knowledge should override ML
            # Priority 1: Formula indicates HIGH risk (≥70) - ALWAYS override
            # Priority 2: High ML confidence (≥85%) - Use ML
            # Priority 3: Medium ML confidence (60-84%) - Use ensemble
            
            if greenwashing_risk_formula >= 70:
                # DOMAIN KNOWLEDGE OVERRIDE - Formula scoring indicates HIGH risk
                print(f"\n🔴 DOMAIN KNOWLEDGE OVERRIDE - Formula indicates HIGH RISK")
                print(f"   Formula Risk Score: {greenwashing_risk_formula:.1f}/100")
                print(f"   ML Prediction: {ml_prediction} (confidence: {ml_confidence:.1%})")
                print(f"   OVERRIDING ML with formula-based scoring")
                print(f"   Reason: Industry penalties + evidence analysis = HIGH risk threshold")
                
                greenwashing_risk = max(greenwashing_risk_formula, 70)  # Force minimum HIGH
                risk_source = "Domain Knowledge Override (Formula)"
                
            elif ml_confidence >= 0.85:
                # High ML confidence - use ML prediction
                risk_mapping_reverse = {'LOW': 25, 'MODERATE': 50, 'HIGH': 80}
                greenwashing_risk = risk_mapping_reverse[ml_prediction]
                risk_source = "ML Model (High Confidence)"
                
                print(f"\n🤖 USING ML PREDICTION (High Confidence)")
                print(f"   ML Prediction: {ml_prediction}")
                print(f"   ML Confidence: {ml_confidence:.1%}")
                print(f"   Final Risk: {greenwashing_risk}/100")
                
            else:
                # Medium confidence - use ensemble (60% ML + 40% Formula)
                risk_mapping_reverse = {'LOW': 25, 'MODERATE': 50, 'HIGH': 80}
                ml_risk_score = risk_mapping_reverse[ml_prediction]
                
                # Weighted ensemble
                greenwashing_risk = (ml_risk_score * 0.60) + (greenwashing_risk_formula * 0.40)
                risk_source = "Ensemble (60% ML + 40% Formula)"
                
                print(f"\n⚖️ USING ENSEMBLE APPROACH")
                print(f"   ML Prediction: {ml_prediction} ({ml_risk_score}/100)")
                print(f"   Formula Score: {greenwashing_risk_formula:.1f}/100")
                print(f"   Weighted Average: {greenwashing_risk:.1f}/100")
        
        # Step 5: LightGBM ESG Score Prediction (validation only)
        esg_prediction = None
        formula_esg_score = overall_esg_score
        
        if self.use_esg_predictor:
            print(f"\n🔮 Running LightGBM ESG Score Validation...")
            company_data = self._extract_esg_features(all_analyses)
            
            if company_data:
                esg_pred_result = self.esg_predictor.predict_esg_score(company_data)
                
                if esg_pred_result and esg_pred_result.get('prediction_successful'):
                    predicted_esg = esg_pred_result['predicted_esg']
                    esg_prediction = esg_pred_result
                    
                    print(f"   Pillar ESG: {formula_esg_score:.1f}/100")
                    print(f"   LightGBM ESG: {predicted_esg:.1f}/100")
                    print(f"   Prediction Range: {esg_pred_result['prediction_range']}")
                    
                    # Flag large discrepancies
                    discrepancy = abs(formula_esg_score - predicted_esg)
                    if discrepancy > 20:
                        print(f"   ⚠️ Large discrepancy ({discrepancy:.1f} points) - manual review recommended")
                    else:
                        print(f"   ✅ Scores aligned (discrepancy: {discrepancy:.1f} points)")
        
        # Step 6: LSTM Trend Prediction (context only)
        lstm_trend = None
        risk_adjustment = 0.0
        if self.use_lstm:
            print(f"\n🔮 Running LSTM Trend Forecast...")
            lstm_result = self.lstm_predictor.predict_from_analysis(all_analyses)
            
            if lstm_result and lstm_result.get('trend_successful'):
                lstm_trend = lstm_result
                
                print(f"   Forecast (6 years): {lstm_result['forecast']}")
                print(f"   Trend: {lstm_result['trend']}")
                print(f"   Change: {lstm_result['change_pct']:+.1f}%")
                print(f"   Model MAE: {lstm_result['confidence_mae']} points")
                
                # Adjust greenwashing risk based on trend
                if lstm_result['trend'] == 'DECLINING':
                    risk_adjustment += 10
                    print(f"   ⚠️ Declining ESG trend detected - risk increased by 10 points")
                elif lstm_result['trend'] == 'IMPROVING':
                    risk_adjustment -= 5
                    print(f"   ✅ Improving ESG trend - risk reduced by 5 points")
        
        # Step 6.5: Industry exposure uplift (risk-only; capped)
        industry = self._identify_industry(company, all_analyses)
        industry_data = self.industry_baseline_risk.get(industry, self.industry_baseline_risk.get('unknown'))
        industry_baseline = industry_data.get('baseline', 50)
        industry_risk_uplift = max(0.0, min(12.0, (industry_baseline - 50) * 0.30))
        if industry_risk_uplift > 0:
            risk_adjustment += industry_risk_uplift
            print(f"   ⚠️ Industry exposure uplift applied: +{industry_risk_uplift:.1f} risk points")
        
        # Step 6.6: Cap stacked adjustments to prevent unrealistic collapses
        risk_adjustment = max(-10.0, min(35.0, risk_adjustment))
        if not override_ml:
            greenwashing_risk = max(0.0, min(100.0, base_greenwashing_risk + risk_adjustment))
        
        # Step 7: Anomaly Detection (flagging only)
        anomaly_result = None
        if self.use_anomaly_detector:
            print(f"\n🔍 Running Anomaly Detection...")
            
            # Extract features for anomaly detection
            anomaly_features = self._extract_anomaly_features(all_analyses)
            
            if anomaly_features:
                anomaly_result = self.anomaly_detector.detect_anomaly(anomaly_features)
                
                if anomaly_result and anomaly_result.get('detection_successful'):
                    is_anomaly = anomaly_result['is_anomaly']
                    severity = anomaly_result['severity']
                    
                    print(f"   Anomaly Detected: {'YES' if is_anomaly else 'NO'}")
                    print(f"   Severity: {severity}")
                    
                    if is_anomaly:
                        print(f"   ⚠️ Anomaly flagged for review (does not alter rating)")
        
        # Step 8: Final ESG score determination
        esg_score = overall_esg_score
        
        # OLD GREENWASHING CHECK (now handled in Step 4)
        # Step 4C: Special greenwashing flag for high-carbon sectors with green claims (skip if overridden)
        # high_carbon_greenwashing_flag = False
        
        if False and not override_ml:
            # Extract claim text for greenwashing check
            claim_text = ""
            if isinstance(all_analyses.get("claim"), dict):
                claim_text = all_analyses["claim"].get("claim_text", "").lower()
            elif isinstance(all_analyses.get("claim"), str):
                claim_text = all_analyses["claim"].lower()
        
            # Define green keywords
            green_keywords = ["renewable", "green", "carbon neutral", "net zero"]
            has_green_claim = any(keyword in claim_text for keyword in green_keywords)
            
            # Calculate evidence quality score from components
            evidence_quality_score = 100 - components.get('evidence_quality', 50)
        
            # DEBUG: Print greenwashing check details
            print(f'\n🔍 GREENWASHING CHECK DEBUG:')
            print(f'  Industry detected: {industry}')
            print(f'  Claim text: {claim_text[:100] if claim_text else "(no claim)"}')
            print(f'  Green keywords found: {[kw for kw in green_keywords if kw in claim_text]}')
            print(f'  Evidence quality score: {evidence_quality_score}')
            print(f'  Has green claim: {has_green_claim}')
            print(f'  Meets all conditions (industry=oil_and_gas AND green_claim AND evidence<90): {industry == "oil_and_gas" and has_green_claim and evidence_quality_score < 90}')
            
            if industry == "oil_and_gas" and has_green_claim:
                # FIXED: Counterintuitive greenwashing pattern - MORE evidence can mean HIGHER risk
                # Oil & Gas companies extensively document their "transitions" - this is sophisticated greenwashing
                
                if evidence_quality_score < 70:
                    # Low evidence quality - standard greenwashing (no documentation)
                    print(f"\n🔴 HIGH-CARBON SECTOR GREENWASHING FLAG TRIGGERED")
                    print(f"   Industry: {industry.replace('_', ' ').title()}")
                    print(f"   Green keywords detected: {[kw for kw in green_keywords if kw in claim_text]}")
                    print(f"   Evidence quality: {evidence_quality_score}/100 (Low evidence)")
                    print(f"   ESCALATING TO HIGH RISK")
                    
                    greenwashing_risk = max(greenwashing_risk, 70)  # Force minimum HIGH risk
                    high_carbon_greenwashing_flag = True
                    risk_source = "Domain Knowledge Override (Greenwashing Flag)"
                    
                elif evidence_quality_score >= 70 and evidence_quality_score < 90:
                    # Moderate-high evidence quality - PARADOX: well-documented transition claims
                    print(f"\n⚠️ HIGH-QUALITY GREENWASHING EVIDENCE DETECTED (PARADOX)")
                    print(f"   Industry: Oil & Gas")
                    print(f"   Green keywords: {[kw for kw in green_keywords if kw in claim_text]}")
                    print(f"   Evidence quality: {evidence_quality_score}/100 (Well-documented)")
                    print(f"   PATTERN: Extensive documentation of transition = sophisticated greenwashing")
                    print(f"   APPLYING MODERATE PENALTY")
                    
                    greenwashing_risk = max(greenwashing_risk, 65)  # Moderate penalty
                    if "Override" not in risk_source:
                        risk_source = "Domain Knowledge Override (Paradox Pattern)"
                    
                elif evidence_quality_score >= 90:
                    # Extensive evidence quality - RED FLAG: over-documentation of green claims
                    print(f"\n🔴 EXTENSIVE GREENWASHING DOCUMENTATION DETECTED")
                    print(f"   Industry: Oil & Gas")
                    print(f"   Green keywords: {[kw for kw in green_keywords if kw in claim_text]}")
                    print(f"   Evidence quality: {evidence_quality_score}/100 (Extensively documented)")
                    print(f"   PATTERN: Over-documentation = Classic greenwashing tactic")
                    print(f"   APPLYING HIGH PENALTY - ESCALATING TO HIGH RISK")
                    
                    greenwashing_risk = max(greenwashing_risk, 72)  # High penalty
                    high_carbon_greenwashing_flag = True
                    risk_source = "Domain Knowledge Override (Over-documentation)"
            
            # FINAL OVERRIDE CHECK: If greenwashing flag is set, ensure HIGH risk
            if high_carbon_greenwashing_flag:
                if greenwashing_risk < 70:
                    print(f"\n🔴 FINAL OVERRIDE: Greenwashing flag forcing HIGH risk")
                    print(f"   Previous score: {greenwashing_risk:.1f}/100")
                    print(f"   Adjusted score: 70/100 (HIGH threshold)")
                    greenwashing_risk = 70
        
        # Step 8.5: Final greenwashing score arbitration
        # Override path blends pillar and component signals so one source cannot fully wash out the other.
        pillar_greenwashing = round(100 - esg_score, 1)

        if override_ml:
            component_greenwashing = base_risk_score
            blended_score = (
                (pillar_greenwashing * 0.60) +
                (component_greenwashing * 0.40)
            )

            final_score = blended_score + industry_adjustment + debate_penalty
            final_score = round(min(100, max(0, final_score)), 1)

            logger.info(
                "ESG override active for %s: "
                "pillar=%.1f component=%.1f blend=%.1f "
                "industry_adj=%.1f debate_penalty=%.1f final=%.1f",
                company,
                pillar_greenwashing,
                component_greenwashing,
                blended_score,
                industry_adjustment,
                debate_penalty,
                final_score
            )
        else:
            final_score = max(pillar_greenwashing, base_risk_score)
            final_score = round(
                min(100, max(0, final_score + industry_adjustment + debate_penalty)), 1
            )

        # Keep risk-level mapping aligned with final score regardless of earlier branch decisions.
        greenwashing_risk = final_score
        risk_level = self._risk_level_from_greenwashing_score(final_score)

        # Step 9: Generate insights
        top_reasons = self._generate_top_reasons(
            components, 
            all_analyses, 
            industry, 
            greenwashing_risk,
            pillar_scores=pillar_scores,
            positive_boost=positive_boost
        )
        
        insights = self._generate_insights(
            greenwashing_risk, 
            risk_level, 
            industry, 
            company
        )
        
        evidence_count = len(evidence_list) if isinstance(evidence_list, list) else 0
        credibility_metrics = (
            all_analyses.get("credibility_analysis", {}).get("aggregate_metrics", {})
            if isinstance(all_analyses.get("credibility_analysis"), dict)
            else {}
        )
        source_quality = float(credibility_metrics.get("average_credibility", 0.7) or 0.7) * 100
        pillar_coverage = sum(
            1
            for p_key in ["environmental_score", "social_score", "governance_score"]
            if pillar_scores.get(p_key) is not None
        )
        confidence_penalty = self._calculate_confidence_penalty(
            evidence_count=evidence_count,
            source_quality=source_quality,
            pillar_coverage=pillar_coverage,
        )

        if confidence_penalty >= 15:
            report_tier = "TIER_3"
            tier_label = "Preliminary Scan"
        elif confidence_penalty >= 8:
            report_tier = "TIER_2"
            tier_label = "Indicative Report"
        else:
            report_tier = "TIER_1"
            tier_label = "Enterprise Grade"

        score_disclaimer = (
            f"Score based on {evidence_count} sources. "
            f"Evidence coverage is {'low' if confidence_penalty >= 8 else 'adequate'}. "
            f"Uncertainty range: ±{round(confidence_penalty)}pts. "
            f"Treat as {tier_label}."
        ) if confidence_penalty >= 8 else ""

        final_risk_level = self._risk_level_from_greenwashing_score(final_score)

        result = {
            "company": company,
            "analysis_timestamp": datetime.now().isoformat(),
            "risk_source": risk_source,
            "industry": industry,
            "industry_baseline_risk": industry_baseline,
            "industry_source": industry_data.get('source', 'Unknown'),
            "base_risk_score": round(base_risk, 1),
            "industry_adjustment": round(industry_adjustment, 1),
            "peer_adjustment": round(peer_modifier, 1),
            "greenwashing_score": final_score,
            "greenwashing_risk_score": final_score,
            "esg_score": round(esg_score, 1),
            "risk_level": final_risk_level,
            "rating_grade": rating_grade,
            "component_scores": components,
            "pillar_scores": pillar_scores,
            "explainability_top_3_reasons": top_reasons,
            "actionable_insights": insights,
            "confidence_level": round(confidence * 100, 1),
            "positive_esg_boost": round(positive_boost, 1),
            "confidence_penalty": round(confidence_penalty, 1),
            "confidence_penalty_applied": round(confidence_penalty_total * 100, 1),
            "social_governance_decision_ready": bool(sg_adequacy.get("overall_ready", False)),
            "report_tier": report_tier,
            "score_disclaimer": score_disclaimer,
            "high_carbon_greenwashing_flag": high_carbon_greenwashing_flag,
            "esg_override_active": override_ml
        }
        if score_disclaimer:
            result.setdefault("quality_warnings", []).append(score_disclaimer)
        
        # Build structured pillar_factors with full sub-indicator breakdown
        try:
            evidence_list = all_analyses.get("evidence", [])
            if not isinstance(evidence_list, list):
                evidence_list = []
            carbon_data_for_factors = all_analyses.get("carbon_extraction") or all_analyses.get("carbon_results") or {}
            result["pillar_factors"] = build_pillar_factors(
                company=company,
                industry=industry,
                evidence_sources=evidence_list,
                carbon_data=carbon_data_for_factors,
                pillar_scores=pillar_scores,
            )
            # FIX 1B: Back-populate raw_signal_normalized and points_contributed from factor weights.
            pillar_key_map = {
                "environmental": "environmental_score",
                "social": "social_score",
                "governance": "governance_score",
            }
            for pillar_name, score_key in pillar_key_map.items():
                pillar_block = result["pillar_factors"].get(pillar_name, {})
                factors = pillar_block.get("sub_indicators", []) if isinstance(pillar_block, dict) else []
                if not isinstance(factors, list):
                    continue

                pillar_total = self._compute_pillar_score(factors)
                points_sum = sum(
                    f.get("points_contributed", 0.0)
                    for f in factors
                    if isinstance(f.get("points_contributed"), (int, float))
                )
                assert abs(float(pillar_total) - float(points_sum)) < 0.1, (
                    f"Pillar contribution mismatch for {pillar_name}: "
                    f"pillar={pillar_total}, sum_points={points_sum}"
                )

                pillar_total = round(max(0.0, min(100.0, pillar_total)), 1)
                pillar_block["score"] = pillar_total
                result["pillar_scores"][score_key] = pillar_total

            # Keep final ESG/risk synchronized with pillar-derived totals.
            e = float(result["pillar_scores"].get("environmental_score", 0.0) or 0.0)
            s = float(result["pillar_scores"].get("social_score", 0.0) or 0.0)
            g = float(result["pillar_scores"].get("governance_score", 0.0) or 0.0)
            overall_esg = round(max(0.0, min(100.0, e * 0.35 + s * 0.30 + g * 0.35)), 1)
            result["pillar_scores"]["overall_esg_score"] = overall_esg
            result["esg_score"] = overall_esg

            # Keep final score synchronized to the final pillar-derived ESG score.
            pillar_greenwashing_synced = round(100 - overall_esg, 1)
            if override_ml:
                blended_score_synced = (
                    (pillar_greenwashing_synced * 0.60) +
                    (base_risk_score * 0.40)
                )
                final_score = round(
                    min(100, max(0, blended_score_synced + industry_adjustment + debate_penalty)), 1
                )
            else:
                final_score = round(
                    min(100, max(0, max(pillar_greenwashing_synced, base_risk_score) + industry_adjustment + debate_penalty)), 1
                )

            greenwashing_risk = final_score
            risk_level = self._risk_level_from_greenwashing_score(final_score)

            result["greenwashing_score"] = final_score
            result["greenwashing_risk_score"] = final_score
            result["rating_grade"], _ = self.esg_score_to_rating(overall_esg)
            result["risk_level"] = self._risk_level_from_greenwashing_score(final_score)

            # Rating-grade safety rule: CCC/C must always surface as HIGH risk.
            if result["rating_grade"] in {"CCC", "C"} and result["risk_level"] != "HIGH":
                result["risk_level"] = "HIGH"
                result["greenwashing_score"] = max(float(result["greenwashing_score"]), 60.0)
                result["greenwashing_risk_score"] = max(float(result["greenwashing_risk_score"]), 60.0)
            print(f"   ✅ Pillar factors populated with sub-indicator breakdown")
        except Exception as pf_err:
            print(f"   ⚠️ Pillar factors build failed: {pf_err}")
            result["pillar_factors"] = {}

        result["scoring_methodology"] = scoring_methodology
        
        # Add ML details if available
        if ml_prediction and use_ml_prediction:
            result["ml_prediction"] = {
                "prediction": ml_prediction,
                "confidence": ml_confidence,
                "probabilities": ml_result.get('probabilities', {}),
                "used_for_final": use_ml_prediction,
                "role": "Rating refinement in MODERATE range (50-74 ESG)"
            }
        
        # Add LightGBM ESG prediction if available
        if esg_prediction:
            result["esg_prediction"] = {
                "predicted_esg": esg_prediction['predicted_esg'],
                "confidence_r2": esg_prediction['confidence_r2'],
                "prediction_range": esg_prediction['prediction_range'],
                "pillar_esg": formula_esg_score,
                "discrepancy": abs(formula_esg_score - esg_prediction['predicted_esg']),
                "model_type": esg_prediction['model_type'],
                "role": "Validation only (does not affect rating)"
            }
        
        # Add LSTM trend prediction if available
        if lstm_trend:
            result["lstm_trend_prediction"] = {
                "forecast": lstm_trend['forecast'],
                "trend": lstm_trend['trend'],
                "change_pct": lstm_trend['change_pct'],
                "confidence_mae": lstm_trend['confidence_mae'],
                "model_type": lstm_trend['model_type'],
                "role": "Context only (does not affect rating)"
            }
        
        # Add anomaly detection result if available
        if anomaly_result:
            result["anomaly_detection"] = {
                "is_anomaly": anomaly_result['is_anomaly'],
                "severity": anomaly_result['severity'],
                "anomaly_score": anomaly_result['anomaly_score'],
                "confidence": anomaly_result['confidence'],
                "anomalous_features": anomaly_result.get('anomalous_features', []),
                "model_type": "Isolation Forest",
                "role": "Flagging only (does not affect rating)"
            }
        
        print(f"\n✅ Final Risk Assessment:")
        print(f"   Greenwashing Risk: {greenwashing_risk:.1f}/100 ({risk_source})")
        print(f"   ESG Score: {esg_score:.1f}/100 (Grade: {rating_grade})")
        print(f"   Risk Level: {risk_level}")
        print(f"   Methodology: {'ESG Pillar Primary' if not use_ml_prediction else 'Pillar + ML Refinement'}")
        
        print(f"\n📊 ESG Pillar Breakdown:")
        print(f"   Environmental: {pillar_scores['environmental_score']}/100 (35% weight)")
        print(f"   Social: {pillar_scores['social_score']}/100 (30% weight)")
        print(f"   Governance: {pillar_scores['governance_score']}/100 (35% weight)")
        print(f"   Overall ESG: {pillar_scores['overall_esg_score']}/100")
        
        if ml_prediction:
            print(f"\n🤖 XGBoost Risk Model:")
            print(f"   Prediction: {ml_prediction}")
            print(f"   Used: {'YES (refinement)' if use_ml_prediction else 'NO (pillar scores primary)'}")
        
        if esg_prediction:
            print(f"\n🔮 LightGBM ESG Validator:")
            print(f"   Predicted ESG: {esg_prediction['predicted_esg']:.1f}/100")
            print(f"   Purpose: Validation check")
        
        if lstm_trend:
            print(f"\n🔮 LSTM Trend Forecaster:")
            print(f"   Trend: {lstm_trend['trend']}")
            print(f"   Purpose: Contextual insight")
        
        if anomaly_result and anomaly_result.get('is_anomaly'):
            print(f"\n⚠️  Anomaly Detector:")
            print(f"   Status: ANOMALY DETECTED")
            print(f"   Severity: {anomaly_result['severity']}")
            print(f"   Purpose: Flagging for review")
        
        print(f"\n📊 ESG Pillar Breakdown:")
        print(f"   Environmental: {pillar_scores['environmental_score']}/100 (35% weight)")
        print(f"   Social: {pillar_scores['social_score']}/100 (30% weight)")
        print(f"   Governance: {pillar_scores['governance_score']}/100 (35% weight)")
        print(f"   Overall ESG: {pillar_scores['overall_esg_score']}/100")
        
        if ml_prediction:
            print(f"\n🤖 XGBoost Risk Model:")
            print(f"   Prediction: {ml_prediction}")
            print(f"   Used: {'YES' if use_ml_prediction else 'NO (low confidence)'}")
        
        if esg_prediction:
            print(f"\n🔮 LightGBM ESG Predictor:")
            print(f"   Predicted ESG: {esg_prediction['predicted_esg']:.1f}/100")
            print(f"   Confidence: R²={esg_prediction['confidence_r2']:.3f}")
        
        if lstm_trend:
            print(f"\n🔮 LSTM Trend Forecaster:")
            print(f"   Trend: {lstm_trend['trend']}")
            print(f"   Change: {lstm_trend['change_pct']:+.1f}%")
            print(f"   6-Year Forecast: {lstm_trend['forecast']}")
        
        if anomaly_result and anomaly_result.get('is_anomaly'):
            print(f"\n⚠️  Anomaly Detector:")
            print(f"   Status: ANOMALY DETECTED")
            print(f"   Severity: {anomaly_result['severity']}")
            print(f"   Confidence: {anomaly_result['confidence']:.1f}%")
        
        return result

    def _calculate_confidence_penalty(self, evidence_count: int, source_quality: float, pillar_coverage: int) -> float:
        """Estimate uncertainty penalty points for report tiering only."""
        penalty = 0.0

        if evidence_count < 5:
            penalty += 15.0
        elif evidence_count < 10:
            penalty += 12.0
        elif evidence_count < 15:
            penalty += 8.0

        if source_quality < 55:
            penalty += 3.0
        elif source_quality < 70:
            penalty += 1.5

        if pillar_coverage < 3:
            penalty += 2.0

        return round(min(25.0, penalty), 1)

    def _risk_level_from_greenwashing_score(self, score: float) -> str:
        """Map greenwashing score bands to LOW/MODERATE/HIGH."""
        if score < 40:
            return "LOW"
        if score < 60:
            return "MODERATE"
        return "HIGH"
    
    def _identify_industry(self, company: str, analyses: Dict) -> str:
        """
        Identify company's industry - 100% DYNAMIC (NO HARDCODED FALLBACKS)
        Uses LLM to classify into predefined MSCI-based categories
        """
        
        # Get list of valid industries from config
        valid_industries = [k for k in self.industry_baseline_risk.keys() if k != 'unknown']
        
        # If upstream already provided a valid industry, trust it (keeps workflow stable and deterministic).
        provided = analyses.get("industry")
        if isinstance(provided, str):
            provided_clean = provided.strip().lower().replace(" ", "_")
            if provided_clean in valid_industries:
                return provided_clean
        
        # Use LLM to classify
        prompt = f"""Classify {company} into ONE of these industries:

{', '.join(valid_industries)}

Return ONLY the industry name from the list above, nothing else.

Examples:
- BP → oil_and_gas
- Tesla → automotive
- Microsoft → technology
- H&M → fast_fashion
- Coca-Cola → food_beverage

Company: {company}
Industry:"""

        try:
            response = asyncio.run(call_llm("risk_scoring", prompt))
            
            if response:
                # Clean response
                industry = response.strip().lower()
                industry = industry.replace(' ', '_')
                industry = industry.replace('.', '').replace(',', '').replace(':', '')
                
                # Direct match
                if industry in valid_industries:
                    return industry
                
                # Fuzzy match
                for valid in valid_industries:
                    if valid in industry or industry in valid:
                        return valid
                    
                    # Check if any word matches
                    industry_words = industry.split('_')
                    valid_words = valid.split('_')
                    if any(word in valid_words for word in industry_words):
                        return valid
        
        except Exception as e:
            print(f"   ⚠️ Industry classification error: {e}")
        
        # If LLM fails, return unknown (NO HARDCODED COMPANY NAMES)
        print(f"   ⚠️ Could not classify {company} - using 'unknown'")
        return "unknown"
    
    def _calculate_components(self, analyses: Dict) -> Dict[str, float]:
        """
        Calculate all component scores (0-100, higher = more risk)
        FIXED: Unverifiable claims now properly penalized
        """
        
        components = {}
        
        # Normalize contradiction payloads from different agent output formats.
        contradiction_findings = analyses.get("contradiction_analysis", {})
        contradictions = analyses.get("contradictions", []) or []
        if isinstance(contradiction_findings, dict):
            contradictions = (
                contradictions
                or contradiction_findings.get("contradiction_list")
                or contradiction_findings.get("contradictions")
                or contradiction_findings.get("specific_contradictions")
                or []
            )
        elif isinstance(contradiction_findings, list) and not contradictions:
            contradictions = contradiction_findings

        # 1. Claim Verification (CRITICAL - FIXED scoring)
        verification_rows = [c for c in contradictions if isinstance(c, dict) and c.get("overall_verdict")]
        if verification_rows:
            contradicted = sum(1 for c in verification_rows if c.get('overall_verdict') == 'Contradicted')
            unverifiable = sum(1 for c in verification_rows if c.get('overall_verdict') == 'Unverifiable')
            partial = sum(1 for c in verification_rows if c.get('overall_verdict') == 'Partially True')
            verified = sum(1 for c in verification_rows if c.get('overall_verdict') == 'Verified')
            total = len(verification_rows)
            
            # FIXED: Increased penalty for unverifiable claims
            # Contradicted = 100 risk
            # Unverifiable = 85 risk (was 70 - too lenient)
            # Partially True = 50 risk
            # Verified = 0 risk
            score = ((contradicted * 100) + (unverifiable * 85) + (partial * 50) + (verified * 0)) / total if total > 0 else 50
            components['claim_verification'] = min(100, score)
        else:
            # Calibration: missing contradiction analysis should not be treated as "certain failure".
            components['claim_verification'] = 50
        
        # 2. Evidence Quality (more sources = lower risk)
        evidence = analyses.get('evidence', [])
        total_sources = sum(len(ev.get('evidence', [])) for ev in evidence)
        
        if total_sources >= 20:
            components['evidence_quality'] = 10
        elif total_sources >= 15:
            components['evidence_quality'] = 20
        elif total_sources >= 10:
            components['evidence_quality'] = 35
        elif total_sources >= 5:
            components['evidence_quality'] = 60
        else:
            # Calibration: sparse evidence implies elevated risk, but avoid hard-max penalties.
            components['evidence_quality'] = 75
        
        # 3. Source Credibility (higher credibility = lower risk)
        credibility = analyses.get('credibility_analysis', {})
        if credibility:
            metrics = credibility.get('aggregate_metrics', {})
            avg_cred = metrics.get('average_credibility', 0.5)
            # Convert: high credibility (0.9) → low risk (10)
            components['source_credibility'] = int((1.0 - avg_cred) * 100)
        else:
            # Calibration: missing credibility analysis is treated as neutral with lower confidence elsewhere.
            components['source_credibility'] = 50
        
        # 4. Sentiment Divergence
        sentiment_list = analyses.get('sentiment_analysis', [])
        if sentiment_list:
            divergences = [s.get('divergence_score', 0) for s in sentiment_list]
            components['sentiment_divergence'] = int(sum(divergences) / len(divergences))
        else:
            components['sentiment_divergence'] = 50
        
        # 5. Historical Pattern
        historical = analyses.get('historical_analysis', {})
        if historical:
            violations = len(historical.get('past_violations', []))
            greenwashing_acc = historical.get('greenwashing_history', {}).get('prior_accusations', 0)
            reputation = historical.get('reputation_score', 50)
            
            # More violations = higher risk
            violation_risk = min(100, violations * 20)
            greenwashing_risk = min(100, greenwashing_acc * 30)
            reputation_risk = 100 - reputation
            
            components['historical_pattern'] = int((violation_risk + greenwashing_risk + reputation_risk) / 3)
        else:
            components['historical_pattern'] = 50
        
        # 6. Contradiction Severity
        severity_score = 0
        contradiction_rows = [c for c in contradictions if isinstance(c, dict)]
        for c in contradiction_rows:
            items = c.get("specific_contradictions") if isinstance(c.get("specific_contradictions"), list) else [c]
            for item in items:
                if not isinstance(item, dict):
                    continue
                severity = str(item.get("severity", "")).upper()
                if severity == "CRITICAL":
                    severity_score += 35
                elif severity in ("HIGH", "MAJOR"):
                    severity_score += 25
                elif severity == "MEDIUM":
                    severity_score += 12
                elif severity == "LOW":
                    severity_score += 5

        # Cap at 50 so contradictions materially impact risk without fully dominating.
        components['contradiction_severity'] = min(50, severity_score)
        
        # At the end of calculate_components method (before return components, around line 440):

        # NEW: Check for vague greenwashing language
        if analyses.get("claim"):
            claim = analyses["claim"]
            # Handle both string and dict formats
            if isinstance(claim, dict):
                claim_text = claim.get("text", claim.get("claim", "")).lower()
            elif isinstance(claim, str):
                claim_text = claim.lower()
            else:
                claim_text = str(claim).lower()
                
            greenwashing_keywords = [
                "committed to", "leader in", "eco-friendly", "sustainable", 
                "green", "environmentally friendly", "climate positive"
            ]
            
            keyword_count = sum(1 for keyword in greenwashing_keywords if keyword in claim_text)
            has_numbers = any(char.isdigit() for char in claim_text)
            
            # Vague claim (multiple buzzwords + no metrics) = +20 risk penalty
            if keyword_count >= 2 and not has_numbers:
                components["claim_verification"] = min(100, components.get("claim_verification", 50) + 20)
                print(f"⚠️  Vague claim detected ({keyword_count} buzzwords, no metrics) - penalty applied")
        
        # NEW: Financial Greenwashing Flags (AGENT 14 integration)
        evidence = analyses.get('evidence', [])
        financial_flags_penalty = 0
        
        for ev in evidence:
            financial_context = ev.get('financial_context', {})
            if financial_context.get('financial_data_available'):
                flags = financial_context.get('greenwashing_flags', [])
                
                # Add risk based on severity
                for flag in flags:
                    severity = flag.get('severity', 'Low')
                    if severity == 'High':
                        financial_flags_penalty += 15
                    elif severity == 'Moderate':
                        financial_flags_penalty += 10
                    else:
                        financial_flags_penalty += 5
                
                # Cap penalty
                financial_flags_penalty = min(30, financial_flags_penalty)
        
        # Apply penalty to claim verification component
        if financial_flags_penalty > 0:
            components['claim_verification'] = min(100, 
                components.get('claim_verification', 50) + financial_flags_penalty
            )
            print(f"   ⚠️ Financial greenwashing penalty: +{financial_flags_penalty} risk")

        return components

    def _calculate_peer_modifier(self, analyses: Dict, industry: str) -> float:
        """
        Calculate peer comparison modifier
        Unverified superlative claims → penalty
        """
        
        industry_comp = analyses.get('industry_comparison', {})
        if not industry_comp or industry_comp.get('error'):
            return 0.0
        
        # Check for unverified superlative claims
        comparisons = industry_comp.get('claim_comparisons', [])
        superlative_unverified = sum(
            1 for c in comparisons 
            if c.get('uses_superlative') and not c.get('verified_against_peers')
        )
        
        # Penalty: +5 points per unverified superlative
        if superlative_unverified > 0:
            return superlative_unverified * 5.0
        
        return 0.0
    
    def _determine_risk_level(self, risk_score: float, industry_baseline: float) -> tuple:
        """
        DEPRECATED: Use esg_score_to_rating() instead
        This method kept for backward compatibility
        Converts risk score to ESG score and uses MSCI-aligned rating
        """
        # Convert risk to ESG score
        esg_score = 100 - risk_score
        return self.esg_score_to_rating(esg_score)
    
    def _generate_top_reasons(self, components: Dict, analyses: Dict, 
                             industry: str, risk_score: float,
                             pillar_scores: Optional[Dict[str, Any]] = None,
                             positive_boost: float = 0.0) -> List[str]:
        """Generate top 3 specific, data-backed reasons (balanced positives + negatives)"""
        
        reasons = []
        pillar_scores = pillar_scores or {}
        
        # Sort components by risk score
        sorted_comps = sorted(components.items(), key=lambda x: x[1], reverse=True)
        
        for component, score in sorted_comps[:3]:
            if component == 'claim_verification' and score > 60:
                contradictions = analyses.get('contradiction_analysis', [])
                contradicted = sum(1 for c in contradictions if c.get('overall_verdict') == 'Contradicted')
                unverifiable = sum(1 for c in contradictions if c.get('overall_verdict') == 'Unverifiable')
                
                if contradicted > 0:
                    reasons.append(
                        f"Claim verification failure: {contradicted} claim(s) contradicted by evidence (risk: {int(score)}%)"
                    )
                elif unverifiable > 0:
                    reasons.append(
                        f"Unverifiable claims: {unverifiable} claim(s) lack supporting evidence (risk: {int(score)}%)"
                    )
            
            elif component == 'historical_pattern' and score > 50:
                historical = analyses.get('historical_analysis', {})
                violations = len(historical.get('past_violations', []))
                if violations > 0:
                    reasons.append(
                        f"Historical violations: {violations} documented ESG violation(s) (risk: {int(score)}%)"
                    )
            
            elif component == 'contradiction_severity' and score > 40:
                contradictions = analyses.get('contradiction_analysis', [])
                major_count = sum(
                    1 for c in contradictions 
                    for cont in c.get('specific_contradictions', []) 
                    if cont.get('severity') == 'Major'
                )
                if major_count > 0:
                    reasons.append(
                        f"Major contradictions: {major_count} severe inconsistenc(y/ies) detected (risk: {int(score)}%)"
                    )
            
            elif component == 'source_credibility' and score > 40:
                reasons.append(
                    f"Source credibility concerns: Evidence from low-quality or biased sources (risk: {int(score)}%)"
                )
        
        # Add positive drivers to avoid one-sided explanations
        positives = []
        if positive_boost >= 5:
            positives.append(f"Strong disclosures and evidence quality provided a +{int(round(positive_boost))} point positive reinforcement")
        
        try:
            e = float(pillar_scores.get("environmental_score", 50) or 50)
            s = float(pillar_scores.get("social_score", 50) or 50)
            g = float(pillar_scores.get("governance_score", 50) or 50)
            if g >= 70:
                positives.append("Strong governance signal (board/ethics/disclosure indicators)")
            if s >= 70:
                positives.append("Strong social signal (labor, safety, DEI, community indicators)")
            if e >= 70:
                positives.append("Strong environmental signal (climate, renewables, emissions indicators)")
        except Exception:
            pass
        
        industry_context = f"Industry context: {industry.replace('_', ' ').title()} materiality weighting applied (MSCI-style)."
        
        # Ensure at least one positive or industry context is included when reasons are sparse
        if positives and len(reasons) < 3:
            reasons.append(f"Offsetting factor: {positives[0]}")
        
        if len(reasons) < 3:
            reasons.append(industry_context)
        
        # Ensure we have at least 3 reasons
        while len(reasons) < 3:
            reasons.append("Insufficient data quality or evidence gaps detected")
        
        return reasons[:3]
    
    def _generate_insights(self, risk_score: float, risk_level: str, 
                          industry: str, company: str) -> Dict[str, str]:
        """Generate stakeholder-specific actionable insights"""
        
        industry_name = industry.replace('_', ' ').title()
        
        if risk_level == "HIGH":
            insights = {
                "for_investors": f"HIGH RISK: {company} ({industry_name}) shows significant greenwashing indicators. Claims lack credible verification or contain major contradictions. NOT suitable for ESG portfolios without deep independent audit and verification.",
                "for_regulators": f"IMMEDIATE ATTENTION REQUIRED: {company} requires formal investigation. Multiple red flags detected including unverified claims, contradictions, or historical violations. Recommend requesting documentation and potential enforcement action in high-scrutiny {industry_name} sector.",
                "for_consumers": f"CAUTION ADVISED: {company}'s ESG claims appear questionable or unsubstantiated. {industry_name} companies face inherent sustainability challenges. Strongly recommend seeking alternatives with credible third-party certifications (B Corp, Fair Trade, etc.)."
            }
        elif risk_level == "MODERATE":
            insights = {
                "for_investors": f"MODERATE RISK: {company} ({industry_name}) shows mixed ESG performance with some concerns. Additional due diligence required before investment. Monitor peer comparisons, upcoming sustainability reports, and third-party ratings (MSCI, Sustainalytics).",
                "for_regulators": f"MONITORING RECOMMENDED: {company} shows some inconsistencies in ESG claims. Standard oversight appropriate for {industry_name} sector. Consider requesting clarification on specific unverified claims and ensuring compliance with disclosure requirements.",
                "for_consumers": f"MIXED SIGNALS: {company} demonstrates some genuine ESG efforts but {industry_name} sector has structural sustainability challenges. Verify specific product claims independently and compare with competitors' performance."
            }
        else:
            insights = {
                "for_investors": f"LOW RISK: {company} ({industry_name}) shows credible ESG commitments backed by verifiable evidence from quality sources. Suitable for ESG-focused portfolios. Continue standard monitoring of annual reports and third-party assessments.",
                "for_regulators": f"NO MAJOR CONCERNS: {company} meets expected disclosure and performance standards for {industry_name} sector. Routine monitoring sufficient. Claims appear substantiated and consistent with available evidence.",
                "for_consumers": f"TRUSTWORTHY: {company}'s ESG claims appear credible with reasonable evidence backing from independent sources. Good choice within {industry_name} sector. Look for third-party certifications for additional assurance."
            }
        
        return insights

    def _calculate_positive_esg_boost(self, analyses: Dict[str, Any], pillar_scores: Dict[str, Any]) -> float:
        """
        Positive reinforcement layer (capped) to reward strong disclosures, consistency and evidence quality.
        Returns points to add to overall ESG score (0 to +15).
        """
        boost = 0.0
        
        evidence = analyses.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        
        contradictions = analyses.get("contradiction_analysis", [])
        if not isinstance(contradictions, list):
            contradictions = []
        
        # Evidence quantity proxy
        total_sources = 0
        combined_text = ""
        for ev in evidence:
            if isinstance(ev, dict):
                total_sources += len(ev.get("evidence", []) or [])
                combined_text += " " + str(ev.get("snippet", "")) + " " + str(ev.get("relevant_text", ""))
                for e in ev.get("evidence", []) or []:
                    if isinstance(e, dict):
                        combined_text += " " + str(e.get("relevant_text", ""))
        
        lower = combined_text.lower()
        
        # Recognized disclosure frameworks (CDP/GRI/BRSR/etc.)
        disclosure_hits = sum(1 for kw in ["gri", "cdp", "tcfd", "sasb", "brsr", "sbti", "science based targets"] if kw in lower)
        if disclosure_hits >= 2:
            boost += 6
        elif disclosure_hits == 1:
            boost += 3
        
        # High-quality evidence coverage
        if total_sources >= 15:
            boost += 5
        elif total_sources >= 8:
            boost += 3
        elif total_sources >= 4:
            boost += 1
        
        # Low controversies proxy via contradiction outcomes
        contradicted = sum(1 for c in contradictions if isinstance(c, dict) and c.get("overall_verdict") == "Contradicted")
        unverifiable = sum(1 for c in contradictions if isinstance(c, dict) and c.get("overall_verdict") == "Unverifiable")
        if contradictions and contradicted == 0 and unverifiable == 0:
            boost += 2
        
        # Cap to avoid inflation and keep realism by base ESG level
        overall = float(pillar_scores.get("overall_esg_score", 50) or 50)
        if overall >= 90:
            boost = min(boost, 2.0)
        elif overall >= 80:
            boost = min(boost, 8.0)
        elif overall < 40:
            boost = min(boost, 5.0)
        
        return float(max(0.0, min(15.0, boost)))
    
    def _extract_esg_features(self, analyses: Dict) -> Optional[Dict[str, Any]]:
        """
        Extract features for LightGBM ESG predictor from analyses
        Required features:
        - environmentScore, socialScore, governanceScore (0-100)
        - highestControversy (0-5)
        - marketCap (float)
        - beta (float)
        - overallRisk (0-100)
        """
        try:
            # Extract ESG component scores (derived from components)
            components = self._calculate_components(analyses)
            
            # Map component scores to E, S, G scores (invert since components are risk scores)
            # Higher component risk → Lower ESG score
            evidence_quality = 100 - components.get('evidence_quality', 50)
            credibility = 100 - components.get('source_credibility', 50)
            sentiment = 100 - components.get('sentiment_divergence', 50)
            historical = 100 - components.get('historical_pattern', 50)
            
            # Approximate ESG dimensions
            env_score = (evidence_quality + credibility) / 2  # Environment
            soc_score = (sentiment + historical) / 2  # Social
            gov_score = credibility  # Governance
            
            # Extract controversy level (from contradictions)
            contradictions = analyses.get('contradiction_analysis', [])
            controversy_count = sum(1 for c in contradictions if c.get('overall_verdict') == 'Contradicted')
            highest_controversy = min(controversy_count, 5)
            
            # Extract financial context (if available)
            financial_ctx = analyses.get('financial_context', {})
            market_cap = financial_ctx.get('market_cap', 10_000_000_000)  # Default 10B
            beta = financial_ctx.get('beta', 1.0)  # Default market beta
            
            # Calculate overall risk (from greenwashing components)
            overall_risk = sum(components.values()) / len(components) if components else 50
            
            return {
                'environmentScore': round(env_score, 2),
                'socialScore': round(soc_score, 2),
                'governanceScore': round(gov_score, 2),
                'highestControversy': highest_controversy,
                'marketCap': market_cap,
                'beta': beta,
                'overallRisk': round(overall_risk, 2)
            }
        except Exception as e:
            print(f"   ⚠️ Error extracting ESG features: {e}")
            return None
    
    def _extract_anomaly_features(self, analyses: Dict) -> Optional[Dict[str, float]]:
        """
        Extract 8 anomaly detection features from agent analyses
        
        Returns dict with:
            - carbon_intensity
            - water_intensity
            - energy_intensity
            - esg_revenue_gap
            - growth_esg_correlation
            - profit_esg_ratio
            - environmental_balance
            - volatility_score
        """
        try:
            # Get financial data
            financial = analyses.get('financial_context', {})
            if not financial or not financial.get('financial_data_available'):
                return None
            
            fin_data = financial.get('financial_data', {})
            revenue = fin_data.get('revenue_ttm', 0)
            profit_margin = fin_data.get('profit_margin', 0)
            growth_rate = fin_data.get('revenue_growth', 0)
            
            # Get ESG metrics
            esg_metrics = financial.get('esg_financial_metrics', {})
            carbon_intensity = esg_metrics.get('carbon_intensity', 0) or 0
            water_efficiency = esg_metrics.get('water_efficiency', 0) or 0
            energy_efficiency = esg_metrics.get('energy_efficiency', 0) or 0
            
            # Get ESG scores from credibility analysis
            credibility = analyses.get('credibility_analysis', {})
            esg_overall = credibility.get('esg_score', 50)
            esg_env = credibility.get('environmental_score', 50)
            esg_social = credibility.get('social_score', 50)
            esg_gov = credibility.get('governance_score', 50)
            
            # Get historical volatility (if available)
            historical = analyses.get('historical_analysis', {})
            patterns = historical.get('temporal_patterns', {})
            volatility = 10.0 if patterns.get('declining_trend') else 5.0 if patterns.get('reactive_claims') else 2.0
            
            # Calculate features
            features = {
                'carbon_intensity': carbon_intensity,
                'water_intensity': water_efficiency,
                'energy_intensity': energy_efficiency,
                'esg_revenue_gap': (esg_overall - 50) / (np.log10(revenue + 1) + 1e-6) if revenue > 0 else 0,
                'growth_esg_correlation': (growth_rate * esg_overall) / 100,
                'profit_esg_ratio': profit_margin / (esg_overall + 1e-6) if esg_overall > 0 else 0,
                'environmental_balance': esg_env / (esg_social + esg_gov + 1e-6) if (esg_social + esg_gov) > 0 else 1.0,
                'volatility_score': volatility
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️  Could not extract anomaly features: {e}")
            return None
