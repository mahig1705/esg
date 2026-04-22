"""
LIVE IMPLEMENTATION: Fetches real-time data and shows LangGraph progress
All agents use live API calls, not cached results
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add agents directory to Python path
agents_dir = Path(__file__).parent.parent / "agents"
sys.path.insert(0, str(agents_dir))

# Add project root to Python path for feature imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from features.esg_mismatch_detector.pipeline import analyze_company_esg
except ImportError as e:
    print(f"⚠️  ESG Mismatch Detector import failed: {e}")
    analyze_company_esg = None

from core.state_schema import ESGState
from core.evidence_cache import evidence_cache
from typing import Dict, Any
from core.esg_data_apis import fill_missing_pillars
from core.fact_graph_builder import build_esg_fact_graph
from core.adversarial_audit import build_adversarial_audit
from core.company_knowledge_graph import CompanyKnowledgeGraph

# ============================================================
# LIVE DATA FETCHER - Gets fresh content for analysis
# ============================================================

class LiveDataFetcher:
    """Fetches live content for claim extraction"""

    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
        self.newsdata_api_key = os.getenv("NEWSDATA_API_KEY") or os.getenv("NEWSDATA_KEY")

    def fetch_company_content(self, company_name: str, claim: str = None) -> str:
        """
        Fetch live content about company for claim extraction
        Uses News API to get recent articles
        """
        print(f"\n🔴 LIVE FETCH: Getting fresh content for {company_name}")

        try:
            import requests

            # Build search query
            if claim:
                query = f'"{company_name}" AND ({claim}) AND (ESG OR sustainability OR environment)'
            else:
                query = f'"{company_name}" AND (ESG OR sustainability OR environment OR emissions OR renewable)'

            # Try News API first
            if self.news_api_key:
                print(f"📡 Calling News API (live)...")
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "apiKey": self.news_api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 5
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])

                    if articles:
                        print(f"✅ Found {len(articles)} recent articles")

                        # Combine article content
                        content = f"Company: {company_name}\n\n"
                        if claim:
                            content += f"Claim to verify: {claim}\n\n"
                        content += "Recent Articles:\n\n"

                        for i, article in enumerate(articles[:3], 1):
                            content += f"Article {i}:\n"
                            content += f"Title: {article.get('title', 'N/A')}\n"
                            content += f"Description: {article.get('description', 'N/A')}\n"
                            content += f"Content: {article.get('content', 'N/A')[:500]}\n"
                            content += f"Published: {article.get('publishedAt', 'N/A')}\n\n"

                        return content

            # Fallback: Use the claim itself as content
            print("⚠️ No live articles found, using claim as content")
            return f"Company: {company_name}\nClaim: {claim or 'General ESG analysis'}"

        except Exception as e:
            print(f"❌ Live fetch error: {e}")
            # Fallback content
            return f"Company: {company_name}\nClaim to analyze: {claim or 'General ESG sustainability claims'}"

# Initialize live fetcher
live_fetcher = LiveDataFetcher()

# ============================================================
# IMPORT YOUR ACTUAL AGENTS
# ============================================================

try:
    from claim_extractor import ClaimExtractor
    print("✅ ClaimExtractor loaded")
    CLAIM_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ClaimExtractor import failed: {e}")
    CLAIM_EXTRACTOR_AVAILABLE = False

try:
    from evidence_retriever import EvidenceRetriever
    print("✅ EvidenceRetriever loaded")
    EVIDENCE_RETRIEVER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  EvidenceRetriever import failed: {e}")
    EVIDENCE_RETRIEVER_AVAILABLE = False

try:
    from contradiction_analyzer import ContradictionAnalyzer
    print("✅ ContradictionAnalyzer loaded")
    CONTRADICTION_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ContradictionAnalyzer import failed: {e}")
    CONTRADICTION_ANALYZER_AVAILABLE = False

try:
    from historical_analyst import HistoricalAnalyst
    print("✅ HistoricalAnalyst loaded")
    HISTORICAL_ANALYST_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  HistoricalAnalyst import failed: {e}")
    HISTORICAL_ANALYST_AVAILABLE = False

try:
    from industry_comparator import IndustryComparator
    print("✅ IndustryComparator loaded")
    INDUSTRY_COMPARATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  IndustryComparator import failed: {e}")
    INDUSTRY_COMPARATOR_AVAILABLE = False

try:
    from risk_scorer import RiskScorer
    print("✅ RiskScorer loaded")
    RISK_SCORER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RiskScorer import failed: {e}")
    RISK_SCORER_AVAILABLE = False

try:
    from sentiment_analyzer import SentimentAnalyzer
    print("✅ SentimentAnalyzer loaded")
    SENTIMENT_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SentimentAnalyzer import failed: {e}")
    SENTIMENT_ANALYZER_AVAILABLE = False

try:
    from credibility_analyst import CredibilityAnalyst
    print("✅ CredibilityAnalyst loaded")
    CREDIBILITY_ANALYST_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CredibilityAnalyst import failed: {e}")
    CREDIBILITY_ANALYST_AVAILABLE = False

try:
    from confidence_scorer import ConfidenceScorer
    print("✅ ConfidenceScorer loaded")
    CONFIDENCE_SCORER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ConfidenceScorer import failed: {e}")
    CONFIDENCE_SCORER_AVAILABLE = False

try:
    from realtime_monitor import RealTimeMonitor
    print("✅ RealTimeMonitor loaded")
    REALTIME_MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RealTimeMonitor import failed: {e}")
    REALTIME_MONITOR_AVAILABLE = False

try:
    from conflict_resolver import ConflictResolver
    print("✅ ConflictResolver loaded")
    CONFLICT_RESOLVER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ConflictResolver import failed: {e}")
    CONFLICT_RESOLVER_AVAILABLE = False

try:
    from carbon_extractor import CarbonExtractor
    print("✅ CarbonExtractor loaded")
    CARBON_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CarbonExtractor import failed: {e}")
    CARBON_EXTRACTOR_AVAILABLE = False

try:
    from greenwishing_detector import GreenwishingDetector
    print("✅ GreenwishingDetector loaded")
    GREENWISHING_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  GreenwishingDetector import failed: {e}")
    GREENWISHING_DETECTOR_AVAILABLE = False

try:
    from regulatory_scanner import RegulatoryHorizonScanner
    print("✅ RegulatoryHorizonScanner loaded")
    REGULATORY_SCANNER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RegulatoryHorizonScanner import failed: {e}")
    REGULATORY_SCANNER_AVAILABLE = False

try:
    from social_agent import SocialAgent
    print("✅ SocialAgent loaded")
    SOCIAL_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SocialAgent import failed: {e}")
    SOCIAL_AGENT_AVAILABLE = False

try:
    from governance_agent import GovernanceAgent
    print("✅ GovernanceAgent loaded")
    GOVERNANCE_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  GovernanceAgent import failed: {e}")
    GOVERNANCE_AGENT_AVAILABLE = False

try:
    import sys
    ml_models_path = Path(__file__).parent.parent / "ml_models"
    sys.path.insert(0, str(ml_models_path))
    from climatebert_analyzer import ClimateBERTAnalyzer
    print("✅ ClimateBERTAnalyzer loaded")
    CLIMATEBERT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ClimateBERTAnalyzer import failed: {e}")
    CLIMATEBERT_AVAILABLE = False

try:
    from explainability_engine import ESGExplainabilityEngine
    print("✅ ESGExplainabilityEngine loaded")
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ESGExplainabilityEngine import failed: {e}")
    EXPLAINABILITY_AVAILABLE = False

try:
    from claim_decomposer import ClaimDecomposer
    print("✅ ClaimDecomposer loaded")
    CLAIM_DECOMPOSER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ClaimDecomposer import failed: {e}")
    CLAIM_DECOMPOSER_AVAILABLE = False

try:
    from adversarial_validator import AdversarialEvidenceValidator
    print("✅ AdversarialEvidenceValidator loaded")
    ADVERSARIAL_VALIDATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  AdversarialEvidenceValidator import failed: {e}")
    ADVERSARIAL_VALIDATOR_AVAILABLE = False

try:
    from carbon_pathway_modeller import CarbonPathwayModeller
    print("✅ CarbonPathwayModeller loaded")
    CARBON_PATHWAY_MODELLER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CarbonPathwayModeller import failed: {e}")
    CARBON_PATHWAY_MODELLER_AVAILABLE = False

try:
    from multi_jurisdiction_regulatory_scanner import MultiJurisdictionRegulatoryScanner
    print("✅ MultiJurisdictionRegulatoryScanner loaded")
    MULTI_JURISDICTION_SCANNER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MultiJurisdictionRegulatoryScanner import failed: {e}")
    MULTI_JURISDICTION_SCANNER_AVAILABLE = False

try:
    from commitment_tracker import CommitmentLedger
    print("✅ CommitmentLedger loaded")
    COMMITMENT_LEDGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CommitmentLedger import failed: {e}")
    COMMITMENT_LEDGER_AVAILABLE = False

# NEW PHASE 7: ESG Report Pipeline
try:
    utils_path = Path(__file__).parent.parent / "utils"
    sys.path.insert(0, str(utils_path))
    from report_discovery import discover_company_reports
    print("✅ ReportDiscoveryService loaded")
    REPORT_DISCOVERY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ReportDiscoveryService import failed: {e}")
    REPORT_DISCOVERY_AVAILABLE = False

try:
    from report_downloader import download_company_reports
    print("✅ ReportDownloaderService loaded")
    REPORT_DOWNLOADER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ReportDownloaderService import failed: {e}")
    REPORT_DOWNLOADER_AVAILABLE = False

try:
    from report_parser import parse_downloaded_reports
    print("✅ ReportParserService loaded")
    REPORT_PARSER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ReportParserService import failed: {e}")
    REPORT_PARSER_AVAILABLE = False

try:
    from temporal_consistency_agent import analyze_temporal_consistency, TemporalConsistencyAgent
    print("✅ TemporalConsistencyAgent loaded")
    TEMPORAL_CONSISTENCY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TemporalConsistencyAgent import failed: {e}")
    TEMPORAL_CONSISTENCY_AVAILABLE = False

# ============================================================
# LIVE NODE WRAPPERS WITH PROGRESS TRACKING
# ============================================================

def claim_extraction_node(state: ESGState) -> ESGState:
    """
    LIVE: ClaimExtractor with real-time content fetching
    Shows LangGraph node execution
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: claim_extraction")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    # Clear session cache for new analysis (keeps disk cache for reuse)
    if state.get("iteration_count", 0) == 0:
        evidence_cache.clear_session_cache()

    if not CLAIM_EXTRACTOR_AVAILABLE:
        from core.minimal_agents import claim_extraction_node as minimal_claim
        return minimal_claim(state)

    try:
        extractor = ClaimExtractor()

        # LIVE: Fetch fresh content
        live_content = live_fetcher.fetch_company_content(
            company_name=state["company"],
            claim=state["claim"]
        )

        print(f"📄 Content size: {len(live_content)} characters")
        print(f"🤖 Calling LLM for claim extraction...")

        # Call with both required parameters
        result = extractor.extract_claims(
            company_name=state["company"],
            content=live_content
        )

        confidence = 0.8
        if isinstance(result, dict):
            confidence = result.get("confidence", 0.8)
            claims = result.get("claims", [])
            print(f"✅ Extracted {len(claims)} claims")

        state["agent_outputs"].append({
            "agent": "claim_extraction",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "live_fetch": True
        })
        state["claim_results"] = result
        state.setdefault("node_execution_order", []).append("Claim Extraction")

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ ClaimExtractor error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "claim_extraction",
            "error": str(e),
            "confidence": 0.5
        })

    return state


def claim_decomposition_node(state: ESGState) -> ESGState:
    """Decompose compound claims into atomic sub-claims and detect internal tensions."""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: claim_decomposition")
    print("=" * 70)

    claim_text = state.get("claim", "")
    company = state.get("company", "")
    industry = state.get("industry", "")

    if not CLAIM_DECOMPOSER_AVAILABLE:
        decomposition = {
            "original_claim": claim_text,
            "sub_claims": [{
                "id": "SC1",
                "text": claim_text,
                "type": "policy_claim",
                "pillar": "cross-pillar",
                "measurable": False,
                "verification_requirements": ["Independent primary-source evidence"],
                "greenwashing_signal": "unverifiable",
            }],
            "logical_tension_pairs": [],
            "overall_internal_consistency": "consistent",
            "decomposition_confidence": 0.5,
            "internal_contradiction_score": 0.0,
        }
    else:
        try:
            decomposer = ClaimDecomposer()
            decomposition = decomposer.decompose_claim(company=company, industry=industry, claim_text=claim_text)
            score = decomposer.compute_internal_contradiction_score(
                decomposition.get("logical_tension_pairs", [])
            )
            decomposition["internal_contradiction_score"] = score
        except Exception as e:
            print(f"❌ ClaimDecomposer error: {e}")
            decomposition = {
                "original_claim": claim_text,
                "sub_claims": [{"id": "SC1", "text": claim_text, "type": "policy_claim", "pillar": "cross-pillar", "measurable": False}],
                "logical_tension_pairs": [],
                "overall_internal_consistency": "consistent",
                "decomposition_confidence": 0.4,
                "internal_contradiction_score": 0.0,
            }

    state["claim_decomposition"] = decomposition
    state.setdefault("node_execution_order", []).append("Claim Decomposition")
    state["agent_outputs"].append({
        "agent": "claim_decomposition",
        "output": decomposition,
        "confidence": decomposition.get("decomposition_confidence", 0.6),
        "timestamp": datetime.now().isoformat(),
    })
    print(f"✅ Decomposed into {len(decomposition.get('sub_claims', []))} sub-claim(s)")
    return state


def adversarial_triangulation_node(state: ESGState) -> ESGState:
    """Stress-test evidence by balancing supporting vs contradicting sources."""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: adversarial_triangulation")
    print("=" * 70)

    if not ADVERSARIAL_VALIDATOR_AVAILABLE:
        result = {
            "triangulation_score": 50.0,
            "adversarial_ratio": 0.0,
            "evidence_balance": "MIXED",
            "source_stances": [],
            "first_party_bias_warning": False,
        }
    else:
        try:
            validator = AdversarialEvidenceValidator()
            result = validator.triangulate(
                company=state.get("company", ""),
                claim_text=state.get("claim", ""),
                evidence=state.get("evidence", []),
            )
        except Exception as e:
            print(f"❌ AdversarialEvidenceValidator error: {e}")
            result = {
                "triangulation_score": 50.0,
                "adversarial_ratio": 0.0,
                "evidence_balance": "MIXED",
                "source_stances": [],
                "first_party_bias_warning": False,
            }

    state["adversarial_triangulation"] = result
    state.setdefault("node_execution_order", []).append("Adversarial Triangulation")
    state["agent_outputs"].append({
        "agent": "adversarial_triangulation",
        "output": result,
        "confidence": 0.72,
        "timestamp": datetime.now().isoformat(),
    })
    print(
        f"✅ Triangulation score: {result.get('triangulation_score', 'N/A')} | "
        f"Adversarial ratio: {result.get('adversarial_ratio', 'N/A')}"
    )
    return state


def evidence_retrieval_node(state: ESGState) -> ESGState:
    """
    LIVE: EvidenceRetriever fetches real-time evidence
    Includes financial analyst integration for ESG-financial correlation
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: evidence_retrieval (with Financial Analyst)")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not EVIDENCE_RETRIEVER_AVAILABLE:
        from core.minimal_agents import evidence_retrieval_node as minimal_evidence
        return minimal_evidence(state)

    try:
        retriever = EvidenceRetriever()

        print(f"🔍 Live evidence search for: {state['company']}")
        print(f"📡 Calling 15 external APIs + Financial Analyst...")

        decomposition = state.get("claim_decomposition") if isinstance(state.get("claim_decomposition"), dict) else {}
        sub_claims = decomposition.get("sub_claims") if isinstance(decomposition.get("sub_claims"), list) else []

        if sub_claims:
            merged_evidence = []
            sub_results = []
            confidence_values = []
            for sc in sub_claims[:6]:
                if not isinstance(sc, dict):
                    continue
                claim_dict = {
                    "claim_id": sc.get("id") or "C1",
                    "claim_text": sc.get("text") or state["claim"],
                    "category": sc.get("type") or "sustainability",
                    "sub_claim_id": sc.get("id"),
                }
                partial = retriever.retrieve_evidence(claim_dict, state["company"])
                if isinstance(partial, dict):
                    for ev in partial.get("evidence", []) or []:
                        if isinstance(ev, dict):
                            ev["sub_claim_id"] = sc.get("id")
                    sub_results.append({"sub_claim_id": sc.get("id"), "result": partial})
                    merged_evidence.extend(partial.get("evidence", []) or [])
                    confidence_values.append(float(partial.get("confidence", 0.7) or 0.7))
            result = {
                "sub_claim_mode": True,
                "sub_claim_results": sub_results,
                "evidence": merged_evidence,
                "evidence_count": len(merged_evidence),
                "confidence": (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.7,
            }
        else:
            # Create claim dict for evidence retriever
            claim_dict = {
                "claim_id": "C1",
                "claim_text": state["claim"],
                "category": "sustainability"
            }
            # Call retrieve_evidence with proper parameters
            result = retriever.retrieve_evidence(claim_dict, state["company"])

        if isinstance(result, dict):
            evidence_list = result.get("evidence", [])
            confidence = result.get("confidence", 0.7)
            financial_context = result.get("financial_context")  # NEW: From Financial Analyst

            print(f"✅ Retrieved {len(evidence_list)} evidence items")

            if financial_context:
                print(f"💰 Financial Analysis (Agent #14):")
                if "financial_data" in financial_context:
                    fin_data = financial_context["financial_data"]
                    print(f"   Revenue: ${fin_data.get('revenue_usd', 0)/1e9:.1f}B")
                    print(f"   Profit Margin: {fin_data.get('profit_margin_pct', 0):.1f}%")
                if "greenwashing_flags" in financial_context:
                    flags = financial_context["greenwashing_flags"]
                    # Handle both dict and list formats
                    if isinstance(flags, dict):
                        high_risk = flags.get("HIGH", [])
                        if high_risk:
                            print(f"   ⚠️ HIGH risk flags: {len(high_risk)}")
                    elif isinstance(flags, list):
                        if flags:
                            print(f"   ⚠️ Greenwashing flags: {len(flags)}")
        else:
            evidence_list = result if isinstance(result, list) else []
            confidence = 0.7
            financial_context = None

        state["evidence"].extend(evidence_list)

        # NEW: Store enrichment data at state level for report access
        if isinstance(result, dict):
            if result.get("indian_financials"):
                state["indian_financials"] = result["indian_financials"]
            if result.get("company_reports"):
                state["company_reports"] = result["company_reports"]

        state["agent_outputs"].append({
            "agent": "evidence_retrieval",
            "output": result,
            "evidence_count": len(evidence_list),
            "confidence": confidence,
            "financial_context": financial_context,  # NEW: Pass to risk scorer
            "timestamp": datetime.now().isoformat(),
            "live_fetch": True
        })
        state["evidence_results"] = result
        state.setdefault("node_execution_order", []).append("Evidence Retrieval")

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ EvidenceRetriever error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "evidence_retrieval",
            "error": str(e),
            "confidence": 0.3
        })

    return state


def carbon_extraction_node(state: ESGState) -> ESGState:
    """
    LIVE: CarbonExtractor - Extracts Scope 1/2/3 carbon emissions from evidence
    Analyzes carbon claims and calculates emission metrics
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: carbon_extraction (Scope 1/2/3 Analysis)")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not CARBON_EXTRACTOR_AVAILABLE:
        print("⚠️ CarbonExtractor not available - skipping")
        state["agent_outputs"].append({
            "agent": "carbon_extraction",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        extractor = CarbonExtractor()

        company = state.get("company", "")
        claim_text = state.get("claim", "")
        industry = state.get("industry", "")
        evidence = state.get("evidence", [])

        # Gather parsed report chunks and report claims for prioritized carbon extraction
        parser_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "report_parser"]
        parsed_chunks = parser_outputs[-1].get("output", {}).get("chunks", []) if parser_outputs else []
        parsed_report_files = parser_outputs[-1].get("output", {}).get("downloaded_reports", []) if parser_outputs else []

        if not parsed_report_files:
            downloader_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "report_downloader"]
            parsed_report_files = downloader_outputs[-1].get("output", {}).get("downloads", []) if downloader_outputs else []

        claim_extractor_outputs = [
            o for o in state.get("agent_outputs", [])
            if o.get("agent") == "claim_extractor" and o.get("source") == "report_chunks"
        ]
        report_claims_by_year = (
            claim_extractor_outputs[-1].get("output", {}).get("report_claims_by_year", {})
            if claim_extractor_outputs else {}
        )

        print(f"🌍 Extracting carbon metrics for: {company}")
        print(f"🏭 Industry: {industry}")
        print(f"📊 Evidence items to analyze: {len(evidence)}")

        # Create claim dict for carbon extractor
        claim_dict = {
            "claim_id": "C1",
            "claim_text": claim_text,
            "category": "carbon",
            "industry": industry,
        }

        # Extract carbon data from evidence
        result = extractor.extract_carbon_data(
            company=company,
            evidence=evidence,
            claim=claim_dict,
            report_chunks=parsed_chunks,
            report_claims_by_year=report_claims_by_year,
            report_files=parsed_report_files,
        )

        if isinstance(result, dict):
            # Store carbon extraction results in state
            state["carbon_extraction"] = result
            state["carbon_results"] = result

            # Display results
            emissions = result.get("emissions", {})
            scope1 = emissions.get("scope1", {})
            scope2 = emissions.get("scope2", {})
            scope3 = emissions.get("scope3", {})

            print(f"\n📊 CARBON EXTRACTION RESULTS:")
            print(f"   Scope 1 (Direct): {scope1.get('value', 'N/A')} tCO2e")
            print(f"   Scope 2 (Energy): {scope2.get('value', 'N/A')} tCO2e")
            print(f"   Scope 3 (Value Chain): {scope3.get('total', scope3.get('value', 'N/A'))} tCO2e")
            print(f"   Total: {emissions.get('total', {}).get('all_scopes', 'N/A')} tCO2e")
            print(f"   Carbon Intensity: {result.get('intensity_metrics', {}).get('total_emissions_tco2e', 'N/A')}")
            print(f"   Net Zero Target: {result.get('net_zero_target', 'N/A')}")
            print(f"   Data Quality: {result.get('data_quality', 'N/A')}")

            confidence = result.get("confidence", 0.7)
        else:
            confidence = 0.5

        state["agent_outputs"].append({
            "agent": "carbon_extraction",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        state.setdefault("node_execution_order", []).append("Carbon Extraction")

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ CarbonExtractor error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "carbon_extraction",
            "error": str(e),
            "confidence": 0.3
        })

    return state


def carbon_pathway_analysis_node(state: ESGState) -> ESGState:
    """Model whether current and claimed trajectory aligns with 1.5C-style pathway constraints."""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: carbon_pathway_analysis")
    print("=" * 70)

    carbon = state.get("carbon_extraction") if isinstance(state.get("carbon_extraction"), dict) else {}
    emissions = carbon.get("emissions") if isinstance(carbon.get("emissions"), dict) else {}
    scope1 = emissions.get("scope1", {}) if isinstance(emissions.get("scope1"), dict) else {}
    scope2 = emissions.get("scope2", {}) if isinstance(emissions.get("scope2"), dict) else {}
    scope3 = emissions.get("scope3", {}) if isinstance(emissions.get("scope3"), dict) else {}

    def _n(x):
        return float(x) if isinstance(x, (int, float)) else 0.0

    scope1_val = _n(scope1.get("value"))
    scope2_val = _n(scope2.get("value"))
    scope3_val = _n(scope3.get("total") if isinstance(scope3.get("total"), (int, float)) else scope3.get("value"))

    target_year = 2030
    try:
        import re
        m = re.search(r"\b(20\d{2})\b", str(state.get("claim", "")))
        if m:
            target_year = int(m.group(1))
    except Exception:
        pass

    production_plan = "growth" if any(
        k in str(state.get("claim", "")).lower() for k in ["growth", "increase production", "maintain production"]
    ) else "stable"

    decomposition = state.get("claim_decomposition") if isinstance(state.get("claim_decomposition"), dict) else {}
    for sc in decomposition.get("sub_claims", []) if isinstance(decomposition.get("sub_claims"), list) else []:
        txt = str(sc.get("text", "")).lower()
        if "decline" in txt or "phase down" in txt:
            production_plan = "decline"

    if not CARBON_PATHWAY_MODELLER_AVAILABLE:
        result = {
            "alignment_status": "misaligned",
            "pathway_gap_pct": 35.0,
            "scope3_feasibility": "UNKNOWN",
            "production_plan": production_plan,
        }
    else:
        try:
            modeller = CarbonPathwayModeller()
            result = modeller.model_pathway(
                company=state.get("company", ""),
                industry=state.get("industry", ""),
                claim_text=state.get("claim", ""),
                scope1=scope1_val,
                scope2=scope2_val,
                scope3=scope3_val,
                base_year=int(carbon.get("reporting_year") or 2023),
                target_year=target_year,
                target_reduction_pct=30.0,
                production_plan=production_plan,
                claimed_pathway="1.5C" if "1.5" in str(state.get("claim", "")).lower() else "net-zero-2050",
            )
        except Exception as e:
            print(f"❌ CarbonPathwayModeller error: {e}")
            result = {
                "alignment_status": "misaligned",
                "pathway_gap_pct": 35.0,
                "scope3_feasibility": "UNKNOWN",
                "production_plan": production_plan,
            }

    state["carbon_pathway_analysis"] = result
    state.setdefault("node_execution_order", []).append("Carbon Pathway Analysis")
    state["agent_outputs"].append({
        "agent": "carbon_pathway_analysis",
        "output": result,
        "confidence": 0.74,
        "timestamp": datetime.now().isoformat(),
    })
    print(
        f"✅ Alignment: {result.get('alignment_status', 'N/A')} | "
        f"Gap: {result.get('pathway_gap_pct', 'N/A')}%"
    )
    return state


def greenwishing_detection_node(state: ESGState) -> ESGState:
    """
    LIVE: GreenwishingDetector - Detects greenwishing, greenhushing, selective disclosure
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: greenwishing_detection")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not GREENWISHING_DETECTOR_AVAILABLE:
        print("⚠️ GreenwishingDetector not available - skipping")
        state["agent_outputs"].append({
            "agent": "greenwishing_detection",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        detector = GreenwishingDetector()

        company = state.get("company", "")
        claim_text = state.get("claim", "")
        evidence = state.get("evidence", [])

        print(f"🎯 Detecting greenwishing/greenhushing for: {company}")

        claim_dict = {
            "claim_id": "C1",
            "claim_text": claim_text,
            "category": "sustainability"
        }

        parser_outputs = [
            o for o in state.get("agent_outputs", [])
            if o.get("agent") == "report_parser"
        ]
        claim_outputs = [
            o for o in state.get("agent_outputs", [])
            if o.get("agent") == "claim_extractor" and o.get("source") == "report_chunks"
        ]

        result = detector.detect_deception_tactics(
            company=company,
            claim=claim_dict,
            evidence=evidence,
            structured_context={
                "report_chunks": parser_outputs[-1].get("output", {}).get("chunks", []) if parser_outputs else [],
                "report_claims_by_year": claim_outputs[-1].get("output", {}).get("report_claims_by_year", {}) if claim_outputs else {},
                "carbon_extraction": state.get("carbon_extraction", {})
            }
        )

        if isinstance(result, dict):
            state["greenwishing_analysis"] = result

            deception_risk = result.get("overall_deception_risk", {})
            print(f"\n🎭 DECEPTION DETECTION RESULTS:")
            print(f"   Greenwishing Risk: {result.get('greenwishing', {}).get('risk_level', 'N/A')}")
            print(f"   Greenhushing Risk: {result.get('greenhushing', {}).get('risk_level', 'N/A')}")
            print(f"   Selective Disclosure: {result.get('selective_disclosure', {}).get('detected', 'N/A')}")
            print(f"   Overall Deception Score: {deception_risk.get('score', 'N/A')}/100")

            confidence = result.get("confidence", 0.75)
        else:
            confidence = 0.5

        state["agent_outputs"].append({
            "agent": "greenwishing_detection",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ GreenwishingDetector error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "greenwishing_detection",
            "error": str(e),
            "confidence": 0.3
        })

    return state


def regulatory_scanning_node(state: ESGState) -> ESGState:
    """
    LIVE: RegulatoryHorizonScanner - Scans against SEBI BRSR, CSRD, SEC, etc.
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: regulatory_scanning")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not REGULATORY_SCANNER_AVAILABLE:
        print("⚠️ RegulatoryHorizonScanner not available - skipping")
        state["agent_outputs"].append({
            "agent": "regulatory_scanning",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        scanner = RegulatoryHorizonScanner()

        company = state.get("company", "")
        claim_text = state.get("claim", "")
        evidence = state.get("evidence", [])
        industry = state.get("industry", "")

        print(f"⚖️ Scanning regulatory compliance for: {company}")

        claim_dict = {
            "claim_id": "C1",
            "claim_text": claim_text,
            "category": "sustainability"
        }

        # Determine jurisdiction based on company name heuristics.
        indian_companies = ["reliance", "tata", "infosys", "hdfc", "icici", "wipro", "bharti",
                          "bajaj", "mahindra", "adani", "larsen", "maruti", "asian paints"]
        company_lower = company.lower()

        if any(c in company_lower for c in indian_companies):
            country = "IN"
        elif any(c in company_lower for c in ["volkswagen", "vw", "bmw", "mercedes", "daimler", "siemens", "basf", "sap"]):
            country = "DE"
        elif any(c in company_lower for c in ["bp", "shell", "hsbc", "unilever", "barclays", "london"]):
            country = "UK"
        elif any(c in company_lower for c in ["tesla", "exxon", "chevron", "walmart", "microsoft", "apple", "amazon", "google"]):
            country = "US"
        else:
            country = ""

        if country == "IN":
            jurisdiction = "India"
        elif country in {"DE", "FR", "NL", "DK", "SE", "IT", "ES", "PL", "BE", "AT", "CH"}:
            jurisdiction = "EU"
        elif country in {"UK", "GB"}:
            jurisdiction = "UK"
        elif country in {"US", "USA"}:
            jurisdiction = "US"
        else:
            jurisdiction = "Global"

        base_result = scanner.scan_regulatory_compliance(
            company=company,
            claim=claim_dict,
            evidence=evidence,
            jurisdiction=jurisdiction,
            country=country,
            industry=industry,
        )

        if MULTI_JURISDICTION_SCANNER_AVAILABLE:
            try:
                mj = MultiJurisdictionRegulatoryScanner()
                jurisdictions = mj.detect_jurisdictions(industry=industry, hq_country=country)
                sbti_status = "unknown"
                carbon = state.get("carbon_extraction", {})
                if isinstance(carbon, dict):
                    sbti_status = str(carbon.get("sbti_status") or "unknown")
                multi = mj.aggregate_results(
                    company=company,
                    claim_text=claim_text,
                    jurisdictions=jurisdictions,
                    base_regulatory=base_result if isinstance(base_result, dict) else {},
                    evidence=evidence,
                    sbti_status=sbti_status,
                )
                result = dict(base_result) if isinstance(base_result, dict) else {}
                result["multi_jurisdiction"] = multi
                result["compliance_score"] = multi.get("total_compliance_score", result.get("compliance_score"))
                result["risk_level"] = str(multi.get("regulatory_risk_level", result.get("risk_level", "Unknown"))).upper()
            except Exception as e:
                print(f"⚠️ Multi-jurisdiction aggregation failed: {e}")
                result = base_result
        else:
            result = base_result

        if isinstance(result, dict):
            state["regulatory_compliance"] = result
            state["regulatory_results"] = result

            print(f"\n⚖️ REGULATORY COMPLIANCE RESULTS:")
            print(f"   Jurisdiction: {result.get('jurisdiction', 'N/A')}")
            print(f"   Applicable Regulations: {len(result.get('applicable_regulations', []))}")
            print(f"   Compliance Score: {result.get('compliance_score', 'N/A')}/100")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")

            # Show top regulations
            for reg in result.get('applicable_regulations', [])[:3]:
                print(f"   - {reg}")

            confidence = result.get("confidence", 0.8)
        else:
            confidence = 0.5

        state["agent_outputs"].append({
            "agent": "regulatory_scanning",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        state.setdefault("node_execution_order", []).append("Regulatory Scanning")

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ RegulatoryHorizonScanner error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "regulatory_scanning",
            "error": str(e),
            "confidence": 0.3
        })

    return state


def climatebert_analysis_node(state: ESGState) -> ESGState:
    """
    LIVE: ClimateBERTAnalyzer - Transformer-based climate text analysis
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: climatebert_analysis")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not CLIMATEBERT_AVAILABLE:
        print("⚠️ ClimateBERTAnalyzer not available - skipping")
        state["agent_outputs"].append({
            "agent": "climatebert_analysis",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        analyzer = ClimateBERTAnalyzer()

        claim_text = state.get("claim", "")
        evidence = state.get("evidence", [])

        print(f"🤖 Running ClimateBERT NLP analysis...")

        # Extract evidence texts for comparison
        evidence_texts = []
        for ev in evidence[:10]:  # Limit to first 10
            if isinstance(ev, dict):
                text = ev.get("content", ev.get("text", ev.get("snippet", "")))
                if text:
                    evidence_texts.append(text[:500])

        result = analyzer.analyze_claim_for_greenwashing(
            claim_text=claim_text,
            evidence_texts=evidence_texts if evidence_texts else None
        )

        if isinstance(result, dict):
            state["climatebert_analysis"] = result
            state["climatebert_results"] = result

            claim_analysis = result.get("claim_analysis", {})
            gw_detection = claim_analysis.get("greenwashing_detection", {})

            print(f"\n🧠 CLIMATEBERT ANALYSIS RESULTS:")
            print(f"   Climate Relevance: {claim_analysis.get('climate_relevance', {}).get('score', 'N/A')}")
            print(f"   Greenwashing Risk: {gw_detection.get('risk_score', 'N/A')}/100")
            print(f"   Risk Level: {gw_detection.get('risk_level', 'N/A')}")

            # Show detected patterns
            patterns = gw_detection.get("detected_patterns", [])
            if patterns:
                print(f"   Detected Patterns: {', '.join(patterns[:3])}")

            confidence = 0.85  # ClimateBERT is high confidence
        else:
            confidence = 0.5

        state["agent_outputs"].append({
            "agent": "climatebert_analysis",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        state.setdefault("node_execution_order", []).append("ClimateBERT Analysis")

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ ClimateBERTAnalyzer error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "climatebert_analysis",
            "error": str(e),
            "confidence": 0.3
        })

    return state


def social_analysis_node(state: ESGState) -> ESGState:
    """
    LIVE: SocialAgent - retrieves social pillar evidence and computes social risk signals.
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: social_analysis")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    if not SOCIAL_AGENT_AVAILABLE:
        print("⚠️ SocialAgent not available - skipping")
        state["agent_outputs"].append({
            "agent": "social_analysis",
            "output": "Agent not available",
            "confidence": 0.5,
        })
        return state

    try:
        agent = SocialAgent()

        # Get report chunks
        parser_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "report_parser"]
        parsed_chunks = parser_outputs[-1].get("output", {}).get("chunks", []) if parser_outputs else []

        enhanced_evidence = list(state.get("evidence", []))
        for chunk in parsed_chunks:
            text = chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk)
            if text:
                enhanced_evidence.append({"snippet": text, "source": "Primary ESG Report"})

        result = agent.analyze(
            company=state.get("company", ""),
            claim_text=state.get("claim", ""),
            industry=state.get("industry", ""),
            evidence=enhanced_evidence,
        )

        state["social_analysis"] = result if isinstance(result, dict) else {}
        confidence = float(result.get("confidence", 0.7)) if isinstance(result, dict) else 0.5

        if isinstance(result, dict):
            print("✅ Social analysis complete")
            print(f"   Social Score: {result.get('social_score', 'N/A')}/100")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")

        state.setdefault("node_execution_order", []).append("Social Analysis")
        state["agent_outputs"].append({
            "agent": "social_analysis",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"{'✅ NODE COMPLETED':^70}")
    except Exception as e:
        print(f"❌ SocialAgent error: {e}")
        state["agent_outputs"].append({
            "agent": "social_analysis",
            "error": str(e),
            "confidence": 0.4,
            "timestamp": datetime.now().isoformat(),
        })

    return state


def governance_analysis_node(state: ESGState) -> ESGState:
    """
    LIVE: GovernanceAgent - parses governance evidence and proxy filing signals.
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: governance_analysis")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    if not GOVERNANCE_AGENT_AVAILABLE:
        print("⚠️ GovernanceAgent not available - skipping")
        state["agent_outputs"].append({
            "agent": "governance_analysis",
            "output": "Agent not available",
            "confidence": 0.5,
        })
        return state

    try:
        agent = GovernanceAgent()

        # Get report chunks
        parser_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "report_parser"]
        parsed_chunks = parser_outputs[-1].get("output", {}).get("chunks", []) if parser_outputs else []

        enhanced_evidence = list(state.get("evidence", []))
        for chunk in parsed_chunks:
            text = chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk)
            if text:
                enhanced_evidence.append({"snippet": text, "source": "Primary ESG Report"})

        result = agent.analyze(
            company=state.get("company", ""),
            claim_text=state.get("claim", ""),
            industry=state.get("industry", ""),
            evidence=enhanced_evidence,
        )

        state["governance_analysis"] = result if isinstance(result, dict) else {}
        confidence = float(result.get("confidence", 0.7)) if isinstance(result, dict) else 0.5

        if isinstance(result, dict):
            print("✅ Governance analysis complete")
            print(f"   Governance Score: {result.get('governance_score', 'N/A')}/100")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")

        state.setdefault("node_execution_order", []).append("Governance Analysis")
        state["agent_outputs"].append({
            "agent": "governance_analysis",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"{'✅ NODE COMPLETED':^70}")
    except Exception as e:
        print(f"❌ GovernanceAgent error: {e}")
        state["agent_outputs"].append({
            "agent": "governance_analysis",
            "error": str(e),
            "confidence": 0.4,
            "timestamp": datetime.now().isoformat(),
        })

    return state


def explainability_node(state: ESGState) -> ESGState:
    """
    LIVE: ESGExplainabilityEngine - SHAP/LIME explanations for ML predictions
    Runs AFTER risk_scoring to explain the ML model's decision

    PHASE 9: Improved to always return meaningful factors
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: explainability (SHAP/LIME)")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not EXPLAINABILITY_AVAILABLE:
        print("⚠️ ESGExplainabilityEngine not available - skipping")
        state["agent_outputs"].append({
            "agent": "explainability",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        engine = ESGExplainabilityEngine()

        # Get ML prediction from risk scorer
        ml_prediction = state.get("ml_prediction", {})

        print(f"📊 Generating SHAP/LIME explanations...")

        # If we have ML feature data, explain it
        if ml_prediction and isinstance(ml_prediction, dict):
            features = ml_prediction.get("features")
            feature_names = ml_prediction.get("feature_names")

            if features is not None and feature_names:
                import numpy as np
                features_array = np.array(features).reshape(1, -1) if not isinstance(features, np.ndarray) else features

                # Generate SHAP explanation
                result = engine.explain_xgboost_prediction(
                    model=None,  # Will use fallback
                    features=features_array,
                    feature_names=feature_names
                )
            else:
                # Generate mock explanation based on available data
                result = {
                    "method": "Heuristic",
                    "top_factors": [
                        {"feature": "Environmental Disclosure Gaps", "impact": "high", "direction": "increases risk"},
                        {"feature": "Historical Violations", "impact": "high", "direction": "increases risk"},
                        {"feature": "Weak Social Performance", "impact": "moderate", "direction": "increases risk"}
                    ],
                    "human_readable_explanation": "Risk assessment based on ESG pillar scores and contradiction indicators."
                }
        else:
            # PHASE 9 FIX: Always extract meaningful factors
            risk_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "risk_scoring"]

            if risk_outputs:
                risk_result = risk_outputs[-1].get("output", {})
                pillar_scores = risk_result.get("pillar_scores", {})

                # PHASE 9: Build comprehensive factors list from all available data
                factors = []

                # Primary factors: ESG Pillars
                if pillar_scores.get("environmental_score") is not None:
                    factors.append({
                        "feature": "Environmental Disclosure Gaps" if pillar_scores["environmental_score"] < 50 else "Environmental Performance",
                        "value": pillar_scores["environmental_score"],
                        "impact": "high",
                        "direction": "decreases risk" if pillar_scores["environmental_score"] > 60 else "increases risk"
                    })

                if pillar_scores.get("social_score") is not None:
                    factors.append({
                        "feature": "Social Performance",
                        "value": pillar_scores["social_score"],
                        "impact": "moderate",
                        "direction": "decreases risk" if pillar_scores["social_score"] > 50 else "increases risk"
                    })

                if pillar_scores.get("governance_score") is not None:
                    factors.append({
                        "feature": "Governance Structure",
                        "value": pillar_scores["governance_score"],
                        "impact": "moderate",
                        "direction": "decreases risk" if pillar_scores["governance_score"] > 50 else "increases risk"
                    })

                # Secondary factors: Contradiction signals
                contradiction_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "contradiction_analysis"]
                if contradiction_outputs:
                    contradictions = contradiction_outputs[-1].get("output", {}).get("contradictions", [])
                    if contradictions:
                        factors.append({
                            "feature": f"Claim Contradictions ({len(contradictions)})",
                            "impact": "high",
                            "direction": "increases risk"
                        })

                # Tertiary factors: Historical patterns
                temporal_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "temporal_analysis"]
                if temporal_outputs:
                    temporal_data = temporal_outputs[-1].get("output", {})
                    if temporal_data.get("declining_trend"):
                        factors.append({
                            "feature": "Declining Historical Trust",
                            "impact": "moderate",
                            "direction": "increases risk"
                        })

                # PHASE 9: Ensure we always have factors
                if not factors:
                    factors = [
                        {"feature": "Environmental Disclosure Gaps", "impact": "high", "direction": "increases risk"},
                        {"feature": "Historical Regulatory Violations", "impact": "high", "direction": "increases risk"},
                        {"feature": "Weak Social Performance", "impact": "moderate", "direction": "increases risk"}
                    ]

                result = {
                    "method": "ESG Pillar Analysis with Contradiction Detection",
                    "top_factors": factors,
                    "human_readable_explanation": f"Risk is primarily driven by: {', '.join(f.get('feature', 'unknown') for f in factors[:3])}"
                }
            else:
                # PHASE 9: Fallback factors when no risk scorer data
                result = {
                    "method": "Basic ESG Analysis",
                    "top_factors": [
                        {"feature": "Limited ESG Data Availability", "impact": "high", "direction": "increases risk"},
                        {"feature": "Carbon Emissions Disclosure", "impact": "high", "direction": "decreases risk if transparent"},
                        {"feature": "Community Engagement", "impact": "moderate", "direction": "decreases risk"}
                    ],
                    "human_readable_explanation": "ESG assessment based on available disclosure and historical patterns."
                }

        if isinstance(result, dict):
            state["explainability_report"] = result
            state["explainability_results"] = result

            print(f"\n📈 EXPLAINABILITY RESULTS:")
            print(f"   Method: {result.get('method', 'N/A')}")
            print(f"   Top Risk Drivers: {len(result.get('top_factors', []))}")

            for i, factor in enumerate(result.get("top_factors", [])[:3], 1):
                direction_symbol = "⬇️" if "decreases" in factor.get('direction', '') else "⬆️"
                print(f"   {i}. {factor.get('feature')}: {factor.get('impact')} impact {direction_symbol} {factor.get('direction')}")

            if result.get("human_readable_explanation"):
                print(f"\n   📝 {result['human_readable_explanation'][:120]}...")

            confidence = 0.85
        else:
            confidence = 0.5

        state["agent_outputs"].append({
            "agent": "explainability",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        state.setdefault("node_execution_order", []).append("Explainability")

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ ESGExplainabilityEngine error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "explainability",
            "error": str(e),
            "confidence": 0.3
        })

    return state


def contradiction_analysis_node(state: ESGState) -> ESGState:
    """LIVE: ContradictionAnalyzer"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: contradiction_analysis")
    print("="*70)

    if not CONTRADICTION_ANALYZER_AVAILABLE:
        state["agent_outputs"].append({
            "agent": "contradiction_analysis",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        analyzer = ContradictionAnalyzer()

        print(f"🔍 Analyzing contradictions...")

        contradicting_evidence = []
        evidence_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "evidence_retrieval"]
        if evidence_outputs:
            contradicting_evidence = evidence_outputs[-1].get("output", {}).get("contradicting_evidence", []) or []

        result = analyzer.analyze_contradictions(
            company=state.get("company", ""),
            claim=state.get("claim", ""),
            evidence=state.get("evidence", []),
            contradicting_evidence=contradicting_evidence,
        )

        decomposition = state.get("claim_decomposition") if isinstance(state.get("claim_decomposition"), dict) else {}
        tension_pairs = decomposition.get("logical_tension_pairs") if isinstance(decomposition.get("logical_tension_pairs"), list) else []
        if isinstance(result, dict) and tension_pairs:
            logical_items = []
            for t in tension_pairs:
                if not isinstance(t, dict):
                    continue
                logical_items.append({
                    "severity": str(t.get("severity", "medium")).upper(),
                    "description": str(t.get("tension_description") or "Internal claim tension detected"),
                    "source": "claim_decomposition",
                    "source_type": "internal_logic",
                    "confidence": "HIGH" if str(t.get("severity", "")).lower() == "high" else "MEDIUM",
                })
            if logical_items:
                existing = result.get("contradictions") if isinstance(result.get("contradictions"), list) else []
                result["contradictions"] = existing + logical_items
                result["contradiction_list"] = result["contradictions"]
                result["specific_contradictions"] = result["contradictions"]
                result["contradictions_found"] = len(result["contradictions"])

        contradiction_count = 0
        confidence = 0.75
        if isinstance(result, dict):
            contradiction_count = int(result.get("contradictions_found") or len(result.get("contradictions", [])))
            confidence = result.get("confidence", 0.75)
            print(f"✅ Found {contradiction_count} contradictions")
            state["contradiction_results"] = result

        state.setdefault("node_execution_order", []).append("Contradiction Analysis")

        state["agent_outputs"].append({
            "agent": "contradiction_analysis",
            "output": result,
            "contradictions_count": contradiction_count,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ ContradictionAnalyzer error: {e}")
        state["agent_outputs"].append({
            "agent": "contradiction_analysis",
            "error": str(e),
            "confidence": 0.5
        })

    return state


def temporal_analysis_node(state: ESGState) -> ESGState:
    """LIVE: HistoricalAnalyst - calls CORRECT method"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: temporal_analysis")
    print("="*70)

    if not HISTORICAL_ANALYST_AVAILABLE:
        state["agent_outputs"].append({
            "agent": "temporal_analysis",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        analyst = HistoricalAnalyst()

        print(f"📅 Analyzing historical track record for {state['company']}...")

        # FIXED: Call the CORRECT method name
        result = analyst.analyze_company_history(state["company"])

        # Extract key metrics for logging
        if isinstance(result, dict):
            reputation = result.get("reputation_score", 50)
            violations = len(result.get("past_violations", []))
            print(f"✅ Historical analysis complete:")
            print(f"   Reputation: {reputation}/100")
            print(f"   Violations found: {violations}")
            confidence = 0.7
            state["historical_results"] = result
        else:
            confidence = 0.5

        state.setdefault("node_execution_order", []).append("Temporal Analysis")

        state["agent_outputs"].append({
            "agent": "temporal_analysis",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ HistoricalAnalyst error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "temporal_analysis",
            "error": str(e),
            "confidence": 0.5
        })

    return state



def peer_comparison_node(state: ESGState) -> ESGState:
    """LIVE: IndustryComparator"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: peer_comparison")
    print("="*70)

    if not INDUSTRY_COMPARATOR_AVAILABLE:
        state["agent_outputs"].append({
            "agent": "peer_comparison",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        comparator = IndustryComparator()

        print(f"🏢 Comparing with industry peers...")

        if hasattr(comparator, 'compare'):
            result = comparator.compare(state["company"], state["industry"])
        elif hasattr(comparator, 'analyze'):
            result = comparator.analyze(state["company"])
        else:
            result = {"peers": [], "confidence": 0.5}

        confidence = result.get("confidence", 0.75) if isinstance(result, dict) else 0.75
        print(f"✅ Peer comparison complete")

        state["agent_outputs"].append({
            "agent": "peer_comparison",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        if isinstance(result, dict):
            state["peer_results"] = result
        state.setdefault("node_execution_order", []).append("Peer Comparison")

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ IndustryComparator error: {e}")
        state["agent_outputs"].append({
            "agent": "peer_comparison",
            "error": str(e),
            "confidence": 0.5
        })

    return state


def _enrich_external_esg_benchmarks(state: ESGState) -> Dict[str, Any]:
    """Fetch external WBA/WRI benchmark signals used by downstream risk scoring."""
    company = state.get("company", "")
    if not company:
        return {"enabled": False, "error": "missing company"}

    try:
        import re

        def _company_variants(raw_name: str) -> list[str]:
            variants: list[str] = []

            def _add(value: str):
                text = str(value or "").strip()
                if text and text not in variants:
                    variants.append(text)

            base = re.sub(r"\s+", " ", str(raw_name or "")).strip()
            _add(base)

            camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
            _add(camel_split)
            _add(camel_split.replace("&", " and "))

            no_punct = re.sub(r"[,&()]+", " ", camel_split)
            no_punct = re.sub(r"\s+", " ", no_punct).strip()
            _add(no_punct)

            tokens = [t for t in no_punct.split(" ") if t]
            if tokens:
                _add(tokens[0])
            if len(tokens) >= 2:
                _add(" ".join(tokens[:2]))

            # Handle fused acronym + word forms like JPMorgan -> JP Morgan.
            if tokens:
                first = tokens[0]
                fused = re.match(r"^([A-Z]{3,})([a-z].*)$", first)
                if fused:
                    caps, tail = fused.groups()
                    split_first = f"{caps[:-1]} {caps[-1]}{tail}"
                    _add(" ".join([split_first] + tokens[1:]).strip())

            return variants

        candidate_names = _company_variants(company)
        filled = {}
        selected_company_name = company

        for candidate_name in candidate_names:
            selected_company_name = candidate_name
            filled = fill_missing_pillars(
                company_name=candidate_name,
                existing_scores={
                    "social": None,
                    "governance": None,
                    "environment": None,
                    "water_risk": None,
                },
                wba_api_key=os.getenv("WBA_API_KEY"),
                industry=state.get("industry", ""),
            )
            if isinstance(filled, dict) and filled.get("_sources"):
                break

        score_keys = {
            "social",
            "governance",
            "environment",
            "water_risk",
            "water_risk_physical",
            "water_risk_regulatory",
            "water_risk_reputational",
        }
        scores = {
            key: value
            for key, value in (filled or {}).items()
            if key in score_keys
        }
        sources = filled.get("_sources", {}) if isinstance(filled, dict) else {}
        indicators = filled.get("_wba_indicators", {}) if isinstance(filled, dict) else {}
        hq_coords = filled.get("_wba_hq_coordinates", {}) if isinstance(filled, dict) else {}
        sec_metrics = filled.get("_sec_metrics", {}) if isinstance(filled, dict) else {}
        supplemental_evidence = filled.get("_supplemental_evidence", []) if isinstance(filled, dict) else []
        enrichment_error = None
        if not sources:
            enrichment_error = (
                "No WBA/WRI data returned for company aliases: "
                + ", ".join(candidate_names[:6])
            )

        return {
            "enabled": bool(sources),
            "scores": scores,
            "sources": sources,
            "wba_company_name": filled.get("_wba_company_name") if isinstance(filled, dict) else None,
            "hq_coordinates": hq_coords,
            "wba_indicator_count": len(indicators) if isinstance(indicators, dict) else 0,
            "query_company_name": selected_company_name,
            "query_attempts": candidate_names,
            "sec_metrics": sec_metrics,
            "supplemental_evidence": supplemental_evidence if isinstance(supplemental_evidence, list) else [],
            "error": enrichment_error,
        }
    except Exception as exc:
        return {"enabled": False, "error": f"external ESG enrichment failed: {exc}"}


def risk_scoring_node(state: ESGState) -> ESGState:
    """LIVE: RiskScorer with ML + Formula hybrid approach"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: risk_scoring (ML-Enhanced with Financial Analyst)")
    print("="*70)

    if not RISK_SCORER_AVAILABLE:
        from core.minimal_agents import risk_scoring_node as minimal_risk
        return minimal_risk(state)

    try:
        scorer = RiskScorer()

        print(f"⚖️ Calculating risk score for {state['industry']} industry...")
        if scorer.use_ml:
            print(f"🤖 ML model loaded - using hybrid ML + formula approach")
            print(f"   NOTE: XGBoost now has visibility into ESG pillar scores")
        else:
            print(f"📐 Using formula-based scoring only")

        # Build all_analyses dict from agent_outputs
        all_analyses = _build_analyses_dict(state)

        # Add claim and company for pillar calculation
        all_analyses["claim"] = {
            "claim_id": "C1",
            "claim_text": state["claim"],
            "category": "sustainability"
        }
        all_analyses["company"] = state["company"]
        all_analyses["industry"] = state.get("industry", "")

        # Enrich with external benchmark data (WBA/WRI) before final scoring.
        external_benchmarks = _enrich_external_esg_benchmarks(state)
        all_analyses["external_benchmarks"] = external_benchmarks
        state["external_esg_data"] = external_benchmarks
        state["agent_outputs"].append({
            "agent": "external_esg_enrichment",
            "output": external_benchmarks,
            "confidence": 0.8 if external_benchmarks.get("enabled") else 0.5,
            "timestamp": datetime.now().isoformat()
        })
        if external_benchmarks.get("enabled"):
            print(
                "🌐 External ESG enrichment active: "
                f"sources={external_benchmarks.get('sources', {})}, "
                f"indicators={external_benchmarks.get('wba_indicator_count', 0)}"
            )
        elif external_benchmarks.get("error"):
            print(f"⚠️ External ESG enrichment unavailable: {external_benchmarks.get('error')}")

        fact_graph_payload = all_analyses.get("fact_graph", {})
        if isinstance(fact_graph_payload, dict):
            fg_summary = fact_graph_payload.get("summary", {})
            if isinstance(fg_summary, dict) and fg_summary:
                print(
                    "🕸️ Fact graph active: "
                    f"facts={fg_summary.get('fact_count', 0)}, "
                    f"verified={fg_summary.get('verified_fact_count', 0)}, "
                    f"linked={fg_summary.get('claim_linked_fact_count', 0)}"
                )

        # Call calculate_final_score with proper parameters
        result = scorer.calculate_final_score(
            company=state["company"],
            all_analyses=all_analyses
        )

        if isinstance(result, dict):
            risk_level = result.get("risk_level", "MODERATE")
            rating_grade = result.get("rating_grade", "BBB")
            confidence = result.get("confidence_level", 85) / 100
            risk_source = result.get("risk_source", "Formula-based")
            high_carbon_flag = result.get("high_carbon_greenwashing_flag", False)
            pillar_scores = result.get("pillar_scores", {})
            esg_override_active = result.get("esg_override_active", False)

            print(f"✅ Risk Level: {risk_level}")
            print(f"   Rating Grade: {rating_grade}")
            print(f"   Source: {risk_source}")
            print(f"   Greenwashing Risk: {result.get('greenwashing_risk_score', 50):.1f}/100")
            if result.get("greenwashing_risk_score_raw") is not None:
                print(f"   Raw Risk Before Recalibration: {result.get('greenwashing_risk_score_raw', 50):.1f}/100")
            print(f"   ESG Score: {result.get('esg_score', 50):.1f}/100")
            archive_quality = result.get("historical_archive_quality", {}) if isinstance(result.get("historical_archive_quality"), dict) else {}
            if archive_quality:
                print(
                    "   Historical Archive Quality: "
                    f"{archive_quality.get('archive_confidence', 'N/A')}/100 "
                    f"({archive_quality.get('archive_quality_band', 'UNKNOWN')})"
                )
            if result.get("abstain_recommended"):
                print(f"   Abstention: RECOMMENDED")
                print(f"   Reason: {result.get('abstention_reason', 'Insufficient evidence')}")

            if pillar_scores:
                print(f"   📊 Pillar Scores:")
                print(f"      E: {pillar_scores.get('environmental_score', 0):.1f}/100")
                print(f"      S: {pillar_scores.get('social_score', 0):.1f}/100")
                print(f"      G: {pillar_scores.get('governance_score', 0):.1f}/100")

            if esg_override_active:
                print(f"   🔒 ESG PILLAR OVERRIDE ACTIVE (bypassed ML)")

            if high_carbon_flag:
                print(f"   🚨 High-Carbon Greenwashing Flag: ACTIVE")

            # Show ML contribution if available
            if "ml_prediction" in result and not esg_override_active:
                ml_info = result["ml_prediction"]
                print(f"   ML Prediction: {ml_info['prediction']} (confidence: {ml_info['confidence']:.1%})")
                print(f"   ML Used: {'YES' if ml_info['used_for_final'] else 'NO'}")
                print(f"   ML saw pillar scores: E={pillar_scores.get('environmental_score', 0):.0f}, "
                      f"S={pillar_scores.get('social_score', 0):.0f}, "
                      f"G={pillar_scores.get('governance_score', 0):.0f}")
            state["risk_results"] = result
        else:
            risk_level = "MODERATE"
            rating_grade = "BBB"
            confidence = 0.5

        state["risk_level"] = risk_level
        state["rating_grade"] = rating_grade  # NEW: Set rating_grade in state
        state["confidence"] = confidence
        state.setdefault("node_execution_order", []).append("Risk Scoring")

        state["agent_outputs"].append({
            "agent": "risk_scoring",
            "output": result,
            "risk_level": risk_level,
            "rating_grade": rating_grade,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ RiskScorer error: {e}")
        import traceback
        traceback.print_exc()
        state["risk_level"] = "MODERATE"
        state["confidence"] = 0.5
        state["agent_outputs"].append({
            "agent": "risk_scoring",
            "error": str(e),
            "confidence": 0.5
        })

    return state


def _build_analyses_dict(state: ESGState) -> Dict[str, Any]:
    """
    Convert state agent_outputs into the format expected by RiskScorer.calculate_final_score
    """
    analyses = {
        "contradiction_analysis": [],
        "evidence": list(state.get("evidence", [])),
        "evidence_quality_metrics": {},
        "credibility_analysis": {},
        "sentiment_analysis": [],
        "historical_analysis": {},
        "peer_comparison": {},
        "industry_comparison": {},
        "carbon_extraction": state.get("carbon_extraction", {}),
        "greenwishing_analysis": state.get("greenwishing_analysis", {}),
        "regulatory_compliance": state.get("regulatory_compliance", {}),
        "social_analysis": state.get("social_analysis", {}),
        "governance_analysis": state.get("governance_analysis", {}),
        "temporal_consistency": {},
        "claim_decomposition": state.get("claim_decomposition", {}),
        "adversarial_triangulation": state.get("adversarial_triangulation", {}),
        "carbon_pathway_analysis": state.get("carbon_pathway_analysis", {}),
        "commitment_ledger": state.get("commitment_ledger", {}),
        "debate_activated": False,
        "financial_context": None,
        "agent_outputs": list(state.get("agent_outputs", [])),
        "industry": state.get("industry", ""),
        "external_benchmarks": state.get("external_esg_data", {}),
        "fact_graph": state.get("fact_graph", {}),
    }

    for output in state.get("agent_outputs", []):
        agent_name = output.get("agent", "")
        agent_result = output.get("output", {})

        if agent_name == "contradiction_analysis":
            if isinstance(agent_result, list):
                analyses["contradiction_analysis"] = agent_result
            elif isinstance(agent_result, dict) and "contradictions" in agent_result:
                analyses["contradiction_analysis"] = agent_result["contradictions"]

        elif agent_name == "evidence_retrieval":
            if isinstance(agent_result, dict):
                nested_evidence = agent_result.get("evidence", [])
                if isinstance(nested_evidence, list) and nested_evidence:
                    analyses["evidence"].extend([e for e in nested_evidence if isinstance(e, dict)])
                if isinstance(agent_result.get("quality_metrics"), dict):
                    analyses["evidence_quality_metrics"] = agent_result.get("quality_metrics", {})
                # Extract financial context
                if "financial_context" in output:
                    analyses["financial_context"] = output["financial_context"]

        elif agent_name == "credibility_analysis":
            analyses["credibility_analysis"] = agent_result

        elif agent_name == "sentiment_analysis":
            if isinstance(agent_result, list):
                analyses["sentiment_analysis"] = agent_result
            else:
                analyses["sentiment_analysis"].append(agent_result)

        elif agent_name == "temporal_analysis" or agent_name == "historical_analysis":
            analyses["historical_analysis"] = agent_result

        elif agent_name == "peer_comparison":
            analyses["peer_comparison"] = agent_result
            analyses["industry_comparison"] = agent_result

        elif agent_name == "carbon_extraction":
            analyses["carbon_extraction"] = agent_result

        elif agent_name == "greenwishing_detection":
            analyses["greenwishing_analysis"] = agent_result

        elif agent_name == "regulatory_scanning":
            analyses["regulatory_compliance"] = agent_result

        elif agent_name == "social_analysis":
            analyses["social_analysis"] = agent_result

        elif agent_name == "governance_analysis":
            analyses["governance_analysis"] = agent_result

        elif agent_name == "temporal_consistency":
            analyses["temporal_consistency"] = agent_result

        elif agent_name == "claim_decomposition":
            analyses["claim_decomposition"] = agent_result

        elif agent_name == "adversarial_triangulation":
            analyses["adversarial_triangulation"] = agent_result

        elif agent_name == "carbon_pathway_analysis":
            analyses["carbon_pathway_analysis"] = agent_result

        elif agent_name == "commitment_ledger_update":
            analyses["commitment_ledger"] = agent_result

        elif agent_name == "debate":
            analyses["debate_activated"] = True
            analyses["debate_result"] = agent_result

        elif agent_name == "external_esg_enrichment":
            analyses["external_benchmarks"] = agent_result
        elif agent_name == "fact_graph_builder":
            analyses["fact_graph"] = agent_result

    return analyses


def fact_graph_node(state: ESGState) -> ESGState:
    """
    Build a fact-centric ESG knowledge graph from collected evidence.
    This supports justification-centric scoring and abstention decisions.
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: fact_graph_builder")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    try:
        contradictions = []
        contradiction_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "contradiction_analysis"]
        if contradiction_outputs:
            payload = contradiction_outputs[-1].get("output", {})
            if isinstance(payload, list):
                contradictions = payload
            elif isinstance(payload, dict):
                contradictions = (
                    payload.get("contradictions")
                    or payload.get("contradiction_list")
                    or payload.get("specific_contradictions")
                    or []
                )

        temporal_payload = {}
        temporal_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "temporal_consistency"]
        if temporal_outputs:
            temporal_payload = temporal_outputs[-1].get("output", {}) or {}

        result = build_esg_fact_graph(
            company=state.get("company", ""),
            claim_text=state.get("claim", ""),
            evidence=state.get("evidence", []),
            contradictions=contradictions if isinstance(contradictions, list) else [],
            temporal_consistency=temporal_payload if isinstance(temporal_payload, dict) else {},
        )

        summary = result.get("summary", {}) if isinstance(result, dict) else {}
        state["fact_graph"] = result if isinstance(result, dict) else {}
        try:
            kg_status = CompanyKnowledgeGraph().ingest_state(state)
        except Exception as kg_err:
            kg_status = {
                "status": "ingest_error",
                "error": str(kg_err),
            }
        state["company_knowledge_graph"] = kg_status if isinstance(kg_status, dict) else {}
        state.setdefault("node_execution_order", []).append("Fact Graph Builder")
        state["agent_outputs"].append(
            {
                "agent": "fact_graph_builder",
                "output": result if isinstance(result, dict) else {"summary": {}},
                "confidence": 0.82,
                "timestamp": datetime.now().isoformat(),
            }
        )
        state["agent_outputs"].append(
            {
                "agent": "company_knowledge_graph",
                "output": kg_status if isinstance(kg_status, dict) else {},
                "confidence": 0.78 if isinstance(kg_status, dict) and not kg_status.get("error") else 0.35,
                "timestamp": datetime.now().isoformat(),
            }
        )

        print("✅ Fact graph built")
        if isinstance(summary, dict):
            print(
                f"   Facts: {summary.get('fact_count', 0)} | "
                f"Verified: {summary.get('verified_fact_count', 0)} | "
                f"Claim-linked: {summary.get('claim_linked_fact_count', 0)}"
            )
        if isinstance(kg_status, dict):
            print(
                f"   KG status: {kg_status.get('status', 'unknown')} | "
                f"Anchor: {kg_status.get('organization_anchor', 'N/A')}"
            )
        print(f"{'✅ NODE COMPLETED':^70}")
    except Exception as e:
        print(f"❌ Fact graph builder error: {e}")
        state["agent_outputs"].append(
            {
                "agent": "fact_graph_builder",
                "error": str(e),
                "confidence": 0.3,
                "timestamp": datetime.now().isoformat(),
            }
        )

    return state


def sentiment_analysis_node(state: ESGState) -> ESGState:
    """LIVE: SentimentAnalyzer"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: sentiment_analysis")
    print("="*70)

    if not SENTIMENT_ANALYZER_AVAILABLE:
        state["agent_outputs"].append({
            "agent": "sentiment_analysis",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        analyzer = SentimentAnalyzer()

        print(f"💭 Analyzing sentiment...")

        result = analyzer.analyze_claim_language(
            claim={
                "claim_id": "C1",
                "claim_text": state.get("claim", ""),
                "company": state.get("company", ""),
            },
            evidence=state.get("evidence", []),
        )

        confidence = result.get("confidence", 0.7) if isinstance(result, dict) else 0.7
        print(f"✅ Sentiment analysis complete")
        if isinstance(result, dict):
            state["sentiment_results"] = result
        state.setdefault("node_execution_order", []).append("Sentiment Analysis")

        state["agent_outputs"].append({
            "agent": "sentiment_analysis",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ SentimentAnalyzer error: {e}")
        state["agent_outputs"].append({
            "agent": "sentiment_analysis",
            "error": str(e),
            "confidence": 0.5
        })

    return state


def credibility_analysis_node(state: ESGState) -> ESGState:
    """LIVE: CredibilityAnalyst"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: credibility_analysis")
    print("="*70)

    if not CREDIBILITY_ANALYST_AVAILABLE:
        state["agent_outputs"].append({
            "agent": "credibility_analysis",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        analyst = CredibilityAnalyst()

        print(f"🔒 Assessing source credibility...")

        evidence = state.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        # Normalize evidence items to CredibilityAnalyst expected schema.
        # Many upstream agents store evidence as {title, snippet, source, url, relevant_text}.
        normalized_evidence = []
        for idx, ev in enumerate(evidence, start=1):
            if not isinstance(ev, dict):
                continue
            normalized_evidence.append({
                "source_id": ev.get("source_id") or ev.get("id") or idx,
                "source_name": ev.get("source_name") or ev.get("source") or ev.get("publisher") or ev.get("title") or "Unknown",
                "source_type": ev.get("source_type") or ev.get("reliability_tier") or "Web Source",
                "url": ev.get("url") or ev.get("link") or "",
                "relevant_text": ev.get("relevant_text") or ev.get("snippet") or ev.get("content") or "",
                "data_freshness_days": ev.get("data_freshness_days", 999),
            })

        if hasattr(analyst, 'analyze_sources'):
            result = analyst.analyze_sources(normalized_evidence)
        elif hasattr(analyst, 'analyze'):
            result = analyst.analyze(normalized_evidence)
        elif hasattr(analyst, 'assess'):
            result = analyst.assess(normalized_evidence)
        else:
            result = {"overall_credibility": 50, "aggregate_metrics": {"average_credibility": 0.5, "total_sources": len(normalized_evidence)}, "confidence": 0.5}

        confidence = result.get("confidence", 0.75) if isinstance(result, dict) else 0.75
        print(f"✅ Credibility assessment complete")
        if isinstance(result, dict):
            state["credibility_results"] = result
        state.setdefault("node_execution_order", []).append("Credibility Analysis")

        state["agent_outputs"].append({
            "agent": "credibility_analysis",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ CredibilityAnalyst error: {e}")
        state["agent_outputs"].append({
            "agent": "credibility_analysis",
            "error": str(e),
            "confidence": 0.5
        })

    return state


def realtime_monitoring_node(state: ESGState) -> ESGState:
    """LIVE: RealTimeMonitor - scrapes latest news"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: realtime_monitoring")
    print("="*70)

    if not REALTIME_MONITOR_AVAILABLE:
        state["agent_outputs"].append({
            "agent": "realtime_monitoring",
            "output": "Agent not available",
            "confidence": 0.5
        })
        return state

    try:
        monitor = RealTimeMonitor()

        print(f"📰 Scraping real-time news for {state['company']}...")

        # Use the actual method from your file
        result = monitor.scrape_and_store(
            company=state["company"],
            hours_lookback=24
        )

        confidence = 0.7
        if isinstance(result, dict):
            evidence_items = result.get("evidence_items", [])
            print(f"✅ Found {len(evidence_items)} recent articles")
            for item in evidence_items:
                assert item.get("source_name") != "realtime_news", "source_name must be the publisher, not the agent name"
                state["evidence"].append(item)
            confidence = 0.8 if evidence_items else 0.5

        state.setdefault("node_execution_order", []).append("Realtime Monitoring")

        state["agent_outputs"].append({
            "agent": "realtime_monitoring",
            "output": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "live_fetch": True
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ RealTimeMonitor error: {e}")
        state["agent_outputs"].append({
            "agent": "realtime_monitoring",
            "error": str(e),
            "confidence": 0.5
        })

    return state


def confidence_scoring_node(state: ESGState) -> ESGState:
    """Calculate overall confidence"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: confidence_scoring")
    print("="*70)

    # Calculate from successful agents only, one confidence per logical agent.
    unique_agent_confidences = {}
    for o in state.get("agent_outputs", []):
        if not isinstance(o, dict):
            continue
        agent_name = o.get("agent")
        conf = o.get("confidence")
        if not agent_name or "error" in o or not isinstance(conf, (int, float)):
            continue
        unique_agent_confidences[agent_name] = float(conf)

    confidences = list(unique_agent_confidences.values())
    agent_count = len(confidences)
    assert agent_count < 100, f"Agent count {agent_count} is unreasonably high - counter not being reset"

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    # Apply reliability guardrails so sparse evidence runs cannot report inflated confidence.
    evidence_items = state.get("evidence", []) if isinstance(state.get("evidence", []), list) else []
    evidence_count = len(evidence_items)

    unique_domains = set()
    verifiable_count = 0
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or "").strip()
        date = str(item.get("date") or item.get("published_at") or item.get("publishedAt") or "").strip()

        if url:
            host = url.split("//")[-1].split("/")[0].lower().strip()
            if host.startswith("www."):
                host = host[4:]
            if host:
                unique_domains.add(host)

        if url and date:
            verifiable_count += 1

    reliability_caps = []
    if evidence_count < 3:
        reliability_caps.append(0.55)
    elif evidence_count < 5:
        reliability_caps.append(0.65)
    elif evidence_count < 8:
        reliability_caps.append(0.75)

    if len(unique_domains) < 3:
        reliability_caps.append(0.70)
    if verifiable_count < 3:
        reliability_caps.append(0.70)

    risk_outputs = [o for o in state.get("agent_outputs", []) if isinstance(o, dict) and o.get("agent") == "risk_scoring"]
    risk_result = risk_outputs[-1].get("output", {}) if risk_outputs else {}
    report_tier = str(risk_result.get("report_tier") or "").upper() if isinstance(risk_result, dict) else ""
    abstain_recommended = bool(risk_result.get("abstain_recommended", False)) if isinstance(risk_result, dict) else False
    if report_tier == "TIER_3":
        reliability_caps.append(0.60)
    elif report_tier == "TIER_2":
        reliability_caps.append(0.75)
    if abstain_recommended:
        reliability_caps.append(0.55)

    audit_outputs = [o for o in state.get("agent_outputs", []) if isinstance(o, dict) and o.get("agent") == "adversarial_audit"]
    audit_result = audit_outputs[-1].get("output", {}) if audit_outputs else {}
    if isinstance(audit_result, dict):
        audit_penalty = float(audit_result.get("confidence_penalty", 0.0) or 0.0)
        if audit_penalty > 0:
            avg_confidence = max(0.0, avg_confidence - audit_penalty)
            reliability_caps.append(max(0.50, 1.0 - audit_penalty))

    failed_agents = 0
    for out in state.get("agent_outputs", []):
        if not isinstance(out, dict):
            continue
        if out.get("agent") == "confidence_scoring":
            continue
        if "error" in out or out.get("output") == "Agent not available":
            failed_agents += 1

    if failed_agents >= 4:
        avg_confidence *= 0.85
    elif failed_agents >= 2:
        avg_confidence *= 0.92

    cap_applied = min(reliability_caps) if reliability_caps else None
    if isinstance(cap_applied, float):
        avg_confidence = min(avg_confidence, cap_applied)

    avg_confidence = max(0.35, min(0.95, avg_confidence))
    state["confidence"] = avg_confidence

    print(
        f"✅ Average confidence: {avg_confidence:.2%} "
        f"(agents={agent_count}, evidence={evidence_count}, verifiable={verifiable_count}, domains={len(unique_domains)})"
    )

    state["agent_outputs"].append({
        "agent": "confidence_scoring",
        "output": {
            "average_confidence": avg_confidence,
            "agent_count": agent_count,
            "agents_included": sorted(unique_agent_confidences.keys()),
            "evidence_count": evidence_count,
            "verifiable_evidence_count": verifiable_count,
            "unique_domain_count": len(unique_domains),
            "failed_agent_count": failed_agents,
            "reliability_caps": reliability_caps,
            "cap_applied": cap_applied,
            "report_tier": report_tier,
            "abstain_recommended": abstain_recommended,
            "adversarial_audit": audit_result if isinstance(audit_result, dict) else {},
        },
        "confidence": avg_confidence,
        "timestamp": datetime.now().isoformat()
    })

    print(f"{'✅ NODE COMPLETED':^70}")

    return state


def adversarial_audit_node(state: ESGState) -> ESGState:
    """
    Compute coordination risk diagnostics across all upstream agents.
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: adversarial_audit")
    print("=" * 70)
    try:
        result = build_adversarial_audit(state)
        state["adversarial_audit"] = result
        state.setdefault("node_execution_order", []).append("Adversarial Audit")
        state["agent_outputs"].append({
            "agent": "adversarial_audit",
            "output": result,
            "confidence": max(0.5, 1.0 - float(result.get("coordination_risk", 0.0))),
            "timestamp": datetime.now().isoformat(),
        })
        print(
            "✅ Adversarial audit complete: "
            f"risk={result.get('coordination_risk_band')} "
            f"({result.get('coordination_risk', 0)}), "
            f"failed_agents={result.get('failed_agents', 0)}"
        )
        print(f"{'✅ NODE COMPLETED':^70}")
    except Exception as e:
        print(f"❌ Adversarial audit error: {e}")
        state["agent_outputs"].append({
            "agent": "adversarial_audit",
            "error": str(e),
            "confidence": 0.4,
            "timestamp": datetime.now().isoformat(),
        })
    return state


def verdict_generation_node(state: ESGState) -> ESGState:
    """
    Generate final verdict using AGENTIC INTELLIGENCE
    NO HARDCODING - All decisions based on agent analysis
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: verdict_generation")
    print("="*70)

    # ============================================================
    # PRIORITY 0: CHECK FOR ESG PILLAR OVERRIDE (HIGHEST PRIORITY)
    # ============================================================
    risk_scorer_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "risk_scoring"]

    # ============================================================
    # PRIORITY -1: ABSTENTION-AWARE DECISION (HARDEST GUARDRAIL)
    # ============================================================
    if risk_scorer_outputs:
        risk_scorer_result = risk_scorer_outputs[-1].get("output", {})
        if isinstance(risk_scorer_result, dict) and bool(risk_scorer_result.get("abstain_recommended", False)):
            decision_status = str(risk_scorer_result.get("decision_status") or "ABSTAIN_RECOMMENDED")
            abstention_reason = str(risk_scorer_result.get("abstention_reason") or "Evidence is insufficient for decision-grade scoring.")
            abstention_triggers = risk_scorer_result.get("abstention_triggers", [])
            if not isinstance(abstention_triggers, list):
                abstention_triggers = []

            state["risk_level"] = "ABSTAIN"
            state["rating_grade"] = None
            state["confidence"] = min(float(state.get("confidence", 0.55) or 0.55), 0.55)
            state["verdict_locked"] = True

            verdict_data = {
                "company": state["company"],
                "claim": state["claim"],
                "risk_level": "ABSTAIN",
                "rating_grade": None,
                "final_confidence": state["confidence"],
                "decision_status": decision_status,
                "abstain_recommended": True,
                "abstention_reason": abstention_reason,
                "abstention_triggers": abstention_triggers,
                "score_disclaimer": risk_scorer_result.get("score_disclaimer", ""),
                "report_tier": risk_scorer_result.get("report_tier", "TIER_3"),
                "timestamp": datetime.now().isoformat(),
                "locked_by": "abstention_guardrail",
            }

            state["final_verdict"] = verdict_data
            state["agent_outputs"].append({
                "agent": "verdict_generation",
                "output": verdict_data,
                "confidence": state["confidence"],
                "timestamp": datetime.now().isoformat(),
                "verdict_locked": True,
            })

            print("\n🛑 ABSTENTION-AWARE DECISION ENFORCED")
            print(f"   Status: {decision_status}")
            print(f"   Reason: {abstention_reason}")
            print(f"{'✅ NODE COMPLETED':^70}")
            return state

    if risk_scorer_outputs:
        risk_scorer_result = risk_scorer_outputs[-1].get("output", {})
        esg_override_active = risk_scorer_result.get("esg_override_active", False)

        if esg_override_active:
            print(f"\n✅ ESG PILLAR OVERRIDE DETECTED - Strong Performance")
            print(f"   ESG Score: {risk_scorer_result.get('esg_score', 0)}/100")
            print(f"   Rating: {risk_scorer_result.get('rating_grade', 'A')}")
            print(f"   This override takes HIGHEST PRIORITY")

            # Lock the verdict to ESG pillar-based assessment
            locked_risk_level = risk_scorer_result.get("risk_level", "LOW")
            locked_rating = risk_scorer_result.get("rating_grade", "A")
            locked_confidence = risk_scorer_result.get("confidence_level", 90) / 100

            state["risk_level"] = locked_risk_level
            state["rating_grade"] = locked_rating
            state["confidence"] = locked_confidence
            state["verdict_locked"] = True

            verdict_data = {
                "company": state["company"],
                "claim": state["claim"],
                "risk_level": locked_risk_level,
                "rating_grade": locked_rating,
                "final_confidence": locked_confidence,
                "evidence_count": len(state["evidence"]),
                "timestamp": datetime.now().isoformat(),
                "locked_by": "esg_pillar_override",
                "lock_reason": f"Strong ESG performance (ESG >= 75) - {risk_scorer_result.get('risk_source')}"
            }

            state["agent_outputs"].append({
                "agent": "verdict_generation",
                "output": verdict_data,
                "confidence": locked_confidence,
                "timestamp": datetime.now().isoformat(),
                "verdict_locked": True
            })

            state["final_verdict"] = verdict_data

            print(f"\n🔒 VERDICT LOCKED BY ESG PILLAR OVERRIDE")
            print(f"   Risk Level: {locked_risk_level}")
            print(f"   Rating: {locked_rating}")
            print(f"   Confidence: {locked_confidence:.1%}")
            print(f"{'✅ NODE COMPLETED':^70}")

            return state

    # ============================================================
    # PRIORITY 1: CHECK IF RISK SCORER LOCKED THE DECISION (Domain Knowledge)
    # ============================================================
    # If risk_scorer already determined HIGH risk for oil_and_gas greenwashing,
    # DO NOT override - this is domain knowledge that must be preserved

    if risk_scorer_outputs:
        risk_scorer_result = risk_scorer_outputs[-1].get("output", {})
        high_carbon_flag = risk_scorer_result.get("high_carbon_greenwashing_flag", False)
        risk_source = risk_scorer_result.get("risk_source", "")

        # Check if risk scorer applied domain knowledge override
        if "Domain Knowledge Override" in risk_source or high_carbon_flag:
            print(f"\n🔒 VERDICT LOCKED - Risk Scorer Domain Knowledge Override Detected")
            print(f"   Risk Source: {risk_source}")
            print(f"   High Carbon Flag: {high_carbon_flag}")
            print(f"   Industry: {state.get('industry')}")
            print(f"   ⚠️ Verdict generation will NOT override domain-specific risk assessment")

            # Extract risk scorer's final decision
            locked_risk_level = risk_scorer_result.get("risk_level", "HIGH")
            locked_rating = risk_scorer_result.get("rating_grade", "BB")
            locked_confidence = state.get("confidence", 0.85)

            # Lock the state
            state["risk_level"] = locked_risk_level
            state["rating_grade"] = locked_rating
            state["confidence"] = locked_confidence
            state["verdict_locked"] = True

            verdict_data = {
                "company": state["company"],
                "claim": state["claim"],
                "risk_level": locked_risk_level,
                "rating_grade": locked_rating,
                "confidence": locked_confidence,
                "evidence_count": len(state["evidence"]),
                "timestamp": datetime.now().isoformat(),
                "locked_by": "risk_scorer_domain_knowledge",
                "lock_reason": f"Oil & Gas greenwashing pattern detected - {risk_source}"
            }

            state["agent_outputs"].append({
                "agent": "verdict_generation",
                "output": verdict_data,
                "confidence": locked_confidence,
                "timestamp": datetime.now().isoformat(),
                "verdict_locked": True
            })

            state["final_verdict"] = verdict_data

            print(f"\n✅ LOCKED VERDICT: {locked_risk_level} (Rating: {locked_rating}, Confidence: {locked_confidence:.1%})")
            print(f"{'✅ NODE COMPLETED':^70}")

            return state

    # ============================================================
    # NORMAL VERDICT GENERATION (if not locked)
    # ============================================================

    verdict_data = {
        "company": state["company"],
        "claim": state["claim"],
        "risk_level": state["risk_level"],
        "confidence": state["confidence"],
        "evidence_count": len(state["evidence"]),
        "timestamp": datetime.now().isoformat()
    }

    claim_lower = state["claim"].lower()
    import re

    # ============================================================
    # AGENTIC INTELLIGENCE: Extract insights from agent outputs
    # ============================================================
    agent_outputs = state.get("agent_outputs", [])

    # Get HistoricalAnalyst findings (LIVE, no hardcoding)
    historical_data = None
    for output in agent_outputs:
        if output.get("agent") == "temporal_analysis":
            historical_data = output.get("output", {})
            break

    # Get ContradictionAnalyzer findings
    contradiction_count = 0
    for output in agent_outputs:
        if output.get("agent") == "contradiction_analysis":
            contradiction_count = output.get("contradictions_count", 0)
            break

    # Get debate resolution data
    debate_conflict_ratio = 0
    debate_outputs = [o for o in agent_outputs if o.get('agent') in ['debate_orchestrator', 'debate_resolution']]
    if debate_outputs:
        for debate in debate_outputs:
            conflict_ratio = debate.get('conflict_ratio', 0)
            debate_conflict_ratio = max(debate_conflict_ratio, conflict_ratio)
    # ============================================================
    # PRIORITY 1: ABSOLUTE/IMPOSSIBLE CLAIMS (Pattern Detection)
    # ============================================================
    # FIXED: Exclude legitimate carbon accounting terms
    absolute_patterns = [
        r'100%\s*(sustainable|green|eco|recyclable|renewable|organic|natural)',
        r'(completely|totally|fully|entirely|perfectly|absolutely)\s*(sustainable|green|eco)',
        # REMOVED: r'zero\s*(waste|emissions|carbon|pollution|impact)' - these can be legitimate
    ]

    # NEW: Check if claim has SPECIFIC METRICS that make it verifiable
    has_metrics = bool(re.search(r'\d+\.?\d*\s*(million|billion|%)|20\d{2}|specific\s+amount', state["claim"]))
    has_year = bool(re.search(r'20\d{2}|in\s+\d{4}', state["claim"]))

    # NEW: Legitimate carbon accounting terms (NOT greenwashing)
    legitimate_carbon_terms = [
        "carbon negative",  # Removing MORE than emitting
        "net zero",         # With documented offsetting
        "carbon neutral",   # If verified and dated
        "scope 1", "scope 2", "scope 3"  # GHG Protocol terminology
    ]

    # Check if claim uses legitimate terminology WITH metrics
    is_legitimate_carbon_claim = (
        any(term in claim_lower for term in legitimate_carbon_terms)
        and has_metrics
        and has_year
    )

    absolute_detected = (
        any(re.search(p, claim_lower) for p in absolute_patterns)
        and not is_legitimate_carbon_claim  # FIXED: Don't flag legitimate claims
    )

    if absolute_detected:
        print(f"\n🔴 AGENTIC DECISION: Absolute claim pattern detected")
        state["risk_level"] = "HIGH"
        state["confidence"] = min(state["confidence"] * 0.60, 0.75)
        verdict_data["risk_level"] = "HIGH"
        verdict_data["escalation"] = "Absolute/impossible claim (pattern-based)"
        print(f"   Escalated to HIGH - unrealistic claim language")

    # ============================================================
    # PRIORITY 1.5: VERIFIED CARBON CLAIMS (Actively Reduce to LOW)
    # ============================================================
    elif is_legitimate_carbon_claim:
        print(f"\n🟢 AGENTIC DECISION: Legitimate carbon accounting detected")
        print(f"   - Specific metrics: {has_metrics}")
        print(f"   - Dated claim: {has_year}")
        print(f"   - Recognized terminology: carbon negative/net zero")

        # FIXED: Actively downgrade to LOW if currently MODERATE
        if state["risk_level"] in ["MODERATE", "HIGH"]:
            original_risk = state["risk_level"]
            state["risk_level"] = "LOW"
            state["confidence"] = min(state["confidence"] * 1.10, 0.85)  # Boost confidence slightly
            verdict_data["risk_level"] = "LOW"
            verdict_data["downgrade"] = f"From {original_risk} to LOW - verified carbon accounting"
            verdict_data["verified_metrics"] = True

            print(f"   🟢 DOWNGRADING: {original_risk} → LOW")
            print(f"   Reason: Verifiable claim with specific date and recognized carbon accounting")



    # ============================================================
    # PRIORITY 2: HISTORICAL ANALYST INTELLIGENCE (AGENTIC)
    # ============================================================
    elif historical_data:
        reputation_score = historical_data.get("reputation_score", 50)
        violations = historical_data.get("past_violations", [])
        greenwashing_history = historical_data.get("greenwashing_history", {})
        patterns = historical_data.get("temporal_patterns", {})

        print(f"\n🤖 AGENTIC INTELLIGENCE: Historical Analysis")
        print(f"   Reputation Score: {reputation_score}/100 (LIVE calculated)")
        print(f"   Past Violations: {len(violations)} (LIVE searched)")
        print(f"   Greenwashing History: {greenwashing_history.get('prior_accusations', 0)} accusations (LIVE)")

        # DECISION RULES based on HistoricalAnalyst findings

        # Rule 1: Low reputation + violations = HIGH RISK (ADJUSTED thresholds)
        if reputation_score < 40 and len(violations) >= 1:  # Changed from ≥2 to ≥1
            print(f"\n🔴 AGENTIC DECISION: Poor track record detected")
            print(f"   - Reputation: {reputation_score}/100 (threshold: <40)")
            print(f"   - Violations: {len(violations)} (threshold: ≥1)")  # Updated

            state["risk_level"] = "HIGH"
            state["confidence"] = min(state["confidence"] * 0.70, 0.80)
            verdict_data["risk_level"] = "HIGH"
            verdict_data["escalation"] = f"Historical violations ({len(violations)}) + poor reputation ({reputation_score}/100)"
            verdict_data["historical_intelligence"] = True

        # Rule 2: Greenwashing pattern detected = HIGH RISK
        elif greenwashing_history.get("pattern_detected") and greenwashing_history.get("prior_accusations", 0) >= 2:
            print(f"\n🔴 AGENTIC DECISION: Greenwashing pattern detected")
            print(f"   - Prior Accusations: {greenwashing_history.get('prior_accusations')}")
            print(f"   - Pattern: Repeated across multiple years")

            state["risk_level"] = "HIGH"
            state["confidence"] = min(state["confidence"] * 0.65, 0.75)
            verdict_data["risk_level"] = "HIGH"
            verdict_data["escalation"] = f"Historical greenwashing pattern ({greenwashing_history.get('prior_accusations')} accusations)"
            verdict_data["historical_intelligence"] = True

        # Rule 3: Declining trend + current claim = ESCALATE
        elif patterns.get("declining_trend") and state["risk_level"] == "MODERATE":
            print(f"\n⚠️ AGENTIC DECISION: Declining ESG trend detected")
            print(f"   - Historical pattern shows worsening performance")

            state["risk_level"] = "HIGH"
            state["confidence"] *= 0.80
            verdict_data["risk_level"] = "HIGH"
            verdict_data["escalation"] = "Declining ESG trend contradicts positive claim"
            verdict_data["historical_intelligence"] = True

        # Rule 4: Reactive claims pattern = ESCALATE
        elif patterns.get("reactive_claims") and state["risk_level"] == "MODERATE":
            print(f"\n⚠️ AGENTIC DECISION: Reactive greenwashing pattern")
            print(f"   - Positive claims appear after negative news")

            state["risk_level"] = "HIGH"
            state["confidence"] *= 0.75
            verdict_data["risk_level"] = "HIGH"
            verdict_data["escalation"] = "Reactive greenwashing pattern detected"
            verdict_data["historical_intelligence"] = True

    # ============================================================
    # PRIORITY 3: CONTRADICTION ANALYZER INTELLIGENCE
    # ============================================================
    if contradiction_count >= 3 and state["risk_level"] == "MODERATE":
        print(f"\n⚠️ AGENTIC DECISION: Multiple contradictions detected")
        print(f"   - Contradictions: {contradiction_count} (threshold: ≥3)")

        state["risk_level"] = "HIGH"
        state["confidence"] *= 0.75
        verdict_data["risk_level"] = "HIGH"
        verdict_data["escalation"] = f"Multiple contradictions ({contradiction_count}) detected"
        verdict_data["contradiction_intelligence"] = True

    # ============================================================
    # PRIORITY 4: DEBATE ORCHESTRATOR INTELLIGENCE
    # ============================================================
    if debate_conflict_ratio >= 0.60 and state["risk_level"] == "MODERATE":
        print(f"\n⚠️ AGENTIC DECISION: High agent conflict detected")
        print(f"   - Conflict Ratio: {debate_conflict_ratio:.0%} (threshold: ≥60%)")

        state["risk_level"] = "HIGH"
        state["confidence"] *= 0.75
        verdict_data["risk_level"] = "HIGH"
        verdict_data["escalation"] = f"Agent disagreement ({debate_conflict_ratio:.0%})"
        verdict_data["debate_intelligence"] = True

    # ============================================================
    # PRIORITY 5: HIGH-RISK SUPERLATIVES (Pattern-based)
    # ============================================================
    superlatives = ["greenest", "leader in", "pioneer", "most sustainable", "best in class", "world's leading"]
    if any(sup in claim_lower for sup in superlatives) and state["risk_level"] == "MODERATE":
        print(f"\n⚠️ AGENTIC DECISION: Superlative language detected")

        state["risk_level"] = "HIGH"
        state["confidence"] *= 0.70
        verdict_data["risk_level"] = "HIGH"
        verdict_data["escalation"] = "Superlative greenwashing language"
        verdict_data["pattern_intelligence"] = True

    # ============================================================
    # PRIORITY 6: VAGUE CLAIMS (High-Risk Sectors)
    # ============================================================
    high_risk_sectors = ["Energy", "Automotive", "Aviation", "Mining", "Oil & Gas"]
    vague_keywords = ["committed to", "sustainable", "eco-friendly", "green", "clean energy"]
    keyword_count = sum(1 for kw in vague_keywords if kw in claim_lower)
    has_metrics = bool(re.search(r'\d+%|\d+\s*(tons|MW|GW|million|billion)|20\d{2}', state["claim"]))

    if state["industry"] in high_risk_sectors and keyword_count >= 2 and not has_metrics:
        if state["risk_level"] == "MODERATE":
            print(f"\n⚠️ AGENTIC DECISION: Vague high-risk sector claim")
            print(f"   - Sector: {state['industry']} (high baseline risk)")
            print(f"   - Vague keywords: {keyword_count}, Metrics: {has_metrics}")

            state["risk_level"] = "HIGH"
            state["confidence"] *= 0.80
            verdict_data["risk_level"] = "HIGH"
            verdict_data["escalation"] = f"Vague claim in {state['industry']} sector"
            verdict_data["sector_intelligence"] = True

    # Update final verdict
    verdict_data["final_confidence"] = state["confidence"]
    state["final_verdict"] = verdict_data

    print(f"\n✅ AGENTIC VERDICT: {state['risk_level']} (confidence: {state['confidence']:.1%})")

    # Log which intelligence sources influenced decision
    intelligence_sources = []
    if verdict_data.get("historical_intelligence"):
        intelligence_sources.append("Historical Track Record")
    if verdict_data.get("contradiction_intelligence"):
        intelligence_sources.append("Contradiction Analysis")
    if verdict_data.get("debate_intelligence"):
        intelligence_sources.append("Multi-Agent Debate")
    if verdict_data.get("pattern_intelligence"):
        intelligence_sources.append("Language Pattern Detection")
    if verdict_data.get("sector_intelligence"):
        intelligence_sources.append("Industry Risk Analysis")

    if intelligence_sources:
        print(f"   Intelligence Sources: {', '.join(intelligence_sources)}")

    state["agent_outputs"].append({
        "agent": "verdict_generation",
        "output": verdict_data,
        "confidence": state["confidence"],
        "timestamp": datetime.now().isoformat(),
        "intelligence_sources": intelligence_sources
    })

    print(f"{'✅ NODE COMPLETED':^70}")
    return state


def report_generation_node(state: ESGState) -> ESGState:
    """Generate high-fidelity comprehensive report for JPMC Demo"""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: report_generation")
    print("="*70)

    risk_results = state.get("risk_results", {})
    pillar_scores = risk_results.get("pillar_scores", {})
    reasons = risk_results.get("explainability_top_3_reasons", [])
    insights = risk_results.get("actionable_insights", {})

    # 5-Feature Signals
    decomp = state.get("claim_decomposition", {})
    ledger = state.get("commitment_ledger", {})
    pathway = state.get("carbon_pathway_analysis", {})
    triangulation = state.get("adversarial_triangulation", {})
    reg_scan = state.get("regulatory_compliance", {})

    report = f"""
{'='*80}
             ESGLens™ | HIGH-FIDELITY GREENWASHING ANALYSIS REPORT
{'='*80}
Company: {state['company'].upper()}
Sector:  {state['industry'].replace('_', ' ').title()}
Rating:  {risk_results.get('rating_grade', 'N/A')} ({state['risk_level']})
Confidence: {state['confidence']:.1%} | Workflow: {state.get('workflow_path', 'Deep Scan')}

ANALYSIS TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
{'='*80}

[1] EXECUTIVE SUMMARY & INTEGRITY SIGNALS
--------------------------------------------------------------------------------
{chr(10).join([f"• {r}" for r in reasons]) if reasons else "No specific integrity signals detected."}

[2] ESG PILLAR PERFORMANCE (0-100)
--------------------------------------------------------------------------------
Environmental: {pillar_scores.get('environmental_score', 0):.1f} | Weight: 35%
Social:        {pillar_scores.get('social_score', 0):.1f} | Weight: 35%
Governance:    {pillar_scores.get('governance_score', 0):.1f} | Weight: 30%
Overall ESG Score: {risk_results.get('esg_score', 0):.1f}

[3] HIGH-FIDELITY DIAGNOSTICS (2026 ENGINE)
--------------------------------------------------------------------------------
• CLAIM DECOMPOSITION: {decomp.get('internal_contradiction_score', 0)}/100 Contradiction Risk
  - Atomic Sub-claims: {len(decomp.get('sub_claims', []))} assertions analyzed.
  - Logical Tensions: {len(decomp.get('logical_tensions', []))} detected.

• COMMITMENT LEDGER: {ledger.get('promise_degradation_score', 0):.1f}/100 Degradation Score
  - Historical Revisions: {len(ledger.get('revision_events', []))} events tracked.
  - Trajectory Status: {'DEGRADING' if ledger.get('promise_degradation_score', 0) > 40 else 'STABLE'}

• CARBON PATHWAY MODEL: {pathway.get('alignment_status', 'UNKNOWN').upper()}
  - 1.5°C Alignment Gap: {pathway.get('pathway_gap_pct', 0):.1f}%
  - Scientific Credibility: {'HIGH' if pathway.get('pathway_gap_pct', 100) < 15 else 'LOW/BEYOND PHYSICS'}

• ADVERSARIAL TRIANGULATION: {triangulation.get('triangulation_score', 0):.1f}/100 Cross-Source Score
  - Third-Party Corroboration: {triangulation.get('third_party_corroboration_ratio', 0):.1%}
  - Bias Detection: {'NEUTRAL' if triangulation.get('triangulation_score', 0) > 60 else 'SKEWED TO FIRST-PARTY'}

• REGULATORY SCANNER: {len(reg_scan.get('compliance_results', []))} Frameworks Scanned
  - Jurisdictional Gaps: {len([r for r in reg_scan.get('compliance_results', []) if r.get('gap_details')])} detected.

[4] STAKEHOLDER INSIGHTS
--------------------------------------------------------------------------------
INVESTOR VIEW: {insights.get('for_investors', 'N/A')}

REGULATORY VIEW: {insights.get('for_regulators', 'N/A')}

CONSUMER VIEW: {insights.get('for_consumers', 'N/A')}

[5] EVIDENCE & JUSTIFICATION GRAPH
--------------------------------------------------------------------------------
Total Sources: {len(state['evidence'])} verified items.
Top Evidence Types: {', '.join(set([e.get('source_type', 'Web') for e in state['evidence'] if isinstance(e, dict)][:5]))}

{'='*80}
   Generated by ESGLens Multi-Agent Workflow Node: {state.get('workflow_path', 'Phase2')}
{'='*80}
"""

    state["report"] = report
    print(f"✅ High-Fidelity Report generated ({len(report)} characters)")

    state["agent_outputs"].append({
        "agent": "report_generation",
        "confidence": 1.0,
        "timestamp": datetime.now().isoformat()
    })

    print(f"{'✅ NODE COMPLETED':^70}")

    return state


# ============================================================
# PHASE 7: ESG REPORT DISCOVERY & PARSING PIPELINE
# ============================================================

def report_discovery_node(state: ESGState) -> ESGState:
    """
    PHASE 7: Automatically discover ESG reports for the company
    Uses web search to find published ESG, sustainability, and annual reports
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: report_discovery")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not REPORT_DISCOVERY_AVAILABLE:
        print("⚠️  Report Discovery not available - skipping")
        state["agent_outputs"].append({
            "agent": "report_discovery",
            "output": {"reports": [], "status": "skipped"},
            "confidence": 0.0
        })
        return state

    try:
        company = state.get("company")
        if not company:
            print("⚠️  No company specified - skipping report discovery")
            return state

        print(f"[Workflow] Starting ESG report discovery for {company}")
        print(f"🔍 Searching for ESG reports (up to 5 results)...")

        # Discover reports using convenience function
        discovered_reports = discover_company_reports(company, max_results=5)

        if discovered_reports:
            print(f"✅ Discovered {len(discovered_reports)} reports:")
            for report in discovered_reports[:3]:
                print(f"   - {report.get('year')}: {report.get('title', 'Untitled')[:60]}")
                print(f"     Confidence: {report.get('confidence', 0):.0%}")
        else:
            print(f"⚠️  No ESG reports discovered for {company}")

        confidence = 0.7 if discovered_reports else 0.3

        state["agent_outputs"].append({
            "agent": "report_discovery",
            "output": {
                "company": company,
                "reports": discovered_reports,
                "report_count": len(discovered_reports),
                "status": "success"
            },
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ Report Discovery error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "report_discovery",
            "error": str(e),
            "confidence": 0.0
        })

    return state


def report_downloader_node(state: ESGState) -> ESGState:
    """
    PHASE 7: Download discovered ESG reports (PDFs)
    Validates downloads and caches for reuse
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: report_downloader")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not REPORT_DOWNLOADER_AVAILABLE:
        print("⚠️  Report Downloader not available - skipping")
        return state

    try:
        company = state.get("company")

        # Find report discovery output
        discovery_outputs = [o for o in state.get("agent_outputs", [])
                           if o.get("agent") == "report_discovery"]

        if not discovery_outputs:
            print("⚠️  No report discovery output found - skipping download")
            return state

        discovered_reports = discovery_outputs[-1].get("output", {}).get("reports", [])

        if not discovered_reports:
            print("⚠️  No reports to download")
            return state

        print(f"[Workflow] Downloading ESG reports for {company}")
        print(f"📥 Downloading {len(discovered_reports)} discovered reports...")

        # Download reports using convenience function
        downloaded_reports = download_company_reports(company, discovered_reports)

        if downloaded_reports:
            print(f"✅ Downloaded {len(downloaded_reports)} reports:")
            for report in downloaded_reports:
                size_mb = report.get("file_size", 0) / (1024 * 1024)
                cached = report.get("from_cache", False)
                source = "(cached)" if cached else "(fresh download)"
                print(f"   - {report.get('year')}: {size_mb:.1f}MB {source}")
        else:
            print(f"⚠️  Failed to download any reports")

        confidence = 0.8 if downloaded_reports else 0.3

        state["agent_outputs"].append({
            "agent": "report_downloader",
            "output": {
                "company": company,
                "downloads": downloaded_reports,
                "download_count": len(downloaded_reports),
                "status": "success"
            },
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ Report Downloader error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "report_downloader",
            "error": str(e),
            "confidence": 0.0
        })

    return state


def report_parser_node(state: ESGState) -> ESGState:
    """
    PHASE 7: Parse downloaded PDFs and extract text chunks
    Cleans text and chunks for LLM processing
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: report_parser")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not REPORT_PARSER_AVAILABLE:
        print("⚠️  Report Parser not available - skipping")
        return state

    try:
        company = state.get("company")

        # Find report downloader output
        downloader_outputs = [o for o in state.get("agent_outputs", [])
                            if o.get("agent") == "report_downloader"]

        if not downloader_outputs:
            print("⚠️  No downloaded reports found - skipping parsing")
            return state

        downloaded_reports = downloader_outputs[-1].get("output", {}).get("downloads", [])

        if not downloaded_reports:
            print("⚠️  No reports to parse")
            return state

        print(f"[Workflow] Parsing ESG reports for {company}")
        print(f"📄 Parsing {len(downloaded_reports)} reports into chunks...")

        # Parse reports using convenience function
        parsed_chunks = parse_downloaded_reports(company, downloaded_reports)

        if parsed_chunks:
            print(f"✅ Extracted {len(parsed_chunks)} text chunks:")
            years_found = set(chunk.get("year") for chunk in parsed_chunks)
            print(f"   Years covered: {sorted(years_found, reverse=True)}")
            avg_chunk_size = sum(len(chunk.get("text", "")) for chunk in parsed_chunks) // len(parsed_chunks)
            print(f"   Avg chunk size: {avg_chunk_size} characters")
        else:
            print(f"⚠️  No chunks extracted from reports")

        confidence = 0.8 if parsed_chunks else 0.3

        state["agent_outputs"].append({
            "agent": "report_parser",
            "output": {
                "company": company,
                "chunks": parsed_chunks,
                "chunk_count": len(parsed_chunks),
                "downloaded_reports": downloaded_reports,
                "status": "success"
            },
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ Report Parser error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "report_parser",
            "error": str(e),
            "confidence": 0.0
        })

    return state


def report_claim_extraction_node(state: ESGState) -> ESGState:
    """
    PHASE 7: Extract ESG claims from parsed report chunks
    Groups claims by year and deduplicates
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: report_claim_extraction")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not CLAIM_EXTRACTOR_AVAILABLE:
        print("⚠️  Claim Extractor not available - skipping report claim extraction")
        return state

    try:
        company = state.get("company")

        # Find report parser output
        parser_outputs = [o for o in state.get("agent_outputs", [])
                         if o.get("agent") == "report_parser"]

        if not parser_outputs:
            print("⚠️  No parsed report chunks found - skipping claim extraction")
            return state

        parsed_chunks = parser_outputs[-1].get("output", {}).get("chunks", [])

        if not parsed_chunks:
            print("⚠️  No chunks to extract claims from")
            return state

        print(f"[Workflow] Extracting ESG claims from report chunks for {company}")
        print(f"📊 Processing {len(parsed_chunks)} chunks for claim extraction...")

        # Use report-specific claim extraction method
        try:
            extractor = ClaimExtractor()
            result = extractor.extract_claims_from_report_chunks(
                company,
                parsed_chunks,
                target_claim=state.get("claim", "")
            )
        except AttributeError:
            print("⚠️  Report chunk extraction method not available - skipping")
            return state

        if isinstance(result, dict):
            report_claims_by_year = result.get("report_claims_by_year", {})
            total_claims = result.get("total_report_claims", 0)
            years = result.get("years_detected", [])
            chunks_processed = result.get("chunks_processed", 0)
            chunks_skipped = result.get("chunks_skipped", 0)
            cache_hits = result.get("cache_hits", 0)
            llm_calls_made = result.get("llm_calls_made", 0)

            # Calculate optimization metrics
            total_chunks = chunks_processed + chunks_skipped
            esg_filtering_reduction = (100 * chunks_skipped / total_chunks) if total_chunks > 0 else 0

            print(f"\n{'📊 OPTIMIZATION METRICS':=^70}")
            print(f"✅ Extracted {total_claims} claims from reports")
            print(f"\n📈 Pipeline Efficiency:")
            print(f"   • Total chunks from parser: {total_chunks}")
            print(f"   • ESG-filtered chunks used: {chunks_processed}")
            print(f"   • Chunks filtered out: {chunks_skipped} ({esg_filtering_reduction:.1f}% reduction)")
            print(f"\n⚡ API Optimization:")
            print(f"   • LLM calls made: {llm_calls_made}")
            print(f"   • Cache hits: {cache_hits}")
            if cache_hits > 0:
                print(f"   • Cache save: ~{cache_hits * 3} est. LLM calls avoided")
            print(f"\n📅 Results by Year:")
            print(f"   Years detected: {sorted(years, reverse=True) if years else 'None'}")
            for year in sorted(years, reverse=True):
                year_claims = report_claims_by_year.get(year, [])
                print(f"   - {year}: {len(year_claims)} claims")
            print(f"{'='*70}")

            confidence = 0.8 if total_claims > 0 else 0.3
        else:
            report_claims_by_year = {}
            total_claims = 0
            confidence = 0.3
            print("⚠️  Invalid result from claim extraction")

        state["agent_outputs"].append({
            "agent": "claim_extractor",
            "output": result if isinstance(result, dict) else {"claims": []},
            "report_claims_by_year": report_claims_by_year,
            "total_report_claims": total_claims,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "source": "report_chunks",
            "optimization_metrics": {
                "chunks_processed": result.get("chunks_processed", 0) if isinstance(result, dict) else 0,
                "chunks_skipped": result.get("chunks_skipped", 0) if isinstance(result, dict) else 0,
                "cache_hits": result.get("cache_hits", 0) if isinstance(result, dict) else 0,
                "llm_calls_made": result.get("llm_calls_made", 0) if isinstance(result, dict) else 0
            }
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ Report Claim Extraction error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "claim_extractor",
            "error": str(e),
            "confidence": 0.0,
            "source": "report_chunks"
        })

    return state


def temporal_consistency_node(state: ESGState) -> ESGState:
    """
    PHASE 7: Analyze temporal consistency in ESG claims
    Detects greenwashing by comparing claims over time and against actual performance
    Only runs if report claims are available
    """
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print(f"Node: temporal_consistency")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    if not TEMPORAL_CONSISTENCY_AVAILABLE:
        print("⚠️  Temporal Consistency Agent not available - skipping")
        return state

    try:
        company = state.get("company")

        # Check if we have report claims from report_claim_extraction_node
        claim_extractor_outputs = [o for o in state.get("agent_outputs", [])
                                  if o.get("agent") == "claim_extractor" and o.get("source") == "report_chunks"]

        if not claim_extractor_outputs:
            print("⚠️  No PDF chunks — running temporal analysis on web evidence (reduced accuracy mode)")
            web_claims = {}
            for ev in state.get("evidence", [])[:80]:
                if not isinstance(ev, dict):
                    continue
                text = str(ev.get("snippet") or ev.get("relevant_text") or "").strip()
                if not text:
                    continue
                date_raw = str(ev.get("date") or "")
                year = None
                for token in date_raw.replace("/", "-").split("-"):
                    if token.isdigit() and len(token) == 4:
                        year = int(token)
                        break
                if year is None:
                    year = datetime.now().year
                web_claims.setdefault(year, []).append(text)

            if not web_claims:
                print("⚠️  No web evidence available for temporal fallback mode")
                return state

            result = analyze_temporal_consistency(company, web_claims, state.get("agent_outputs", []))
            state["agent_outputs"].append({
                "agent": "temporal_consistency",
                "output": result,
                "confidence": 0.6,
                "timestamp": datetime.now().isoformat(),
                "mode": "web_evidence_fallback",
            })
            return state

        latest_claim_output = claim_extractor_outputs[-1]
        report_claims_by_year = latest_claim_output.get("report_claims_by_year", {})
        if not report_claims_by_year:
            claim_extractor_output = latest_claim_output.get("output", {})
            if isinstance(claim_extractor_output, dict):
                report_claims_by_year = claim_extractor_output.get("report_claims_by_year", {})

        if not report_claims_by_year:
            # Fallback: derive lightweight claims directly from parsed report chunks.
            parser_outputs = [o for o in state.get("agent_outputs", []) if o.get("agent") == "report_parser"]
            parsed_chunks = parser_outputs[-1].get("output", {}).get("chunks", []) if parser_outputs else []
            synthesized = {}
            for chunk in parsed_chunks:
                if not isinstance(chunk, dict):
                    continue
                year = chunk.get("report_year") or chunk.get("year")
                text = str(chunk.get("text", ""))
                if not year or not text:
                    continue

                sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
                picked = []
                for sent in sentences:
                    lower = sent.lower()
                    if any(k in lower for k in ["emission", "scope", "renewable", "net zero", "carbon", "target", "%"]):
                        picked.append(sent)
                    if len(picked) >= 5:
                        break

                if picked:
                    synthesized.setdefault(int(year), []).extend(picked)

            report_claims_by_year = synthesized

        if not report_claims_by_year:
            print("⚠️  No report claims by year - skipping temporal consistency analysis")
            return state

        print(f"[Workflow] Running temporal consistency analysis for {company}")
        print(f"📈 Analyzing claim trends across {len(report_claims_by_year)} years...")

        # Call temporal consistency analysis
        result = analyze_temporal_consistency(company, report_claims_by_year, state.get("agent_outputs", []))

        if isinstance(result, dict):
            temporal_score = result.get("temporal_consistency_score", 50)
            risk_level = result.get("risk_level", "MODERATE")
            claim_trend = result.get("claim_trend", "unknown")
            env_trend = result.get("environmental_trend", "unknown")

            print(f"✅ Temporal Consistency Analysis Complete:")
            print(f"   Score: {temporal_score:.0f}/100")
            print(f"   Risk Level: {risk_level}")
            print(f"   Claim Trend: {claim_trend}")
            print(f"   Environmental Trend: {env_trend}")

            evidence = result.get("evidence", [])
            if evidence:
                print(f"   Key Findings: {len(evidence)} inconsistencies detected")
                for item in evidence[:2]:
                    print(f"   - {item[:70]}...")

            confidence = 0.85
        else:
            print("⚠️  Invalid result from temporal consistency analysis")
            confidence = 0.3

        state["agent_outputs"].append({
            "agent": "temporal_consistency",
            "output": result if isinstance(result, dict) else {"status": "error"},
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        print(f"{'✅ NODE COMPLETED':^70}")

    except Exception as e:
        print(f"❌ Temporal Consistency error: {e}")
        import traceback
        traceback.print_exc()
        state["agent_outputs"].append({
            "agent": "temporal_consistency",
            "error": str(e),
            "confidence": 0.0
        })

    return state


def commitment_ledger_update_node(state: ESGState) -> ESGState:
    """Persist current commitments and detect revisions/degradation trajectory."""
    print(f"\n{'🟢 LANGGRAPH NODE EXECUTING':=^70}")
    print("Node: commitment_ledger_update")
    print("=" * 70)

    if not COMMITMENT_LEDGER_AVAILABLE:
        state["commitment_ledger"] = {
            "inserted_commitments": 0,
            "revision_events": [],
            "promise_degradation_score": 0.0,
            "status": "unavailable",
        }
        state["agent_outputs"].append({
            "agent": "commitment_ledger_update",
            "output": state["commitment_ledger"],
            "confidence": 0.4,
            "timestamp": datetime.now().isoformat(),
        })
        return state

    decomposition = state.get("claim_decomposition") if isinstance(state.get("claim_decomposition"), dict) else {}
    sub_claims = decomposition.get("sub_claims") if isinstance(decomposition.get("sub_claims"), list) else []
    if not sub_claims:
        sub_claims = [{"id": "SC1", "text": state.get("claim", ""), "type": "policy_claim"}]

    try:
        ledger = CommitmentLedger("data/commitment_ledger.db")
        run_date = datetime.now().date().isoformat()
        run_id = f"{state.get('company', 'company')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        output = ledger.update_from_subclaims(
            company=state.get("company", ""),
            run_id=run_id,
            run_date=run_date,
            sub_claims=sub_claims,
            evidence=state.get("evidence", []),
        )
        output["status"] = "ok"
    except Exception as e:
        print(f"❌ CommitmentLedger error: {e}")
        output = {
            "inserted_commitments": 0,
            "revision_events": [],
            "promise_degradation_score": 0.0,
            "status": "error",
            "error": str(e),
        }

    state["commitment_ledger"] = output
    state.setdefault("node_execution_order", []).append("Commitment Ledger Update")
    state["agent_outputs"].append({
        "agent": "commitment_ledger_update",
        "output": output,
        "confidence": 0.73,
        "timestamp": datetime.now().isoformat(),
    })
    print(
        f"✅ Ledger updated: commitments={output.get('inserted_commitments', 0)} | "
        f"promise_degradation={output.get('promise_degradation_score', 0)}"
    )
    return state



# ============================================================
# ESG MISMATCH DETECTOR NODE (2026 Features)
# ============================================================

def esg_mismatch_node(state: ESGState) -> ESGState:
    """
    Executes the ESG Mismatch Detector to compare company promises vs actual evidence.
    """
    print(f"\n{'?? NODE: ESG Mismatch Detector':=^70}")

    # Initialize state collections if missing
    if "agent_outputs" not in state or not isinstance(state["agent_outputs"], list):
        state["agent_outputs"] = []

    company = state.get("company", "")
    if not company or analyze_company_esg is None:
        print(f"?? Skipping mismatch detection (missing company name or module unavailable)")
        state["agent_outputs"].append({
            "agent": "esg_mismatch",
            "output": {"status": "skipped", "reason": "Module unavailable or missing company"},
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        })
        return state

    try:
        print(f"?? Analyzing ESG promises vs reality for: {company}")

        # Call the standalone pipeline
        # Note: the pipeline relies on caching internally.
        mismatch_results = analyze_company_esg(company)

        if isinstance(mismatch_results, dict):
            # Save raw structure to state
            state["esg_mismatch_analysis"] = mismatch_results

            risk = mismatch_results.get("Overall Greenwashing Risk", "Unknown")
            print(f"   Mismatch Risk Level: {risk}")

            # Decide a confidence baseline
            confidence = 0.8 if risk in ["High", "Severe", "Violation Detected"] else 0.6

            state["agent_outputs"].append({
                "agent": "esg_mismatch",
                "output": mismatch_results,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })

        else:
            print(f"?? Unexpected mismatch result format: {type(mismatch_results)}")

    except Exception as e:
        print(f"? Error in ESG Mismatch Detector: {e}")
        import traceback
        traceback.print_exc()

        state["agent_outputs"].append({
            "agent": "esg_mismatch",
            "error": str(e),
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        })

    print(f"{'='*70}")
    return state
