
import sys
import os
import json

# Ensure we can import core and agents
sys.path.append(os.getcwd())

from agents.risk_scorer import RiskScorer

def test_risk_scorer_sec_integration():
    scorer = RiskScorer()
    
    company = "Apple"
    industry = "technology"
    
    # Mock analyses with SEC metrics
    all_analyses = {
        "company": company,
        "industry": industry,
        "evidence": [],
        "external_benchmarks": {
            "sec_metrics": {
                "board_diversity_pct": 35.0,
                "executive_pay_ratio": 200,
                "executive_comp_esg_links": True,
                "conflict_minerals_human_rights": True
            }
        },
        "contradiction_analysis": [],
        "historical_analysis": {},
        "credibility_analysis": {},
        "scores": {}
    }
    
    print(f"\n--- Testing RiskScorer with SEC metrics for {company} ---")
    result = scorer.calculate_final_score(all_analyses)
    
    print(f"\nFinal Result for {company}:")
    print(f"Governance Score: {result['pillar_scores']['governance_score']}")
    print(f"Social Score: {result['pillar_scores']['social_score']}")
    print(f"Top Reasons: {result['explainability_top_3_reasons']}")
    
    # Check if SEC metrics are in reasons
    has_diversity = any("board diversity" in r.lower() for r in result['explainability_top_3_reasons'])
    has_comp = any("executive compensation" in r.lower() for r in result['explainability_top_3_reasons'])
    
    if has_diversity and has_comp:
        print("\n✅ SUCCESS: SEC metrics reflected in explainability reasons.")
    else:
        print("\n❌ FAILURE: SEC metrics NOT found in explainability reasons.")

if __name__ == "__main__":
    test_risk_scorer_sec_integration()
