import json
import asyncio
from typing import Dict, Any, List
from core.evidence_cache import evidence_cache
from config.agent_prompts import SOURCE_CREDIBILITY_PROMPT
from core.llm_call import call_llm

class CredibilityAnalyst:
    def __init__(self):
        self.name = "Source Credibility & Bias Analyst"
        
        # Base credibility scores by source type
        self.base_scores = {
            "Academic": 1.0,
            "Government/Regulatory": 0.95,
            "NGO": 0.90,
            "Tier-1 Financial Media": 0.85,
            "General Media": 0.70,
            "ESG Platform": 0.75,
            "Web Source": 0.50,
            "Company-Controlled": 0.35,
            "Sponsored Content": 0.20
        }
    
    def analyze_sources(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze credibility and bias of all evidence sources
        Works with evidence from cache (already fetched by EvidenceRetriever)
        """
        
        print(f"\n{'='*60}")
        print(f"🔍 AGENT 4: {self.name}")
        print(f"{'='*60}")
        print(f"Analyzing {len(evidence)} sources...")
        print(f"📦 Using evidence from cache (zero additional API calls)")
        
        
        source_analyses = []
        credibility_scores = []
        
        for i, ev in enumerate(evidence, 1):
            print(f"\r   Processing source {i}/{len(evidence)}...", end="", flush=True)
            
            analysis = self._analyze_single_source(ev)
            source_analyses.append(analysis)
            credibility_scores.append(analysis['final_credibility_score'])
        
        print(f"\n\n✅ Credibility analysis complete")
        
        # Calculate aggregate metrics
        avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0
        
        # Count by credibility tier
        high_credibility = sum(1 for s in credibility_scores if s >= 0.8)
        medium_credibility = sum(1 for s in credibility_scores if 0.5 <= s < 0.8)
        low_credibility = sum(1 for s in credibility_scores if s < 0.5)
        
        print(f"   Average credibility: {avg_credibility:.2f}")
        print(f"   High (≥0.8): {high_credibility} sources")
        print(f"   Medium (0.5-0.8): {medium_credibility} sources")
        print(f"   Low (<0.5): {low_credibility} sources")
        
        return {
            "source_credibility_analyses": source_analyses,
            "aggregate_metrics": {
                "average_credibility": avg_credibility,
                "high_credibility_count": high_credibility,
                "medium_credibility_count": medium_credibility,
                "low_credibility_count": low_credibility,
                "total_sources": len(evidence)
            },
            "overall_credibility": int(round(avg_credibility * 100)),
            "high_credibility_sources": [
                s.get("source_name")
                for s in source_analyses
                if s.get("final_credibility_score", 0) >= 0.8
            ][:10],
            "low_credibility_sources": [
                s.get("source_name")
                for s in source_analyses
                if s.get("final_credibility_score", 0) < 0.5
            ][:10],
            "unverifiable_sources": sum(1 for s in source_analyses if not (s.get("url") or "").strip() or s.get("url") == "#"),
            "credibility_warning": f"{low_credibility} of {len(evidence)} sources scored low credibility"
        }
    
    def _analyze_single_source(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single source for credibility and bias using LLM"""
        
        source_type = evidence.get('source_type', 'Web Source')
        source_name = evidence.get('source_name', 'Unknown')
        url = evidence.get('url', '')
        content = evidence.get('relevant_text', '')
        
        prompt = f"SOURCE: {source_name}\nTYPE: {source_type}\nURL: {url}\nCONTENT: {content[:2000]}"
        
        try:
            response = asyncio.run(call_llm("credibility_analysis", prompt, system=SOURCE_CREDIBILITY_PROMPT))
            if response:
                import re
                cleaned = re.sub(r'```\s*json?\s*', '', response)
                cleaned = re.sub(r'```\s*', '', cleaned)
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                if start != -1 and end > start:
                    llm_result = json.loads(cleaned[start:end])
                else:
                    llm_result = json.loads(cleaned)
            else:
                llm_result = {}
        except Exception as e:
            print(f"      ⚠️ LLM credibility analysis failed: {e}")
            llm_result = {}
            
        base_cred = llm_result.get("base_credibility", self.base_scores.get(source_type, 0.50))
        final_score = llm_result.get("final_credibility_score", base_cred)
        paid = llm_result.get("paid_content_detected", False)
        bias = llm_result.get("bias_direction", "Neutral")
        adjustments = llm_result.get("adjustments", [])
        
        return {
            "source_id": evidence.get('source_id'),
            "source_name": source_name,
            "source_type": source_type,
            "base_credibility": base_cred,
            "adjustments": adjustments,
            "final_credibility_score": round(final_score, 2),
            "paid_content_detected": paid,
            "bias_direction": bias,
            "url": url
        }
    
    def _detect_paid_content(self, content: str, source_name: str) -> bool:
        """Detect if content is sponsored/paid"""
        indicators = [
            "sponsored", "advertorial", "paid promotion", "partner content",
            "in partnership with", "brought to you by"
        ]
        
        text = (content + " " + source_name).lower()
        return any(indicator in text for indicator in indicators)
    
    def _detect_bias(self, content: str, source_type: str) -> str:
        """Simple bias detection"""
        if not content:
            return "Neutral"
        
        content_lower = content.lower()
        
        # Pro-company indicators
        pro_indicators = ["revolutionary", "groundbreaking", "leader", "best", "innovative"]
        pro_count = sum(1 for ind in pro_indicators if ind in content_lower)
        
        # Critical indicators
        critical_indicators = ["violation", "accused", "greenwashing", "overstated", "failed"]
        critical_count = sum(1 for ind in critical_indicators if ind in content_lower)
        
        if pro_count > critical_count + 2:
            return "Pro-company"
        elif critical_count > pro_count + 2:
            return "Anti-company"
        else:
            return "Neutral"
