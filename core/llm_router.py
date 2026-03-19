from dataclasses import dataclass, field
from enum import Enum

class Provider(Enum):
    GEMINI     = "gemini"
    GROQ       = "groq"
    CEREBRAS   = "cerebras"
    OPENROUTER = "openrouter"

@dataclass
class ModelConfig:
    provider:     Provider
    model_id:     str
    json_mode:    bool  = False
    max_tokens:   int   = 1000
    temperature:  float = 0.1   # low temperature for ESG — determinism matters
    context_note: str   = ""    # why this model was chosen

# Helper constructors for readability
def Gemini(model, **kw):
    return ModelConfig(Provider.GEMINI, model, **kw)
def Groq(model, **kw):
    return ModelConfig(Provider.GROQ, model, **kw)
def Cerebras(model, **kw):
    return ModelConfig(Provider.CEREBRAS, model, **kw)
def OR(model, **kw):
    return ModelConfig(Provider.OPENROUTER, model, **kw)

# ═══════════════════════════════════════════════════════════
# ROUTING TABLE
# Format: "agent_name": [primary, fallback_1, fallback_2]
# First model tried = primary. On rate-limit/error → next.
# ═══════════════════════════════════════════════════════════

ROUTING_TABLE: dict[str, list[ModelConfig]] = {

    # ──────────────────────────────────────────────────────
    # STAGE 2: EXTRACTION
    # ──────────────────────────────────────────────────────

    "supervisor": [
        Groq("llama-3.3-70b-versatile",
             max_tokens=100, temperature=0.0,
             context_note="complexity assessment routing"),
        OR("deepseek/deepseek-r1:free",
           max_tokens=100),
        Gemini("gemini-2.0-flash",
               max_tokens=100),
    ],

    "carbon_extraction": [
        # Gemini Flash: only provider with native PDF multimodal.
        # Critical that unit conversion happens here (MtCO2e → tCO2e).
        Gemini("gemini-2.0-flash",
               max_tokens=1000, json_mode=True,
               context_note="multimodal PDF — exclusive to Gemini"),
        # Fallback: pass extracted text chunks if PDF bytes unavailable
        Groq("llama-3.3-70b-versatile",
             max_tokens=1000, json_mode=True,
             context_note="text-only fallback"),
        OR("google/gemini-2.5-pro-exp-03-25:free",
           max_tokens=1000, json_mode=True),
    ],

    "claim_extraction": [
        # Cerebras Scout: 516 chunks, heavily cached (llm_calls_made≈0),
        # fast model fine, 10M context handles long chunk batches.
        Cerebras("llama3.1-8b",
                 max_tokens=800, json_mode=True,
                 context_note="high chunk volume, cached, speed > intelligence"),
        Groq("llama-3.1-8b-instant",
             max_tokens=800, json_mode=True),
        OR("mistralai/mistral-small-3.1-24b-instruct:free",
           max_tokens=800, json_mode=True),
    ],

    "claim_extractor": [
        # Same as claim_extraction — structured JSON from text chunks.
        Cerebras("llama3.1-8b",
                 max_tokens=1000, json_mode=True,
                 context_note="35 claims across 3 years, structured parse"),
        Groq("llama-3.1-8b-instant",
             max_tokens=1000, json_mode=True),
        OR("mistralai/mistral-small-3.1-24b-instruct:free",
           max_tokens=1000, json_mode=True),
    ],

    # ──────────────────────────────────────────────────────
    # STAGE 4: ANALYSIS
    # ──────────────────────────────────────────────────────

    "contradiction_analysis": [
        # MOST CRITICAL TASK. Groq 70B: strong reasoning, fast, reliable.
        # This determines the greenwashing verdict — quality matters most.
        Groq("llama-3.3-70b-versatile",
             max_tokens=2000, temperature=0.1,
             context_note="most critical — determines verdict"),
        # Fallback 1: DeepSeek R1 chain-of-thought reasoning
        OR("deepseek/deepseek-r1:free",
           max_tokens=2000),
        # Fallback 2: Gemini Flash if both above fail
        Gemini("gemini-2.0-flash",
               max_tokens=2000,
               context_note="last resort — protects Gemini Pro quota"),
    ],

    "sentiment_analysis": [
        # 42 articles to analyse — speed critical. Cerebras Scout.
        # Sentiment on short snippets doesn't need deep reasoning.
        Cerebras("llama3.1-8b",
                 max_tokens=500,
                 context_note="42 articles per report, speed critical"),
        Groq("llama-3.1-8b-instant",
             max_tokens=500),
        OR("meta-llama/llama-4-scout:free",
           max_tokens=500),
    ],

    "credibility_analysis": [
        # CALLED ~42 TIMES PER REPORT. Absolute must use fastest provider.
        # Single source URL + snippet → credibility score + tier.
        # Cerebras is the only sensible choice here.
        Cerebras("llama3.1-8b",
                 max_tokens=200, json_mode=True,
                 context_note="called 42x per report — fastest model mandatory"),
        # Fallback: Groq 8B instant (still fast)
        Groq("llama-3.1-8b-instant",
             max_tokens=200, json_mode=True),
        OR("mistralai/mistral-small-3.1-24b-instruct:free",
           max_tokens=200, json_mode=True),
    ],

    "climatebert_analysis": [
        # Single short claim string → LOW/MODERATE/HIGH verdict.
        # Called once. Groq 8B instant is fast and sufficient.
        # Note: if you have ClimateBERT running locally via HuggingFace,
        # bypass this entirely — local model is free and purpose-built.
        Groq("llama-3.1-8b-instant",
             max_tokens=300, json_mode=True,
             context_note="single claim, short output, local model preferred"),
        Cerebras("llama3.1-8b",
                 max_tokens=300, json_mode=True),
        OR("microsoft/phi-4-reasoning:free",
           max_tokens=300, json_mode=True),
    ],

    "greenwishing_detection": [
        # Detects 4 deception tactics. Needs synthesis across
        # claims + evidence + carbon data. Groq 70B.
        Groq("llama-3.3-70b-versatile",
             max_tokens=1500, json_mode=True,
             context_note="multi-tactic synthesis across all evidence"),
        OR("deepseek/deepseek-r1:free",
           max_tokens=1500),
        Gemini("gemini-2.0-flash",
               max_tokens=1500, json_mode=True),
    ],

    "esg_mismatch": [
        # Pledge vs implementation gap analysis across 3 years.
        # Needs temporal reasoning — Groq 70B.
        Groq("llama-3.3-70b-versatile",
             max_tokens=1500, json_mode=True,
             context_note="temporal pledge-vs-implementation analysis"),
        OR("qwen/qwen3-235b-a22b:free",
           max_tokens=1500),
        Cerebras("llama-3.3-70b",
                 max_tokens=1500, json_mode=True),
    ],

    "temporal_analysis": [
        # Extracts past violations and reputation score.
        # Structured JSON output, company name as input.
        # Fast Cerebras 70B — heavier than Scout for factual recall.
        Cerebras("llama-3.3-70b",
                 max_tokens=800, json_mode=True,
                 context_note="factual recall of violations — 70B over Scout"),
        Groq("llama-3.1-8b-instant",
             max_tokens=800, json_mode=True),
        OR("meta-llama/llama-4-maverick:free",
           max_tokens=800, json_mode=True),
    ],

    "temporal_consistency": [
        # Claim trend vs performance trend scoring.
        # Moderate reasoning. Groq Gemma2 — good at structured analysis.
        Groq("gemma2-9b-it",
             max_tokens=600, json_mode=True,
             context_note="moderate reasoning, structured scoring"),
        Cerebras("llama-3.3-70b",
                 max_tokens=600, json_mode=True),
        OR("mistralai/mistral-small-3.1-24b-instruct:free",
           max_tokens=600, json_mode=True),
    ],

    "conflict_resolution": [
        Groq("llama-3.1-8b-instant",
             max_tokens=800, json_mode=True,
             context_note="fast contradiction rewriting"),
        OR("mistralai/mistral-small-3.1-24b-instruct:free",
           max_tokens=800, json_mode=True),
        Cerebras("llama3.1-8b",
                 max_tokens=800, json_mode=True),
    ],

    "financial_analysis": [
        Groq("llama-3.3-70b-versatile",
             max_tokens=1500, json_mode=True,
             context_note="financial ESG correlation reasoning"),
        OR("deepseek/deepseek-r1:free",
           max_tokens=1500),
        Cerebras("llama-3.3-70b",
                 max_tokens=1500, json_mode=True),
    ],

    # ──────────────────────────────────────────────────────
    # STAGE 5: SCORING
    # ──────────────────────────────────────────────────────

    "regulatory_scanning": [
        # Checks 6 frameworks (SEBI BRSR, SEC, GHG Protocol, 
        # GRI, CDP, SBTi). Needs precise rule-following + JSON.
        # Groq Gemma2: best at deterministic rule checks.
        Groq("gemma2-9b-it",
             max_tokens=2000, json_mode=True,
             context_note="6 frameworks, deterministic rule-following"),
        Groq("llama-3.3-70b-versatile",
             max_tokens=2000, json_mode=True),
        OR("mistralai/mistral-small-3.1-24b-instruct:free",
           max_tokens=2000, json_mode=True),
    ],

    "risk_scoring": [
        # Computes final greenwashing score, ESG score, pillar factors,
        # rating grade. Arithmetic must be correct. JSON must be exact.
        # Second most critical task — Groq 70B.
        Groq("llama-3.3-70b-versatile",
             max_tokens=2000, json_mode=True, temperature=0.0,
             context_note="final score computation — temp=0 for determinism"),
        OR("google/gemini-2.5-pro-exp-03-25:free",
           max_tokens=2000, json_mode=True),
        Gemini("gemini-2.0-flash",
               max_tokens=2000, json_mode=True),
    ],

    "peer_comparison": [
        # Retrieves industry peer ESG scores.
        # Factual recall task. Cerebras 70B (stronger than Scout
        # for factual knowledge retrieval).
        Cerebras("llama-3.3-70b",
                 max_tokens=1000, json_mode=True,
                 context_note="factual peer data recall"),
        Groq("llama-3.1-8b-instant",
             max_tokens=1000, json_mode=True),
        OR("meta-llama/llama-4-maverick:free",
           max_tokens=1000, json_mode=True),
    ],

    "confidence_scoring": [
        # Arithmetic aggregation of confidence scores across agents.
        # Trivially simple — fastest model that returns valid JSON.
        Cerebras("llama3.1-8b",
                 max_tokens=300, json_mode=True,
                 context_note="pure arithmetic aggregation, simplest task"),
        Groq("llama-3.1-8b-instant",
             max_tokens=300, json_mode=True),
        OR("mistralai/mistral-small-3.1-24b-instruct:free",
           max_tokens=300, json_mode=True),
    ],

    # ──────────────────────────────────────────────────────
    # STAGE 6: GENERATION
    # ──────────────────────────────────────────────────────

    "debate_orchestrator": [
        Groq("llama-3.3-70b-versatile",
             max_tokens=800, temperature=0.7,
             context_note="debate reasoning needs higher temperature for diverse thoughts"),
        OR("deepseek/deepseek-r1:free",
           max_tokens=800),
        Gemini("gemini-2.0-flash",
               max_tokens=800),
    ],

    "explainability": [
        # Human-readable explanation of top 3 risk factors.
        # Good writing quality needed. Groq 70B — called once.
        Groq("llama-3.3-70b-versatile",
             max_tokens=600,
             context_note="readable explanation, good writing, called once"),
        OR("meta-llama/llama-4-maverick:free",
           max_tokens=600),
        Cerebras("llama-3.3-70b",
                 max_tokens=600),
    ],

    "verdict_generation": [
        # Final verdict + one-sentence summary + risk_level.
        # Needs reasoning + clean writing. Groq 70B — called once.
        Groq("llama-3.3-70b-versatile",
             max_tokens=500,
             context_note="verdict needs reasoning + writing, called once"),
        OR("deepseek/deepseek-r1:free",
           max_tokens=500),
        Gemini("gemini-2.0-flash",
               max_tokens=500),
    ],

    "professional_report_generation": [
        # Full TXT/PDF report assembly — all sections.
        # Longest output (~3000-4000 tokens). Highest quality needed.
        # This is what enterprise clients READ.
        # Gemini 2.5 Pro: best writing, handles long structured output.
        Gemini("gemini-2.5-pro-preview",
               max_tokens=4000,
               context_note="highest quality task — client-facing output"),
        # Fallback 1: Same model via OpenRouter (separate quota pool)
        OR("google/gemini-2.5-pro-exp-03-25:free",
           max_tokens=4000),
        # Fallback 2: Gemini Flash if Pro quota exhausted both pools
        Gemini("gemini-2.0-flash",
               max_tokens=4000),
    ],
}


# ═══════════════════════════════════════════════════════════
# AGENTS THAT MUST NEVER CALL AN LLM
# If any of these appear in a routing call, raise an error.
# ═══════════════════════════════════════════════════════════
NO_LLM_AGENTS = {
    "report_discovery",
    "report_downloader",
    "report_parser",
    "evidence_retrieval",
    "realtime_monitoring",
}
