# ESGLens - AI-Powered Greenwashing Detection

ESGLens is an enterprise-grade AI system designed to detect greenwashing in corporate claims. Built using **LangGraph**, it features an orchestrated swarm of **14 specialized AI agents**, highly resilient **dynamic LLM routing**, ML-enhanced risk scoring, and real-time financial data integration to provide exhaustive ESG analysis.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API keys. The system uses a multi-provider fallback routing system, so providing multiple keys maximizes resilience.

```env
# LLM Providers (Provide at least 2 for optimal fallback routing)
GROQ_API_KEY=your_groq_key
CEREBRAS_API_KEY=your_cerebras_key
OPENROUTER_API_KEY=your_openrouter_key
GEMINI_API_KEY=your_gemini_key

# Optional Premium Tool APIs
NEWS_API_KEY=your_newsapi_key
NEWSDATA_API_KEY=your_newsdata_key

# WBA Data API (Bearer token)
WBA_API_KEY=your_wba_api_key

# Resource Watch / WRI Aqueduct
RESOURCE_WATCH_API_KEY=your_resource_watch_api_key
RESOURCE_WATCH_TOKEN=your_resource_watch_jwt

# Optional dynamic materiality profile source
# If set, this remote JSON overlays local config/materiality_map.json.
# Supported formats: { "profiles": { ... } } or direct { ... } map.
MATERIALITY_PROFILE_URL=
MATERIALITY_PROFILE_PATH=config/materiality_map.json

# Optional SASB-style dataset source (CSV/JSON) used to auto-build/overlay profiles
SASB_MATERIALITY_DATA_URL=

# Company-centric Knowledge Graph (Neo4j)
KG_ENABLED=true
KG_USE_LLM_GRAPH_TRANSFORMER=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
```

To refresh `config/materiality_map.json` from remote sources:
```bash
venv\Scripts\python.exe scripts\refresh_materiality_profiles.py
```

Research telemetry summary (aggregates all logged runs):
```bash
venv\Scripts\python.exe scripts\summarize_research_runs.py
```

Sector-wise benchmark evaluation artifacts (JSON/CSV/Markdown):
```bash
venv\Scripts\python.exe scripts\run_sector_benchmark.py
```
This now includes bootstrap confidence intervals, near-threshold counts around the 50-point decision boundary, and explicit small-sample warning bands.

Append new validated ground-truth rows from `data/ground_truth_additions.csv`:
```bash
venv\Scripts\python.exe scripts\append_ground_truth_cases.py
```

Each generated report also persists the justification graph to `reports/fact_graphs/` as a JSON artifact for auditability and presentation review.
Each standard/deep analysis run also builds a company-centric KG payload in `reports/company_kg/`, and will ingest it into Neo4j when the `NEO4J_*` environment variables are configured.

Neo4j verification helper:
```bash
venv\Scripts\python.exe scripts\verify_company_kg.py --scenario shell
venv\Scripts\python.exe scripts\verify_company_kg.py --scenario microsoft
```

Example Cypher checks:
```cypher
MATCH (o:Organization {name: 'Shell'})-->(g) RETURN g
MATCH (o:Organization {name: 'Microsoft'})-[:HAS_KPI]->(k:KPI {name: 'GHG_Intensity'}) RETURN k.year, k.value ORDER BY k.year
```

### 3. Run the Application
The main entry point provides a CLI interface to input a company and their ESG claim.
```bash
python main_langgraph.py
```

## 🧠 System Architecture

ESGLens operates using an intelligent, multi-agent workflow orchestrated via **LangGraph**:

1. **Intelligent Orchestration**: A `Supervisor` agent evaluates the complexity of the specific claim and dynamically routes it through appropriate analysis pipelines (Fast Track, Standard Track, or a Deep Analysis involving a multi-agent Debate Orchestrator).
2. **Dynamic LLM Routing**: All LLM calls are routed centrally through `core/llm_router.py`. Each agent is assigned an optimal primary model (e.g., Llama 3 70B for heavy reasoning, 8B for extraction) with a built-in **3-model fallback chain** across Cerebras, Groq, OpenRouter, and Gemini. If a provider rate-limits or fails, the system instantly falls back to the next available provider.
3. **14 Specialized Agents**:
   - `Claim Extractor`
   - `Evidence Retriever` (includes web, semantic, and financial data hooks)
   - `Contradiction Analyzer`
   - `Historical & Temporal Analyst`
   - `Industry & Peer Comparator`
   - `Credibility Analyst`
   - `Risk Scorer` (ML + Formula hybrid)
   - `Sentiment Analyzer`
   - `Realtime Monitor`
   - `Confidence Scorer`
   - `Conflict Resolver`
   - `Debate Orchestrator`
   - `Professional Report Generator`
   - `Supervisor`

## 🔧 Key Features

- **Resilient AI Inference**: Centralized `call_llm()` mechanism with disk-backed caching, exponential backoff, rate limit handling, and automatic multi-provider failovers.
- **Financial & ESG Correlation**: Uncovers nuanced greenwashing patterns by correlating ESG claims with real-time financial data (revenue, margins, debt) via `yfinance`. Detects red flags like aggressive green marketing masking core revenue declines or high carbon-intensity.
- **Multi-Source Evidence**: Automated pipeline retrieval across News APIs, Regulatory & Legal portals, Semantic Scholar, and OpenSanctions.
- **Hybrid Risk Scoring**: Robust scoring engine combining XGBoost Machine Learning classification with strict formula-based rules depending on evidence confidence thresholds.
- **Professional Artifacts**: Automatically generates production-grade PDF reports summarizing findings, risk levels, and specific red flags.

## 📁 Repository Structure

```
ESG/
├── agents/             # 14 specialized LangGraph agents containing analytical logic
├── config/             # System prompts, company aliases, and industry baselines
├── core/               # Centralized LLM routing, LangGraph state schema, and workflows
├── data/               # Essential reference databases (peer DB, emissions floors)
├── features/           # Specialized sub-pipelines (e.g., ESG Mismatch Detector)
├── frontend/           # Next.js web dashboard (Under Construction)
├── ml_models/          # Trained weights and inference wrappers for risk/anomaly detection
├── utils/              # PDF parsing, web crawling, and data source connectors
├── .env.example        # Environment variable template
├── main_langgraph.py   # CLI entry point
└── requirements.txt    # Modern, optimized Python dependency list
```

## 📊 Example Output

After analysis, ESGLens outputs a detailed taxonomy of risk, confidence, and financial implications. A sample terminal output (and corresponding PDF):

```json
{
  "risk_level": "HIGH",
  "greenwashing_risk_score": 82.5,
  "esg_score": 18.0,
  "confidence": 0.88,
  "financial_analysis": {
    "revenue_usd": 94800000000,
    "profit_margin_pct": 4.5,
    "carbon_intensity": 0.85,
    "greenwashing_flags": ["High profit + low ESG execution", "High carbon intensity (>0.8)"]
  },
  "evidence_count": 52,
  "report": "reports/Company_2026_Analysis.pdf"
}
```

---
*Built with LangGraph, Groq, Cerebras, OpenRouter, and XGBoost.*
