# ESG Greenwashing Detection Platform

This repository contains an enterprise-grade ESG intelligence platform for detecting greenwashing and ESG claim mismatch risk using multi-agent AI, machine learning, regulatory mapping, and explainability.

This README is the single consolidated project document and merges the important content that was previously split across multiple files.

## 1. What the Platform Does

Given a company and a sustainability claim, the system:

1. Collects evidence from news, regulatory, financial, and report sources.
2. Runs specialized analysis agents (contradiction, temporal consistency, carbon extraction, deception patterns, regulatory checks, etc.).
3. Produces calibrated greenwashing risk outputs (score, band, confidence, rationale).
4. Generates executive text and JSON reports.
5. Supports both CLI and a Next.js dashboard workflow.

Primary backend entry point:
- main_langgraph.py

## 2. Current Feature Coverage (As of March 2026)

### 2.1 Core Multi-Agent Backend

Implemented agent modules in agents/:

1. claim_extractor.py
2. evidence_retriever.py
3. contradiction_analyzer.py
4. historical_analyst.py
5. temporal_consistency_agent.py
6. industry_comparator.py
7. credibility_analyst.py
8. risk_scorer.py
9. sentiment_analyzer.py
10. realtime_monitor.py
11. confidence_scorer.py
12. conflict_resolver.py
13. financial_analyst.py
14. carbon_extractor.py
15. greenwishing_detector.py
16. regulatory_scanner.py

What these agents cover end-to-end:
- Claim extraction and structuring.
- Multi-source evidence retrieval and filtering.
- Contradiction and timeline drift detection.
- Peer and industry benchmarking.
- Source credibility weighting.
- Greenwashing, greenwishing, and greenhushing pattern detection.
- Scope 1/2/3 carbon data extraction with India-aware support.
- Regulatory mapping across India, EU, US, UK, and global frameworks.
- Final confidence and risk synthesis.

### 2.2 Workflow and Orchestration

Implemented in core/:
- workflow_phase2.py: LangGraph workflow builder.
- supervisor_agent.py: complexity analysis and route selection.
- debate_orchestrator.py: conflict resolution for agent disagreement.
- agent_wrappers.py: node-level orchestration and diagnostics.
- state_schema.py: shared state structure.
- professional_report_generator.py: executive report generation.
- confidence_monitor.py, evidence_cache.py, vector_store.py, llm_client.py.

Routing modes:
- Fast track for simpler claims.
- Standard track for moderate complexity.
- Deep analysis with debate for high complexity or conflicting signals.

### 2.3 ML and Explainability

Implemented in ml_models/:
- xgboost_risk_model.py
- lightgbm_esg_predictor.py
- lstm_trend_predictor.py
- anomaly_detector.py
- sentiment_esg_predictor.py
- climatebert_analyzer.py
- explainability_engine.py

ML/XAI capabilities:
- Hybrid risk scoring (ML + rules/formula behavior).
- Climate language intelligence via ClimateBERT.
- SHAP/LIME explanation generation for report-ready reasoning.
- Trend and anomaly signals for additional risk context.

### 2.4 Carbon, Regulatory, and India-Focused Intelligence

Implemented capabilities include:
- Scope 1/2/3 extraction and consistency checks.
- BRSR-oriented disclosure checks.
- India grid factor handling and Indian number notation support in extraction flows.
- Regulatory scanners for:
  - India: SEBI, MCA, CPCB, RBI/BEE-related contexts.
  - EU: CSRD, EU Taxonomy, SFDR.
  - US: SEC climate guidance, FTC green claims context.
  - UK: FCA anti-greenwashing context.
  - Global: GHG Protocol, SBTi, GRI, CDP.

### 2.5 Data Source Layer and Retrieval

Implemented source and retrieval utilities in utils/ include:
- free_data_sources.py
- enterprise_data_sources.py
- indian_data_sources.py
- indian_financial_data.py
- report_discovery.py
- report_downloader.py
- report_parser.py
- company_report_fetcher.py
- web_search.py
- source_tracker.py

Evidence domains covered:
- News and media feeds.
- Regulatory/legal signals.
- Academic/research references.
- Financial APIs and market context.
- Sustainability report ingestion (PDF and HTML fallback paths).

### 2.6 ESG Mismatch Detector (Dedicated Pipeline)

Location: features/esg_mismatch_detector/

This pipeline independently detects mismatch between corporate ESG promises and externally verified evidence.

Main modules:
- company_resolver.py
- report_collector.py
- promise_extractor.py
- evidence_collector.py
- comparison_engine.py
- pipeline.py

Highlights:
- Deterministic promise-vs-performance gap engine.
- Qualitative violation detection (not only numeric mismatch).
- Action-verb filtering to reduce vague implementation claims.
- 24-hour cache for repeat runs.

### 2.7 Frontend Dashboard (Next.js)

Location: frontend/

Implemented UI routes include:
- Public landing, login, signup.
- Dashboard overview and settings.
- Analyze page with live streamed backend logs.
- Mismatch analysis page.
- Report history and report download experiences.

Implemented API routes include:
- api/analyze-company: starts backend analysis and streams status/log/result events.
- api/mismatch-detect: runs mismatch pipeline.
- api/reports: report listing.
- api/reports/download: secure report download.
- api/reports/pdf: PDF rendering from report artifacts.
- api/auth/login and api/auth/signup endpoints.

### 2.8 Performance and Reliability Enhancements (Integrated)

Integrated improvements documented in project reports include:
- ESG section detection before heavy claim extraction.
- Keyword-based chunk filtering to avoid irrelevant LLM calls.
- Batch chunk processing.
- Claim extraction caching and rate limiting.
- Better diagnostics and integration verification helpers.
- HTML parsing fallback and parsing stability fixes.

Impact reported in implementation docs:
- Significant API-call and cost reduction in claim extraction workflows.
- Faster repeated analyses due to cache reuse.

## 3. Repository Structure

Top-level folders:
- agents/: specialized analysis agents.
- core/: orchestration, workflow state, report generation.
- utils/: source adapters, report ingestion, retrieval utilities.
- ml_models/: model wrappers and explainability logic.
- features/esg_mismatch_detector/: mismatch pipeline.
- frontend/: Next.js application.
- data/: datasets and metadata.
- tests/: integration and quality tests.
- scripts/: maintenance and model compatibility helpers.
- cache/: runtime caches.
- chroma_db/: vector persistence.
- reports/: generated outputs.

Key root files:
- main_langgraph.py
- run_validation.py
- requirements.txt
- pytest.ini

## 4. Setup

### 4.1 Backend Prerequisites

- Python 3.11+ recommended.
- pip and virtual environment tool.
- Internet access for live source retrieval.

### 4.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Environment Variables

1. Copy .env.example to .env.
2. Provide at least one LLM key.

Minimum recommended variables:

```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
USE_LANGGRAPH=true
DEFAULT_JURISDICTION=India
CLIMATEBERT_ENABLED=true
SHAP_ENABLED=true
```

Optional but useful:
- NEWSAPI_KEY
- NEWSDATA_KEY
- THENEWSAPI_KEY
- MEDIASTACK_KEY
- SEC_API_KEY

## 5. Running the Platform

### 5.1 Main ESG Analysis (CLI)

Interactive mode:

```bash
python main_langgraph.py
```

Direct mode:

```bash
python main_langgraph.py --company "Unilever" --claim "Unilever aims to achieve net-zero emissions across its value chain by 2039." --industry "Consumer Goods"
```

Output files are saved to reports/ as text and JSON.

### 5.2 ESG Mismatch Pipeline

```bash
python -m features.esg_mismatch_detector.pipeline "Microsoft"
```

Example cache-bypass pattern (PowerShell/Linux equivalent cleanup command may vary):

```bash
python -m features.esg_mismatch_detector.pipeline "Volkswagen"
```

### 5.3 Validation and Tests

Run project validation workflow:

```bash
python run_validation.py
```

Run pytest suite:

```bash
pytest -v
```

## 6. Frontend App

### 6.1 Install and Start

```bash
cd frontend
npm install
npm run dev
```

Open:
- http://localhost:3000

### 6.2 Frontend Runtime Behavior

- Analyze UI streams backend logs and status from api/analyze-company.
- Generated report files can be listed, downloaded, and rendered to PDF via report endpoints.
- Mismatch detector UI invokes backend mismatch pipeline endpoint.

## 7. Reports and Outputs

Generated artifacts are stored in reports/:
- ESG_Report_<Company>_<timestamp>.txt
- ESG_Report_<Company>_<timestamp>.json
- Optional full debug JSON (environment-flag controlled in backend).

Typical report content includes:
- Claim summary and context.
- Risk score/band and confidence.
- Contradiction and evidence highlights.
- Carbon and regulatory findings when available.
- Executive narrative for analyst use.

## 8. Datasets

Active datasets in data/ include:
- company_esg_financial_dataset.csv
- sp500_esg_data.csv
- data.csv

Additional reference/metadata datasets are also present for future expansion and baselining.

## 9. Known Technical Debt

Current tracked item:
- Pydantic v2 config migration in config/settings.py:
  - Replace deprecated class-based Config usage with model_config = ConfigDict(...).

## 10. ML Compatibility Utilities

Scripts under scripts/ address model compatibility after dependency upgrades:
- test_lstm_loading.py
- regenerate_lstm_model.py
- regenerate_scalers.py

Use these when TensorFlow/Keras or scikit-learn upgrades break serialized model/scaler loading.

## 11. Consolidated Documentation Note

This README consolidates and supersedes the operational content previously spread across:
- PROJECT.README.md
- PROJECT_FLOW_AND_ML_MODELS.md
- NEW_FEATURES_2026.md
- IMPLEMENTATION_SUMMARY.txt
- pipeline.md
- DEMO_SCRIPT_SHELL_MICROSOFT.md
- TECH_DEBT.md
- features/esg_mismatch_detector/README.md
- data/README.md
- scripts/README_ML_FIXES.md
- frontend/README.md

## 12. Current Project Status

The repository now includes:
- A production-oriented multi-agent ESG analysis backend.
- A dedicated mismatch detection pipeline.
- ML and explainability integration.
- Frontend dashboard and API bridge to backend Python pipelines.
- Caching, optimization, and stabilization improvements.

In practical terms, this is beyond MVP stage and functions as a full-stack ESG intelligence platform with research and enterprise-oriented capabilities.
