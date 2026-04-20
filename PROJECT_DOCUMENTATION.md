# ESGLens Project Documentation

## 1. Project Purpose

ESGLens is an ESG intelligence and greenwashing detection platform built around two connected layers:

1. A Python backend that runs a LangGraph-driven multi-agent analysis workflow.
2. A Next.js frontend that lets users submit analysis jobs, stream progress logs, inspect results, and run a separate ESG mismatch detector.

The system is designed to answer one core question:

**"Does a company's public sustainability claim match the evidence available across reports, external sources, financial context, and historical behavior?"**

This project is not just a single scoring script. It is a coordinated pipeline that:

- receives a company name, claim, and industry,
- decides how much analysis depth is needed,
- runs a sequence of specialized agents,
- synthesizes their outputs into a final verdict,
- generates a detailed report,
- stores the report on disk,
- and exposes the result to a browser dashboard.

---

## 2. High-Level Architecture

The repository is split into the following major areas:

### Backend

- `main_langgraph.py`
  Main CLI/programmatic entry point for ESG claim analysis.

- `core/`
  Workflow graph, state schema, supervisor routing, LLM routing, report generation, and shared orchestration logic.

- `agents/`
  Specialized analysis agents such as claim extraction, contradiction analysis, credibility analysis, sentiment analysis, regulatory scanning, and more.

- `features/esg_mismatch_detector/`
  A second feature pipeline focused on promise-vs-actual mismatch analysis for a company.

- `ml_models/`
  Trained model wrappers and stored artifacts used for scoring, anomaly detection, prediction, explainability, and NLP enrichment.

- `utils/`
  Helpers for data fetching, web search, report discovery, report parsing, source tracking, and external data handling.

### Frontend

- `frontend/src/app/`
  Next.js app-router pages and API routes.

- `frontend/src/components/`
  Shared UI building blocks, layout, and the live analysis state provider.

- `frontend/src/lib/server/report-utils.ts`
  The bridge layer that finds generated backend reports, parses them, and derives display-ready values for the frontend.

### Data / Outputs

- `reports/`
  Generated analysis report artifacts.

- `cache/`
  Cached ESG mismatch detector outputs and related temporary artifacts.

- `data/`
  Static reference datasets and mismatch output history.

---

## 3. Main Functional Areas

The project currently provides these main user-facing capabilities:

1. Landing page and product presentation site.
2. File-backed signup and login.
3. Auth-gated dashboard shell.
4. Main ESG claim analysis flow with live progress streaming.
5. Results view with parsed report summary and agent details.
6. Mismatch detector for company promise-vs-actual comparison.
7. Browser-local run history.
8. Browser-local settings/preferences.
9. Downloadable generated reports.

Some areas are fully functional end-to-end, while others are demo-level or partially scaffolded. That distinction is documented below.

---

## 4. Backend Main Analysis Flow

## 4.1 Entry Points

The main backend execution starts in `main_langgraph.py`.

There are three ways it is used:

1. Interactive CLI mode:
   `python main_langgraph.py`

2. Argument-based CLI mode:
   `python main_langgraph.py --company "Tesla" --claim "..." --industry "Automotive"`

3. Programmatic mode:
   `run_esg_analysis(company, claim, industry)`

When a run starts, the backend:

- loads environment variables,
- initializes the peer database,
- builds the LangGraph workflow,
- creates the initial analysis state,
- invokes the graph with a timeout,
- generates reports,
- saves artifacts into `reports/`,
- and returns a structured result.

## 4.2 State Object

The workflow passes a shared `ESGState` object between nodes. This state contains:

- user input: `company`, `claim`, `industry`
- routing metadata: `complexity_score`, `workflow_path`
- collected evidence
- confidence and risk outputs
- ML outputs
- enriched data such as:
  - `indian_financials`
  - `company_reports`
  - `carbon_extraction`
  - `greenwishing_analysis`
  - `regulatory_compliance`
  - `climatebert_analysis`
  - `esg_mismatch_analysis`
  - `explainability_report`
- `agent_outputs`
- `final_verdict`
- `report`

The `agent_outputs` field uses a custom reducer that keeps only the latest output per agent. This is important because it prevents unbounded list growth across LangGraph merges.

## 4.3 Supervisor and Routing

The first node is the supervisor.

Its job is to inspect the incoming claim and assign a `complexity_score` between `0.0` and `1.0`. It considers things like:

- whether the claim is vague or specific,
- whether it contains numeric targets,
- whether there is a timeframe,
- whether it is publicly verifiable,
- and whether it likely needs a full evidence pipeline.

Based on that, the workflow chooses one of three paths:

### Fast Track

Used only for genuinely simple, non-quantitative claims.

Execution order:

1. Claim extraction
2. Risk scoring
3. Confidence scoring
4. Verdict generation
5. Save peer data
6. Report generation

### Standard Track

Used for most practical claims.

Execution order:

1. Claim extraction
2. Evidence retrieval
3. Report discovery
4. Report download
5. Report parsing
6. Report claim extraction
7. Carbon extraction
8. Greenwishing detection
9. Regulatory scanning
10. ClimateBERT analysis
11. Temporal analysis
12. Temporal evidence injection
13. Contradiction analysis
14. ESG mismatch analysis
15. Peer comparison
16. Credibility analysis
17. Sentiment analysis
18. Realtime monitoring
19. Temporal consistency analysis
20. Risk scoring
21. Explainability generation
22. Confidence scoring
23. Verdict generation
24. Save peer data
25. Report generation

### Deep Analysis

This is the standard flow plus a debate stage before final report generation.

Execution order is the same as the standard path, with:

24. Verdict generation
25. Debate orchestrator
26. Save peer data
27. Report generation

This path is intended for the most ambiguous or high-risk claims.

---

## 5. What Each Main Backend Feature Does

## 5.1 Claim Extraction

Purpose:
Break a company claim into analyzable sub-claims.

Why it matters:
A broad marketing statement is too vague for direct scoring. The extractor converts it into smaller statements that later agents can verify.

Output examples:

- extracted claims
- claim structure
- temporal hints
- claim fragments from official reports

## 5.2 Evidence Retrieval

Purpose:
Collect supporting and contradictory evidence from external sources.

Why it matters:
The system should not score a claim based only on the company's own statement. It needs outside evidence.

Typical evidence channels include:

- web/news sources,
- legal/regulatory sources,
- report-derived content,
- financial context,
- cached or structured ESG enrichment.

## 5.3 Report Discovery, Download, and Parsing

Purpose:
Locate the company's official ESG/sustainability reports, download them, parse them, and extract metrics and claims.

Why it matters:
Official reports are one of the strongest primary sources for self-reported metrics.

This sub-flow is critical because it converts unstructured PDFs into structured report signals such as:

- emissions values,
- renewable energy percentages,
- water metrics,
- workforce metrics,
- governance ratios,
- net-zero target years,
- internal claim language.

## 5.4 Carbon Extraction

Purpose:
Identify and standardize carbon-related disclosures such as Scope 1, Scope 2, Scope 3, and total emissions.

Why it matters:
Many ESG claims are emissions-related, so carbon extraction is one of the most important factual grounding steps.

## 5.5 Greenwishing Detection

Purpose:
Detect aspirational sustainability language that lacks real execution backing.

Why it matters:
The project distinguishes between:

- greenwashing: misleading or contradictory environmental claims
- greenwishing: ambitious but weakly supported promises
- greenhushing/selective disclosure: material sustainability information that is omitted or softened

This helps the platform avoid treating all problematic ESG language as the same type of issue.

## 5.6 Regulatory Scanning

Purpose:
Compare the company context and claims against regulatory/compliance expectations.

Why it matters:
A claim may appear positive in isolation but be weak relative to mandatory or emerging requirements.

## 5.7 ClimateBERT Analysis

Purpose:
Apply climate-focused NLP interpretation to the claim and related evidence.

Why it matters:
This adds language-pattern analysis on top of factual retrieval. It is useful for identifying rhetoric, framing, and greenwashing-style wording.

## 5.8 Temporal Analysis and Temporal Consistency

Purpose:
Check whether the claim aligns with historical records, past violations, and realistic timelines.

Why it matters:
A target may sound strong but be inconsistent with:

- previous disclosures,
- missed targets,
- known incidents,
- or an unrealistic transition timeline.

The workflow also injects relevant past temporal violations back into the evidence pool so downstream reasoning uses them.

## 5.9 Contradiction Analysis

Purpose:
Compare extracted claims against available evidence to find mismatches.

Why it matters:
This is one of the core greenwashing detection steps. It looks for cases where public messaging and observed reality diverge.

## 5.10 ESG Mismatch Analysis

Purpose:
Run a dedicated promise-vs-performance consistency check inside the main pipeline.

Why it matters:
This complements contradiction analysis with a more explicit mismatch framing.

## 5.11 Peer Comparison

Purpose:
Compare the company to industry peers and benchmarks.

Why it matters:
A company's ESG posture should not be judged in a vacuum. Industry-relative performance matters.

The workflow also saves completed company scoring back into a peer database, allowing the comparison layer to improve over time.

## 5.12 Credibility Analysis

Purpose:
Estimate how trustworthy the retrieved sources and signals are.

Why it matters:
Evidence quality affects how much weight later scoring should assign to any finding.

## 5.13 Sentiment Analysis

Purpose:
Measure sentiment-related patterns around the company and claim context.

Why it matters:
This adds another directional signal to the overall risk picture, especially when negative external discourse clusters around ESG controversies.

## 5.14 Realtime Monitoring

Purpose:
Look for recent signals that may materially affect the interpretation of the claim.

Why it matters:
A claim can become outdated if a recent event changes the risk profile.

## 5.15 Risk Scoring

Purpose:
Produce the main risk assessment for the run.

Why it matters:
This is where the system synthesizes upstream outputs into a final greenwashing risk posture.

The scoring layer can use:

- rule-based logic,
- calibrated pillar scoring,
- ML models,
- industry baselines,
- and enriched external data.

## 5.16 Explainability

Purpose:
Generate human-readable reasoning for ML-assisted outputs.

Why it matters:
The result should be inspectable and explainable, not just a black-box score.

## 5.17 Confidence Scoring

Purpose:
Score how reliable the final run is based on evidence sufficiency and output quality.

Why it matters:
A confident low-risk result and a low-confidence low-risk result are not operationally the same.

## 5.18 Verdict Generation

Purpose:
Create the final verdict block that summarizes the run outcome.

Why it matters:
This becomes the main result consumed by reports and frontend display.

## 5.19 Debate Orchestration

Purpose:
Run an additional resolution phase in deep-analysis mode when the run is complex enough to justify it.

Why it matters:
It gives the system a place to reconcile conflicting agent views before the report is finalized.

## 5.20 Professional Report Generation

Purpose:
Turn the full state into:

- a readable text report,
- a structured JSON export,
- metadata and quality warnings.

Why it matters:
This is the packaging layer that makes the analysis usable by humans and the frontend.

The report generator also performs quality checks such as:

- evidence coverage,
- agent success consistency,
- peer quality,
- score traceability,
- confidence labeling.

---

## 6. Reports and Artifacts

After a successful main analysis, the backend writes outputs into `reports/`.

Typical generated files:

- `ESG_Report_<Company>_<Timestamp>.txt`
- `ESG_Report_<Company>_<Timestamp>.json`

Optionally, if enabled by environment variable:

- `ESG_Report_<Company>_<Timestamp>_FULL.json`

### TXT Report

The TXT report is the human-readable executive report.

### JSON Report

The JSON report is the machine-readable version used by the frontend for:

- agent tables,
- contribution breakdown,
- parsed summary fields,
- structured deep-dive views.

### Full JSON Report

This is a larger debug/state export, only saved when full result export is explicitly enabled.

---

## 7. Frontend Application Flow

The frontend is a Next.js application in `frontend/`.

It is composed of a public marketing site plus an authenticated dashboard shell.

## 7.1 Landing Page

Route:
`/`

Purpose:
Present the product, architecture, agent summary, and platform narrative.

Behavior:

- uses animated marketing sections,
- explains the product vision,
- routes users toward login/signup,
- does not run backend analysis itself.

## 7.2 Signup Flow

Route:
`/signup`

API:
`/api/auth/signup`

Behavior:

- user enters name, email, password, and role,
- backend reads `frontend/src/data.json`,
- creates a new user if the email does not already exist,
- writes the user back to the JSON file,
- returns the created user without the password,
- frontend stores the returned user in `localStorage`,
- user is redirected to `/dashboard`.

Important note:
This is a simple file-backed authentication flow, not a production auth system. Passwords are stored in plain form in the JSON file.

## 7.3 Login Flow

Route:
`/login`

API:
`/api/auth/login`

Behavior:

- user submits email and password,
- backend reads `frontend/src/data.json`,
- credentials are matched directly,
- on success the frontend stores user data in `localStorage`,
- user is redirected to `/dashboard`.

## 7.4 Dashboard Shell

Routes under:
`/dashboard/*`

Behavior:

- wrapped in `AnalysisRunProvider`,
- uses a sidebar and top bar,
- checks for `localStorage.user`,
- redirects to `/login` if no local user is found.

Important note:
This is client-side session gating only. There is no server-side auth/session enforcement yet.

---

## 8. Main Analysis Page Flow

Route:
`/dashboard/analyze`

This is the primary operational page for the main backend pipeline.

## 8.1 User Input

The page collects:

- company name
- industry
- ESG claim

When the user clicks "Run Main Pipeline", the page calls:

`POST /api/analyze-company`

## 8.2 API Route Behavior

The `analyze-company` API route does the following:

1. Loads backend environment variables.
2. Validates that `company_name` and `claim` are present.
3. Verifies required runtime credentials exist.
4. Resolves the Python executable.
5. Resolves the backend script path.
6. Spawns the Python process:
   `python main_langgraph.py --company ... --claim ... --industry ...`
7. Streams progress back to the browser using Server-Sent Events (SSE).

The route emits events of type:

- `status`
- `log`
- `result`
- `error`
- `end`

## 8.3 Live Log Streaming

While the Python process is running:

- stdout and stderr are buffered,
- important log lines are classified,
- the frontend receives them in near real time,
- a heartbeat status is emitted every few seconds.

This is what powers the "Live Backend Dashboard" panel in the UI.

## 8.4 Result Resolution

When the Python process exits successfully:

- the API locates the latest generated report in `reports/`,
- reads the TXT and JSON report files,
- derives display-ready values,
- parses the main report into structured fields,
- sends a final `result` event to the frontend.

If the process does not produce a fresh report, the route attempts to serve the latest available matching report as a fallback.

## 8.5 Frontend State Management

The analysis page uses `AnalysisRunProvider`.

This provider is responsible for:

- storing live logs,
- tracking current run state,
- storing the latest result,
- deduplicating repeated log lines,
- persisting compact live run state to `localStorage`,
- storing a compact history of completed runs.

The provider exposes three app states:

- `input`
- `processing`
- `results`

## 8.6 Results Display

Once a run completes, the page renders:

1. Executive summary
2. Main ESG scoring report cards
3. Parsed report table
4. Narrative paragraphs
5. Run metadata
6. Agent contribution breakdown
7. Claim extraction trace
8. Agent deep-dive cards

This page is one of the most complete end-to-end parts of the application.

---

## 9. Mismatch Detector Flow

Route:
`/dashboard/mismatch`

API:
`POST /api/mismatch-detect`

This feature is separate from the main LangGraph run. It uses a dedicated Python pipeline under:

`features/esg_mismatch_detector/pipeline.py`

## 9.1 Execution Steps

When the user enters a company and submits:

1. The frontend calls `/api/mismatch-detect`.
2. The API route spawns:
   `python -m features.esg_mismatch_detector.pipeline "<company>"`
3. The Python pipeline runs:
   - company resolution
   - ESG report fetch
   - promise extraction
   - external evidence collection
   - promise-vs-actual comparison
   - mismatch formatting
   - cache save

## 9.2 Caching

Mismatch results are cached in:

- `cache/esg_analysis/<company>.json`

Cache characteristics:

- 24-hour TTL
- schema-version checking
- fallback-to-cache when live execution fails

## 9.3 Output Shape

The detector produces a user-friendly result with sections such as:

- company analyzed
- report availability
- overall greenwashing risk
- executive summary
- data coverage
- confidence score
- future commitments and progress
- past promise-implementation gaps

## 9.4 Frontend Display

The mismatch page renders:

- summary cards,
- a future commitments table,
- a past mismatches table,
- fallback notes when only partial data is available.

This flow is functional and independent of the main LangGraph dashboard run.

---

## 10. History and Settings

## 10.1 History Page

Route:
`/dashboard/history`

Behavior:

- reads `esg-analysis-history` from browser `localStorage`,
- displays recent run summaries,
- shows company, claim, industry, risk, confidence, and timestamp.

This feature is client-side only. It reflects runs launched from the same browser.

## 10.2 Settings Page

Route:
`/dashboard/settings`

Behavior:

- stores default industry in `localStorage`,
- stores a workflow timeout value in `localStorage`,
- displays a simple saved confirmation.

Important note:
The timeout stored here is not currently wired into the backend environment automatically. It behaves as a demo/user-preference store, not a fully connected runtime control.

---

## 11. Current Data and Storage Behavior

The project uses multiple storage patterns:

### Disk files

- reports in `reports/`
- user records in `frontend/src/data.json`
- mismatch caches in `cache/esg_analysis/`
- static datasets in `data/`

### Local browser storage

- logged-in user
- live analysis state
- compact analysis history
- default settings/preferences

### Runtime memory

- LangGraph state
- analysis provider in-memory logs/results
- temporary evidence and report generation structures

---

## 12. LLM and Model Layer

The system includes several backend components for model usage:

- centralized LLM calling and routing
- multiple provider support
- disk-backed and session-aware caching
- ML models for risk, anomaly detection, trend prediction, sentiment, calibration, and explainability

At runtime, the analysis can combine:

- LLM-based reasoning,
- deterministic scoring logic,
- structured external data,
- stored model artifacts.

This hybrid design is important because the project is not purely prompt-based. It mixes retrieval, symbolic logic, and trained model inference.

---

## 13. What Is Fully Working vs. What Is Partial

### Functioning End-to-End

- main Python LangGraph analysis trigger
- report generation and report saving
- frontend analysis submission
- SSE log streaming
- result retrieval from saved reports
- mismatch detector pipeline and cache fallback
- signup and login using file-backed user storage
- client-side session gating
- history and settings pages at browser-storage level

### Present but Demo-Level / Partial / Incomplete

- dashboard overview widgets use static demo data
- sidebar includes `/dashboard/reports`, but that page is not present in the current codebase
- settings values are not fully wired back into backend runtime behavior
- authentication is not production-safe
- some agent capabilities depend on environment keys and available external sources

This distinction matters for anyone extending the project. The core analysis engine is the strongest implemented area; some surrounding product features are still scaffolding or demo-grade.

---

## 14. End-to-End User Journey

A typical user journey through the current product looks like this:

1. User opens the landing page and navigates to signup or login.
2. User creates an account or logs in.
3. User reaches the dashboard.
4. User opens "Analyze Company".
5. User submits a company, claim, and industry.
6. The frontend calls the analysis API.
7. The API spawns the Python backend process.
8. The backend runs the LangGraph workflow.
9. Logs are streamed back to the frontend.
10. The backend writes report files into `reports/`.
11. The API reads the latest matching report.
12. The frontend renders the report summary, tables, narrative, and agent details.
13. The result is stored into browser history.
14. User can also run the separate mismatch detector for company-level promise-vs-actual analysis.

---

## 15. How to Run the Project

## 15.1 Backend

Install dependencies:

```bash
pip install -r requirements.txt
```

Run interactive backend:

```bash
python main_langgraph.py
```

Run direct backend analysis:

```bash
python main_langgraph.py --company "Unilever" --claim "Unilever aims to achieve net-zero emissions across its value chain by 2039." --industry "Consumer Goods"
```

## 15.2 Frontend

From `frontend/`:

```bash
npm install
npm run dev
```

The frontend expects access to the backend project root because it spawns `main_langgraph.py` and reads the `reports/` directory from there.

---

## 16. Summary

ESGLens is a multi-layer ESG analysis platform centered on a LangGraph-based orchestration engine. Its most important working capability is the full claim-analysis pipeline: claim intake, supervisor routing, multi-agent analysis, report generation, report persistence, and browser-based live result display.

The frontend is not just a static UI. It actively launches backend runs, streams execution logs, retrieves saved report artifacts, parses them, and presents them in a structured analyst-friendly view.

The mismatch detector is a second, separate feature that focuses on promise-versus-actual behavior and includes its own cache-backed execution model.

Overall, the project already contains a meaningful and functioning ESG intelligence workflow, with a few product-level pieces still at demo or scaffold stage.
