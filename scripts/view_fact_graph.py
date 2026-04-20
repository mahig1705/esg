r"""
Generate and render a fact-centric ESG graph to an HTML file.

Usage examples:
    venv\Scripts\python.exe scripts\view_fact_graph.py --company "Shell" --claim "We are on track to be a net-zero emissions energy business by 2050."
    venv\Scripts\python.exe scripts\view_fact_graph.py --graph reports\fact_graphs\Shell_demo_fact_graph.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fact_graph_builder import build_esg_fact_graph  # noqa: E402
from core.fact_graph_persistence import persist_fact_graph  # noqa: E402
from data.known_cases import get_known_contradictions  # noqa: E402


DEFAULT_CLAIMS = {
    "jpmorgan chase": "We are aligned to the Paris Agreement and committed to net zero financing by 2050.",
    "shell": "We are on track to be a net-zero emissions energy business by 2050.",
    "bp": "We aim to be a net zero company by 2050 or sooner.",
    "volkswagen": "Clean diesel technology meets strictest emission standards.",
}


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip())
    return text.strip("_") or "graph"


def _guess_source_type(case: Dict[str, Any]) -> str:
    regulatory_body = str(case.get("regulatory_body") or "").lower()
    source = str(case.get("source") or "").lower()
    if any(token in regulatory_body for token in ["court", "epa", "authority", "ftc", "asa", "doj"]):
        return "Government/Regulatory"
    if "reuters" in source or "financial times" in source or "guardian" in source or "bloomberg" in source:
        return "Tier-1 Financial Media"
    if any(token in regulatory_body for token in ["clientearth", "influencemap", "ngo"]):
        return "NGO"
    return "Compliance/Sanctions Database"


def _ground_truth_evidence(company: str, claim: str) -> List[Dict[str, Any]]:
    dataset_path = PROJECT_ROOT / "data" / "ground_truth_dataset.csv"
    if not dataset_path.exists():
        return []

    df = pd.read_csv(dataset_path)
    if "company_name" not in df.columns:
        return []

    company_norm = company.strip().lower()
    rows = df[df["company_name"].astype(str).str.strip().str.lower() == company_norm]
    evidence: List[Dict[str, Any]] = []
    for _, row in rows.head(5).iterrows():
        source_url = str(row.get("source_url") or "").strip()
        year = row.get("year")
        case_type = str(row.get("case_type") or "known_case").replace("_", " ")
        regulator = str(row.get("regulatory_body") or "Public source")
        evidence.append(
            {
                "source_id": f"gt_{len(evidence) + 1}",
                "relevant_text": f"{regulator} recorded a {case_type} case related to the claim: {row.get('claim_text')}",
                "url": source_url,
                "source_name": regulator,
                "source_type": "Government/Regulatory",
                "year": year,
            }
        )

    if not evidence and claim:
        evidence.append(
            {
                "source_id": "claim_seed_1",
                "relevant_text": claim,
                "url": "",
                "source_name": "Claim seed",
                "source_type": "Company Disclosure",
                "year": datetime.now(timezone.utc).year,
            }
        )
    return evidence


def _local_demo_graph(company: str, claim: str) -> Dict[str, Any]:
    contradictions = get_known_contradictions(company, claim)
    evidence = _ground_truth_evidence(company, claim)

    for idx, case in enumerate(contradictions, start=1):
        evidence.append(
            {
                "source_id": f"known_case_{idx}",
                "relevant_text": str(case.get("contradiction_text") or case.get("description") or "").strip(),
                "url": str(case.get("source_url") or "").strip(),
                "source_name": str(case.get("source") or case.get("regulatory_body") or "Known case"),
                "source_type": _guess_source_type(case),
                "year": case.get("year"),
            }
        )

    temporal_evidence = []
    for case in contradictions:
        year = case.get("year")
        text = case.get("description") or case.get("contradiction_text") or ""
        if year and text:
            temporal_evidence.append(f"{year}: {text}")

    return build_esg_fact_graph(
        company=company,
        claim_text=claim,
        evidence=evidence,
        contradictions=contradictions,
        temporal_consistency={"evidence": temporal_evidence},
    )


def _render_html(graph: Dict[str, Any], source_json_path: str, output_html_path: Path) -> None:
    payload = json.dumps(graph, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ESG Fact Graph Viewer</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffdf8;
      --ink: #1f2a2c;
      --muted: #667275;
      --line: #d4ccbc;
      --claim: #df6d3c;
      --fact: #2f7f6d;
      --source: #3676b8;
      --contradiction: #b54040;
      --temporal: #8a5aa6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(223,109,60,.16), transparent 32%),
        radial-gradient(circle at top right, rgba(54,118,184,.14), transparent 28%),
        linear-gradient(180deg, #f8f4ea 0%, #efe7da 100%);
    }}
    .shell {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: rgba(255,255,255,.8);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(31,42,44,.08);
      border-radius: 24px;
      padding: 24px 28px;
      box-shadow: 0 18px 44px rgba(31,42,44,.08);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 900px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .chip {{
      background: #f7f1e5;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(31,42,44,.08);
      border-radius: 22px;
      box-shadow: 0 16px 36px rgba(31,42,44,.06);
      overflow: hidden;
    }}
    .card h2 {{
      font-size: 17px;
      margin: 0;
      padding: 18px 20px 8px;
    }}
    .card-body {{
      padding: 0 20px 20px;
    }}
    #graph {{
      width: 100%;
      height: 760px;
      display: block;
      background:
        radial-gradient(circle at center, rgba(47,127,109,.06), transparent 30%),
        linear-gradient(180deg, #fffdf8 0%, #f8f3ea 100%);
    }}
    .stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }}
    .stat {{
      background: #faf6ee;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .stat .value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }}
    .detail {{
      border-top: 1px solid rgba(31,42,44,.08);
      padding-top: 14px;
      margin-top: 14px;
    }}
    .detail h3 {{
      margin: 0 0 10px;
      font-size: 14px;
    }}
    .mono {{
      font-family: Consolas, monospace;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}
    .facts {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
      max-height: 360px;
      overflow: auto;
      padding-right: 4px;
    }}
    .fact {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fffaf1;
    }}
    .fact .head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .fact p {{
      margin: 0;
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 1080px) {{
      .grid {{ grid-template-columns: 1fr; }}
      #graph {{ height: 560px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Fact-Centric ESG Knowledge Graph</h1>
      <p>This viewer shows the graph your system builds from a claim, linked facts, source nodes, contradictions, and temporal evidence. The current artifact was rendered from <span class="mono">{source_json_path}</span>.</p>
      <div class="meta">
        <span class="chip" id="companyChip"></span>
        <span class="chip" id="claimChip"></span>
        <span class="chip" id="decisionChip"></span>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Graph View</h2>
        <div class="card-body">
          <canvas id="graph" width="920" height="760"></canvas>
          <div class="legend">
            <span class="legend-item"><span class="dot" style="background: var(--claim)"></span>Claim</span>
            <span class="legend-item"><span class="dot" style="background: var(--fact)"></span>Fact</span>
            <span class="legend-item"><span class="dot" style="background: var(--source)"></span>Source</span>
            <span class="legend-item"><span class="dot" style="background: var(--contradiction)"></span>Contradiction</span>
            <span class="legend-item"><span class="dot" style="background: var(--temporal)"></span>Temporal</span>
          </div>
        </div>
      </article>

      <aside class="card">
        <h2>Graph Summary</h2>
        <div class="card-body">
          <div class="stats">
            <div class="stat"><div class="label">Facts</div><div class="value" id="factCount"></div></div>
            <div class="stat"><div class="label">Verified Facts</div><div class="value" id="verifiedFacts"></div></div>
            <div class="stat"><div class="label">Claim Links</div><div class="value" id="claimLinked"></div></div>
            <div class="stat"><div class="label">Graph Density</div><div class="value" id="graphDensity"></div></div>
          </div>
          <div class="detail">
            <h3>Pillar Coverage</h3>
            <div id="pillarCoverage"></div>
          </div>
          <div class="detail">
            <h3>Selected Node</h3>
            <div id="selectionText">Click a node to inspect it.</div>
          </div>
          <div class="detail">
            <h3>Top Facts</h3>
            <div class="facts" id="factsList"></div>
          </div>
        </div>
      </aside>
    </section>
  </div>

  <script>
    const graph = {payload};
    const colors = {{
      claim: "#df6d3c",
      fact: "#2f7f6d",
      source: "#3676b8",
      contradiction_fact: "#b54040",
      temporal_fact: "#8a5aa6",
      default: "#59656a"
    }};

    const summary = graph.summary || {{}};
    document.getElementById("companyChip").textContent = `Company: ${{graph.company || "Unknown"}}`;
    document.getElementById("claimChip").textContent = `Claim: ${{(graph.claim_text || "").slice(0, 90)}}${{(graph.claim_text || "").length > 90 ? "..." : ""}}`;
    document.getElementById("decisionChip").textContent = summary.is_decision_ready ? "Decision ready: Yes" : "Decision ready: No";
    document.getElementById("factCount").textContent = summary.fact_count || 0;
    document.getElementById("verifiedFacts").textContent = summary.verified_fact_count || 0;
    document.getElementById("claimLinked").textContent = summary.claim_linked_fact_count || 0;
    document.getElementById("graphDensity").textContent = summary.graph_density || 0;

    const coverage = summary.coverage_by_pillar || {{}};
    document.getElementById("pillarCoverage").innerHTML = `
      <div class="stat"><div class="label">Environmental</div><div class="value">${{coverage.E || 0}}</div></div>
      <div class="stat"><div class="label">Social</div><div class="value">${{coverage.S || 0}}</div></div>
      <div class="stat"><div class="label">Governance</div><div class="value">${{coverage.G || 0}}</div></div>
    `;

    const factsList = document.getElementById("factsList");
    const topFacts = (graph.facts || []).slice(0, 12);
    factsList.innerHTML = topFacts.map(f => `
      <div class="fact">
        <div class="head">
          <span>${{f.pillar || "?"}} pillar</span>
          <span>Verifiability ${{f.verifiability_score || 0}}</span>
        </div>
        <p>${{f.text || ""}}</p>
      </div>
    `).join("");

    const canvas = document.getElementById("graph");
    const ctx = canvas.getContext("2d");
    const nodes = graph.nodes || [];
    const edges = graph.edges || [];
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    const typedNodes = {{
      claim: nodes.filter(n => n.node_type === "claim"),
      fact: nodes.filter(n => n.node_type === "fact"),
      source: nodes.filter(n => n.node_type === "source"),
      contradiction_fact: nodes.filter(n => n.node_type === "contradiction_fact"),
      temporal_fact: nodes.filter(n => n.node_type === "temporal_fact"),
    }};

    function placeRing(items, radius, startAngle) {{
      const total = items.length || 1;
      items.forEach((node, index) => {{
        const angle = startAngle + ((Math.PI * 2) / total) * index;
        node.x = centerX + Math.cos(angle) * radius;
        node.y = centerY + Math.sin(angle) * radius;
      }});
    }}

    if (typedNodes.claim[0]) {{
      typedNodes.claim[0].x = centerX;
      typedNodes.claim[0].y = centerY;
    }}
    placeRing(typedNodes.fact, 170, -Math.PI / 2);
    placeRing(typedNodes.contradiction_fact, 265, -Math.PI / 2 + 0.3);
    placeRing(typedNodes.temporal_fact, 330, -Math.PI / 3);
    placeRing(typedNodes.source, 410, -Math.PI / 2);

    const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
    const radii = {{
      claim: 17,
      fact: 11,
      source: 9,
      contradiction_fact: 11,
      temporal_fact: 10,
      default: 8
    }};

    let selectedNode = null;

    function draw() {{
      ctx.clearRect(0, 0, width, height);

      edges.forEach(edge => {{
        const from = nodeById[edge.from];
        const to = nodeById[edge.to];
        if (!from || !to) return;
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.strokeStyle = "rgba(31,42,44,0.18)";
        ctx.lineWidth = Math.max(1, (edge.weight || 0.5) * 1.6);
        ctx.stroke();
      }});

      nodes.forEach(node => {{
        const color = colors[node.node_type] || colors.default;
        const radius = radii[node.node_type] || radii.default;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        if (selectedNode && selectedNode.id === node.id) {{
          ctx.lineWidth = 3;
          ctx.strokeStyle = "#1f2a2c";
          ctx.stroke();
        }}

        ctx.fillStyle = "#1f2a2c";
        ctx.font = "12px Segoe UI";
        const label = (node.source_name || node.text || node.id || "").slice(0, node.node_type === "source" ? 24 : 30);
        ctx.fillText(label, node.x + radius + 6, node.y + 4);
      }});
    }}

    function setSelection(node) {{
      selectedNode = node;
      const text = node.text || node.source_name || node.id || "";
      const extra = [
        `Type: ${{node.node_type || "unknown"}}`,
        node.pillar ? `Pillar: ${{node.pillar}}` : "",
        node.year ? `Year: ${{node.year}}` : "",
        node.verifiability_score !== undefined ? `Verifiability: ${{node.verifiability_score}}` : "",
        node.source_type ? `Source Type: ${{node.source_type}}` : "",
        node.url ? `URL: ${{node.url}}` : "",
      ].filter(Boolean);
      document.getElementById("selectionText").innerHTML = `
        <strong>${{text || "Untitled node"}}</strong>
        <div style="margin-top:8px; color:#667275; font-size:13px; line-height:1.5;">${{extra.join("<br>")}}</div>
      `;
      draw();
    }}

    canvas.addEventListener("click", event => {{
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (event.clientX - rect.left) * scaleX;
      const y = (event.clientY - rect.top) * scaleY;
      let hit = null;
      nodes.forEach(node => {{
        const radius = radii[node.node_type] || radii.default;
        const dist = Math.hypot(node.x - x, node.y - y);
        if (dist <= radius + 5) {{
          hit = node;
        }}
      }});
      if (hit) {{
        setSelection(hit);
      }}
    }});

    draw();
    if (typedNodes.claim[0]) {{
      setSelection(typedNodes.claim[0]);
    }}
  </script>
</body>
</html>
"""
    output_html_path.write_text(html, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ESG fact graph to HTML.")
    parser.add_argument("--graph", help="Path to an existing fact graph JSON artifact.", default="")
    parser.add_argument("--company", help="Company name for a local demo graph.", default="JPMorgan Chase")
    parser.add_argument("--claim", help="Claim text for local demo graph.", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    graph_path = Path(args.graph) if args.graph else None

    if graph_path and graph_path.exists():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        source_json_path = str(graph_path)
    else:
        company = args.company.strip()
        claim = args.claim.strip() or DEFAULT_CLAIMS.get(company.lower(), "")
        if not claim:
            raise SystemExit("No claim supplied and no default claim available for that company.")

        graph = _local_demo_graph(company=company, claim=claim)
        source_json_path = persist_fact_graph(
            fact_graph=graph,
            company=company,
            report_id="local_demo",
        )

    html_dir = PROJECT_ROOT / "reports" / "fact_graphs"
    html_dir.mkdir(parents=True, exist_ok=True)
    html_name = f"{_slugify(graph.get('company', 'graph'))}_viewer.html"
    html_path = html_dir / html_name
    _render_html(graph=graph, source_json_path=source_json_path, output_html_path=html_path)

    print("Fact graph viewer generated.")
    print(f"JSON: {source_json_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
