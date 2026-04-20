"""
Persistence helpers for fact-centric ESG graph artifacts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import re


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip())
    return text.strip("_") or "unknown"


def persist_fact_graph(
    fact_graph: Dict[str, Any],
    company: str,
    report_id: str | None = None,
    output_dir: str = "reports/fact_graphs",
) -> str:
    if not isinstance(fact_graph, dict) or not fact_graph:
        raise ValueError("Fact graph payload is empty or invalid.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    company_slug = _slugify(company)
    report_slug = _slugify(report_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    path = out_dir / f"{company_slug}_{report_slug}_fact_graph.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(fact_graph, f, indent=2, ensure_ascii=False)

    return str(path)
