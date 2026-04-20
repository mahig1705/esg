"""
Summarize research telemetry logs into aggregate metrics.

Usage:
    venv\\Scripts\\python.exe scripts\\summarize_research_runs.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.research_telemetry import summarize_runs  # noqa: E402


def main() -> int:
    summary = summarize_runs("reports/research_runs.jsonl")
    out_path = Path("reports/research_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Research summary generated:")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

