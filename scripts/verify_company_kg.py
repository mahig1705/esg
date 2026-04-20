r"""
Verification helper for the company-centric Neo4j ESG knowledge graph.

Usage:
    venv\Scripts\python.exe scripts\verify_company_kg.py --scenario shell
    venv\Scripts\python.exe scripts\verify_company_kg.py --scenario microsoft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from core.company_knowledge_graph import CompanyKnowledgeGraph  # noqa: E402


def _shell_state() -> dict:
    return {
        "company": "Shell",
        "industry": "Energy",
        "claim": "We are on track to be a net-zero emissions energy business by 2050.",
        "evidence": [
            {
                "source_id": "shell_annual_report_2023",
                "relevant_text": "Shell reported its pathway to become a net-zero emissions energy business by 2050 in its 2023 annual reporting.",
                "source_name": "Shell Annual Report 2023",
                "source_type": "Company-Controlled",
                "url": "https://www.shell.com/",
                "year": 2023,
            },
            {
                "source_id": "shell_court_ruling_2021",
                "relevant_text": "Dutch court ruled Shell must reduce absolute emissions by 45% by 2030 versus 2019 levels.",
                "source_name": "Dutch Court Ruling",
                "source_type": "Legal/Court Documents",
                "url": "https://uitspraken.rechtspraak.nl/inziendocument?id=ECLI:NL:RBDHA:2021:5339",
                "year": 2021,
            },
        ],
        "carbon_extraction": {
            "emissions": {
                "scope1": {"value": 517000000.0, "unit": "tCO2e", "year": 2023, "source_type": "Company-Controlled"},
            }
        },
        "agent_outputs": [
            {
                "agent": "contradiction_analysis",
                "output": [
                    {
                        "description": "Dutch court ruled Shell must reduce absolute emissions by 45% by 2030 versus 2019 levels.",
                        "contradiction_text": "Dutch court ruled Shell must reduce absolute emissions by 45% by 2030 versus 2019 levels.",
                        "source": "Milieudefensie v Shell, District Court The Hague, May 2021",
                        "source_url": "https://uitspraken.rechtspraak.nl/inziendocument?id=ECLI:NL:RBDHA:2021:5339",
                        "source_type": "Legal/Court Documents",
                        "year": 2021,
                        "severity": "high",
                        "regulatory_body": "Dutch District Court, The Hague",
                    }
                ],
            }
        ],
    }


def _microsoft_state() -> dict:
    return {
        "company": "Microsoft",
        "industry": "Technology",
        "claim": "Microsoft aims to be carbon negative by 2030.",
        "evidence": [
            {
                "source_id": "microsoft_2023",
                "relevant_text": "Microsoft reported GHG intensity evidence for 2023.",
                "source_name": "Microsoft Climate Report 2023",
                "source_type": "Company-Controlled",
                "url": "https://www.microsoft.com/",
                "year": 2023,
            },
            {
                "source_id": "microsoft_2024",
                "relevant_text": "Microsoft reported GHG intensity evidence for 2024.",
                "source_name": "Microsoft Climate Report 2024",
                "source_type": "Company-Controlled",
                "url": "https://www.microsoft.com/",
                "year": 2024,
            },
            {
                "source_id": "microsoft_2025",
                "relevant_text": "Microsoft reported GHG intensity evidence for 2025.",
                "source_name": "Microsoft Climate Report 2025",
                "source_type": "Tier-1 Financial Media",
                "url": "https://news.microsoft.com/on-the-issues/2025/05/26/building-new-markets-to-advance-sustainability/",
                "year": 2025,
            },
        ],
        "json_export": json.dumps(
            {
                "pillar_factors": {
                    "environmental": {
                        "sub_indicators": [
                            {
                                "name": "GHG Emissions Intensity",
                                "raw_value": "Scope 1: 4800000.0 tCO2e",
                                "unit": "tCO2e",
                                "data_year": 2023,
                                "verified": True,
                                "data_source": "Microsoft Climate Report 2023",
                                "source_url": "https://www.microsoft.com/",
                            },
                            {
                                "name": "GHG Emissions Intensity",
                                "raw_value": "Scope 1: 5200000.0 tCO2e",
                                "unit": "tCO2e",
                                "data_year": 2024,
                                "verified": True,
                                "data_source": "Microsoft Climate Report 2024",
                                "source_url": "https://www.microsoft.com/",
                            },
                            {
                                "name": "GHG Emissions Intensity",
                                "raw_value": "Scope 1: 6057000.0 tCO2e",
                                "unit": "tCO2e",
                                "data_year": 2025,
                                "verified": False,
                                "data_source": "Reuters / Microsoft sustainability coverage",
                                "source_url": "https://news.microsoft.com/on-the-issues/2025/05/26/building-new-markets-to-advance-sustainability/",
                            },
                        ]
                    }
                }
            }
        ),
        "agent_outputs": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Neo4j company-centric KG scenarios.")
    parser.add_argument("--scenario", choices=["shell", "microsoft"], required=True)
    args = parser.parse_args()

    state = _shell_state() if args.scenario == "shell" else _microsoft_state()
    status = CompanyKnowledgeGraph().ingest_state(state)

    print("Company KG verification scenario completed.")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
