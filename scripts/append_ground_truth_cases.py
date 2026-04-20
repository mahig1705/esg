"""
Append validated new ground-truth rows into data/ground_truth_dataset.csv.

Usage:
    venv\\Scripts\\python.exe scripts\\append_ground_truth_cases.py

Input file expected:
    data/ground_truth_additions.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_COLUMNS = [
    "company_name",
    "sector",
    "jurisdiction",
    "claim_text",
    "greenwashing_label",
    "confidence",
    "source_url",
    "year",
    "case_type",
    "regulatory_body",
    "wikirate_ghg",
    "wikirate_ghg_year",
]


def main() -> int:
    base_path = PROJECT_ROOT / "data" / "ground_truth_dataset.csv"
    additions_path = PROJECT_ROOT / "data" / "ground_truth_additions.csv"

    if not additions_path.exists():
        print(f"Input file missing: {additions_path}")
        print("Create it from data/ground_truth_additions_template.csv first.")
        return 1

    base = pd.read_csv(base_path)
    add = pd.read_csv(additions_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in add.columns]
    if missing:
        print(f"Additions file missing required columns: {missing}")
        return 1

    add = add.copy()
    add["greenwashing_label"] = add["greenwashing_label"].astype(int)
    invalid_labels = add[~add["greenwashing_label"].isin([0, 1])]
    if not invalid_labels.empty:
        print("Invalid greenwashing_label values detected (must be 0 or 1).")
        return 1

    for col in ["company_name", "claim_text", "sector"]:
        if add[col].astype(str).str.strip().eq("").any():
            print(f"Column '{col}' has empty values in additions.")
            return 1

    combined = pd.concat([base, add[base.columns.intersection(add.columns)]], ignore_index=True)
    # Keep first occurrence by key; incoming duplicates are dropped.
    dedupe_key = ["company_name", "claim_text", "year"]
    for key in dedupe_key:
        if key not in combined.columns:
            print(f"Cannot dedupe: missing key column '{key}'")
            return 1
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedupe_key, keep="first")
    after = len(combined)
    added = after - len(base)
    dropped_duplicates = before - after

    combined.to_csv(base_path, index=False)
    print(f"Ground truth dataset updated: +{added} rows appended.")
    print(f"Duplicates dropped during merge: {dropped_duplicates}")
    print(f"New dataset size: {len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

