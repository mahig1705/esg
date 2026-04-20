"""
Refresh materiality profiles from remote sources and write config/materiality_map.json.

Usage (venv):
    venv\\Scripts\\python.exe scripts\\refresh_materiality_profiles.py

Optional env vars:
    MATERIALITY_PROFILE_URL=<remote json overlay url>
    SASB_MATERIALITY_DATA_URL=<sasb-like csv/json dataset url>
    MATERIALITY_PROFILE_PATH=<target path>  # default: config/materiality_map.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

# Ensure project root imports resolve when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.materiality_profile_loader import load_materiality_profiles


def main() -> int:
    load_dotenv()

    target_path = os.getenv("MATERIALITY_PROFILE_PATH", "config/materiality_map.json")
    remote_profile_url = os.getenv("MATERIALITY_PROFILE_URL", "").strip()
    sasb_dataset_url = os.getenv("SASB_MATERIALITY_DATA_URL", "").strip()

    profiles = load_materiality_profiles(
        local_path=target_path,
        remote_profile_url=remote_profile_url,
        sasb_dataset_url=sasb_dataset_url,
        timeout=15,
    )
    if not isinstance(profiles, dict) or not profiles:
        print("No materiality profiles loaded. Keeping existing file unchanged.")
        return 1

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"Materiality profiles refreshed: {len(profiles)} entries -> {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
