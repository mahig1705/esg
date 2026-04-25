"""
api/reports.py
--------------
GET /api/reports          → list of all completed reports (HistoryEntry)
GET /api/reports/{id}     → full ESGReport by id
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from api.mappers import map_report_to_history, map_report_to_schema, _short_id
from api.models import ESGReport, HistoryEntry

router = APIRouter(prefix="/api")

REPORTS_DIR = Path(os.getenv("ESG_REPORTS_DIR", "reports"))


def _iter_report_jsons():
    """Yield (report_id, parsed_dict) for every valid JSON report on disk."""
    if not REPORTS_DIR.exists():
        return
    for p in sorted(REPORTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        # Skip lineage / debug files
        if "lineage" in p.stem or "FULL" in p.stem or "research_runs" in p.stem:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "scores" in data:
                yield _short_id(str(p)), data
        except Exception:
            continue


@router.get("/reports", response_model=List[HistoryEntry])
def get_all_reports():
    """Return all completed analyses for the History page."""
    entries = []
    for report_id, raw in _iter_report_jsons():
        try:
            entries.append(map_report_to_history(raw, report_id))
        except Exception:
            continue
    return entries


@router.get("/reports/{report_id}", response_model=ESGReport)
def get_report(report_id: str):
    """Return the full report for a given ID."""
    if not REPORTS_DIR.exists():
        raise HTTPException(status_code=404, detail="No reports directory found")

    # Search by report_id (which is derived from the filename stem)
    for p in REPORTS_DIR.glob("*.json"):
        if "lineage" in p.stem or "FULL" in p.stem or "research_runs" in p.stem:
            continue
        if _short_id(str(p)) == report_id or p.stem == report_id:
            try:
                with open(p, encoding="utf-8") as f:
                    raw = json.load(f)
                return map_report_to_schema(raw, report_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse report: {e}")

    raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
