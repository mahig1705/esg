"""
api/reports.py
--------------
GET /api/reports          → list of all completed reports (HistoryEntry)
GET /api/reports/{id}     → full ESGReport by id
GET /api/reports/{id}/pdf → audit-ready PDF download
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

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


def _find_raw(report_id: str):
    """Return (raw_dict, path) for a given report_id, or raise 404."""
    if not REPORTS_DIR.exists():
        raise HTTPException(status_code=404, detail="No reports directory found")
    for p in REPORTS_DIR.glob("*.json"):
        if "lineage" in p.stem or "FULL" in p.stem or "research_runs" in p.stem:
            continue
        if _short_id(str(p)) == report_id or p.stem == report_id:
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f), p
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse report: {e}")
    raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")


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
    raw, _ = _find_raw(report_id)
    return map_report_to_schema(raw, report_id)


@router.get("/reports/{report_id}/pdf")
def get_report_pdf(report_id: str):
    """Generate and return an audit-ready PDF for the given report ID."""
    raw, _ = _find_raw(report_id)
    try:
        report_schema = map_report_to_schema(raw, report_id)
        report_dict = report_schema.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build report model: {e}")

    try:
        from api.pdf_generator import build_pdf
        pdf_bytes = build_pdf(report_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    company = report_dict.get("company", "Report").replace(" ", "_").replace("&", "and")
    filename = f"ESGLens_{company}_{report_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
