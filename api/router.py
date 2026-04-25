"""
api/router.py
-------------
Main API router — mounts all sub-routers into a single include.
"""
from __future__ import annotations

from fastapi import APIRouter

from api.analysis import router as analysis_router
from api.reports import router as reports_router
from api.upload import router as upload_router
from api.pipeline_ws import router as ws_router

api_router = APIRouter()

api_router.include_router(analysis_router)
api_router.include_router(reports_router)
api_router.include_router(upload_router)
api_router.include_router(ws_router)
