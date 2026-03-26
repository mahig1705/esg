"""
esg_data_apis.py
================
Structured API clients for WBA (World Benchmarking Alliance) and
WRI Aqueduct 4.0 to fill Social, Governance, and Water-Risk pillar
scores that IR page scraping fails to provide.

Dependencies
------------
    pip install httpx python-dotenv
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USER_AGENT = (
    "ESGLens/1.0 (greenwashing-verification-platform; "
    "https://github.com/mahig1705/esg; research-use)"
)
_TIMEOUT = 20  # seconds

# WBA endpoints
_WBA_BASE = "https://api.worldbenchmarkingalliance.org/v1"

# WRI Aqueduct 4.0 endpoint
_WRI_BASE = "https://api.resourcewatch.org/v1/aqueduct"

# WRI indicator → risk category mapping (Aqueduct 4.0, all 13 indicators)
_WRI_PHYSICAL: list[str] = ["bws", "bwd", "iav", "sev", "gtd", "rfr", "cfr", "drr"]
_WRI_REGULATORY: list[str] = ["udw", "usa", "cep"]
_WRI_REPUTATIONAL: list[str] = ["rri", "eco"]


# ===================================================================
# WBA — World Benchmarking Alliance
# ===================================================================

def get_wba_company_assessment(
    company_name: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch a company's WBA assessment via the WBA API v1.

    **Step 1** — Search for the company by name.
    **Step 2** — Fetch the full assessment using the company ID.

    Parameters
    ----------
    company_name : str
        Company name to search for (e.g. ``"Shell"``).
    api_key : str | None
        WBA API key.  Falls back to ``os.environ.get("WBA_API_KEY")``.

    Returns
    -------
    dict
        Keys: ``found``, ``company_name``, ``scores``, ``indicators``,
        ``raw``, ``error``.
    """
    result: dict[str, Any] = {
        "found": False,
        "company_name": company_name,
        "scores": {},
        "indicators": {},
        "raw": None,
        "error": None,
    }

    api_key = api_key or os.environ.get("WBA_API_KEY")
    if not api_key:
        result["error"] = "WBA_API_KEY not set"
        logger.warning("WBA: API key not available — skipping")
        return result

    headers = {
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    # --- Step 1: search for company ---
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            search_resp = client.get(
                f"{_WBA_BASE}/companies",
                headers=headers,
                params={"search": company_name, "limit": 5},
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()

            companies = search_data.get("data") or search_data.get("results") or []
            if isinstance(search_data, list):
                companies = search_data

            if not companies:
                result["error"] = f"No WBA company found for '{company_name}'"
                logger.info("WBA: no results for '%s'", company_name)
                return result

            company = companies[0]
            company_id = company.get("id") or company.get("company_id")
            result["company_name"] = company.get("name", company_name)

            # --- Step 2: fetch full assessment ---
            assess_resp = client.get(
                f"{_WBA_BASE}/companies/{company_id}/assessments",
                headers=headers,
            )
            assess_resp.raise_for_status()
            assess_data = assess_resp.json()
            result["raw"] = assess_data

            # Parse pillar scores
            assessments = (
                assess_data.get("data")
                or assess_data.get("assessments")
                or (assess_data if isinstance(assess_data, list) else [assess_data])
            )

            for assessment in assessments:
                scores = assessment.get("scores") or assessment.get("pillar_scores") or {}
                if isinstance(scores, dict):
                    for key in ("social", "governance", "environment", "total"):
                        if key in scores and scores[key] is not None:
                            result["scores"][key] = scores[key]

                # Parse individual indicators
                indicators = (
                    assessment.get("indicators")
                    or assessment.get("measurements")
                    or {}
                )
                if isinstance(indicators, list):
                    for ind in indicators:
                        name = ind.get("name") or ind.get("indicator_name", "unknown")
                        val = ind.get("score") or ind.get("value")
                        result["indicators"][name] = val
                elif isinstance(indicators, dict):
                    result["indicators"].update(indicators)

            result["found"] = bool(result["scores"] or result["indicators"])
            logger.info(
                "WBA: fetched assessment for '%s' — %d pillar scores, %d indicators",
                result["company_name"],
                len(result["scores"]),
                len(result["indicators"]),
            )

    except httpx.TimeoutException:
        result["error"] = f"WBA API timed out for '{company_name}'"
        logger.warning("WBA: request timed out for '%s'", company_name)
    except httpx.HTTPStatusError as exc:
        result["error"] = f"WBA API HTTP {exc.response.status_code}"
        logger.warning("WBA: HTTP %d for '%s'", exc.response.status_code, company_name)
    except httpx.HTTPError as exc:
        result["error"] = f"WBA API error: {exc}"
        logger.warning("WBA: request failed for '%s' — %s", company_name, exc)
    except (ValueError, KeyError, TypeError) as exc:
        result["error"] = f"WBA parse error: {exc}"
        logger.warning("WBA: failed to parse response for '%s' — %s", company_name, exc)

    return result


# ===================================================================
# WRI Aqueduct 4.0
# ===================================================================

def get_wri_water_risk(
    lat: float,
    lon: float,
    industry: str | None = None,
) -> dict[str, Any]:
    """Fetch water risk data from WRI Aqueduct via the Resource Watch API.

    Uses a two-step workflow:
        1. Register a GeoJSON geometry as a geostore on Resource Watch.
        2. Query the Aqueduct analysis endpoint using the geostore ID.

    Categorizes the 13 Aqueduct indicators into physical risk (bws, bwd,
    iav, sev, gtd, rfr, cfr, drr), regulatory risk (udw, usa, cep),
    and reputational risk (rri, eco).

    Parameters
    ----------
    lat : float
        Latitude of the location (e.g. company HQ).
    lon : float
        Longitude of the location.
    industry : str | None
        Optional industry filter for sector-weighted scoring.

    Returns
    -------
    dict
        Keys: ``found``, ``location``, ``overall_risk``,
        ``physical_risk``, ``regulatory_risk``, ``reputational_risk``,
        ``raw``, ``error``.
    """
    result: dict[str, Any] = {
        "found": False,
        "location": {"lat": lat, "lon": lon},
        "overall_risk": None,
        "physical_risk": {},
        "regulatory_risk": {},
        "reputational_risk": {},
        "raw": None,
        "error": None,
    }

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    # Create a small polygon buffer around the point (~1km) since
    # the Aqueduct raster may not intersect a bare Point geometry.
    BUFFER = 0.01  # ~1km in degrees
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - BUFFER, lat - BUFFER],
                    [lon + BUFFER, lat - BUFFER],
                    [lon + BUFFER, lat + BUFFER],
                    [lon - BUFFER, lat + BUFFER],
                    [lon - BUFFER, lat - BUFFER],
                ]],
            },
            "properties": {},
        }],
    }

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            # --- Step 1: Create geostore ---
            gs_resp = client.post(
                "https://api.resourcewatch.org/v1/geostore",
                json={"geojson": geojson},
                headers=headers,
            )
            gs_resp.raise_for_status()
            geostore_id = gs_resp.json().get("data", {}).get("id")

            if not geostore_id:
                result["error"] = "Failed to create geostore (no ID returned)"
                logger.warning("WRI: geostore creation returned no ID")
                return result

            logger.info("WRI: geostore created — id=%s", geostore_id)

            # --- Step 2: Query Aqueduct analysis ---
            params: dict[str, str] = {
                "geostore": geostore_id,
                "wscheme": "aqueduct",
                "analysis_type": "annual",
                "year": "baseline",
            }
            if industry:
                params["industry"] = industry

            analysis_resp = client.get(
                "https://api.resourcewatch.org/v1/aqueduct/analysis",
                headers=headers,
                params=params,
            )
            analysis_resp.raise_for_status()
            data = analysis_resp.json()
            result["raw"] = data

            # Parse indicator values from response
            indicators: dict[str, Any] = {}

            # Handle various response shapes
            risk_data = (
                data.get("data")
                or data.get("rows")
                or data.get("indicators")
                or data
            )
            if isinstance(risk_data, list) and risk_data:
                risk_data = risk_data[0]
            if isinstance(risk_data, dict):
                attrs = risk_data.get("attributes") or risk_data
                for key, val in attrs.items():
                    indicators[key.lower()] = val

            # Categorize indicators into risk buckets
            for ind_key in _WRI_PHYSICAL:
                for full_key, val in indicators.items():
                    if ind_key in full_key and val is not None:
                        result["physical_risk"][ind_key] = val
                        break

            for ind_key in _WRI_REGULATORY:
                for full_key, val in indicators.items():
                    if ind_key in full_key and val is not None:
                        result["regulatory_risk"][ind_key] = val
                        break

            for ind_key in _WRI_REPUTATIONAL:
                for full_key, val in indicators.items():
                    if ind_key in full_key and val is not None:
                        result["reputational_risk"][ind_key] = val
                        break

            # Compute overall risk (average of available numeric values)
            all_numeric = [
                v for bucket in (
                    result["physical_risk"],
                    result["regulatory_risk"],
                    result["reputational_risk"],
                )
                for v in bucket.values()
                if isinstance(v, (int, float))
            ]
            if all_numeric:
                result["overall_risk"] = round(
                    sum(all_numeric) / len(all_numeric), 3,
                )

            result["found"] = bool(
                result["physical_risk"]
                or result["regulatory_risk"]
                or result["reputational_risk"]
            )
            logger.info(
                "WRI: water risk for (%.4f, %.4f) — overall=%s, "
                "physical=%d, regulatory=%d, reputational=%d indicators",
                lat, lon, result["overall_risk"],
                len(result["physical_risk"]),
                len(result["regulatory_risk"]),
                len(result["reputational_risk"]),
            )

    except httpx.TimeoutException:
        result["error"] = f"WRI API timed out for ({lat}, {lon})"
        logger.warning("WRI: request timed out for (%.4f, %.4f)", lat, lon)
    except httpx.HTTPStatusError as exc:
        result["error"] = f"WRI API HTTP {exc.response.status_code}"
        logger.warning(
            "WRI: HTTP %d for (%.4f, %.4f)", exc.response.status_code, lat, lon,
        )
    except httpx.HTTPError as exc:
        result["error"] = f"WRI API error: {exc}"
        logger.warning("WRI: request failed for (%.4f, %.4f) — %s", lat, lon, exc)
    except (ValueError, KeyError, TypeError) as exc:
        result["error"] = f"WRI parse error: {exc}"
        logger.warning(
            "WRI: failed to parse response for (%.4f, %.4f) — %s", lat, lon, exc,
        )

    return result


# ===================================================================
# Top-level pillar filler
# ===================================================================

def fill_missing_pillars(
    company_name: str,
    lat: float | None = None,
    lon: float | None = None,
    existing_scores: dict[str, Any] | None = None,
    wba_api_key: str | None = None,
) -> dict[str, Any]:
    """Fill missing Social, Governance, and Water-Risk pillar scores.

    Called when either social or governance pillar is ``None`` in the
    existing scores.  Queries WBA for pillar scores and WRI Aqueduct
    for water risk data.

    Parameters
    ----------
    company_name : str
        Company name for WBA lookup.
    lat, lon : float | None
        HQ coordinates for WRI Aqueduct lookup.
    existing_scores : dict | None
        Current pillar scores dict.  Missing/``None`` values trigger
        API lookups.
    wba_api_key : str | None
        Optional WBA API key override.

    Returns
    -------
    dict
        Updated scores dict with ``_sources`` annotation showing
        which API filled each value.
    """
    scores = dict(existing_scores or {})
    sources: dict[str, str] = {}

    social_missing = scores.get("social") is None
    governance_missing = scores.get("governance") is None
    water_missing = scores.get("water_risk") is None

    # --- WBA: fill social & governance ---
    if social_missing or governance_missing:
        logger.info(
            "Filling missing pillars via WBA for '%s' "
            "(social=%s, governance=%s)",
            company_name,
            "missing" if social_missing else "present",
            "missing" if governance_missing else "present",
        )
        wba = get_wba_company_assessment(company_name, api_key=wba_api_key)

        if wba["found"]:
            if social_missing and "social" in wba["scores"]:
                scores["social"] = wba["scores"]["social"]
                sources["social"] = "WBA"

            if governance_missing and "governance" in wba["scores"]:
                scores["governance"] = wba["scores"]["governance"]
                sources["governance"] = "WBA"

            # Bonus: fill environment if also missing
            if scores.get("environment") is None and "environment" in wba["scores"]:
                scores["environment"] = wba["scores"]["environment"]
                sources["environment"] = "WBA"

            scores["_wba_indicators"] = wba["indicators"]
            scores["_wba_company_name"] = wba["company_name"]
        else:
            logger.warning(
                "WBA lookup failed for '%s': %s",
                company_name, wba.get("error"),
            )

    # --- WRI: fill water risk ---
    if water_missing and lat is not None and lon is not None:
        logger.info(
            "Filling water_risk via WRI Aqueduct for (%.4f, %.4f)",
            lat, lon,
        )
        wri = get_wri_water_risk(lat, lon)

        if wri["found"]:
            scores["water_risk"] = wri["overall_risk"]
            scores["water_risk_physical"] = wri["physical_risk"]
            scores["water_risk_regulatory"] = wri["regulatory_risk"]
            scores["water_risk_reputational"] = wri["reputational_risk"]
            sources["water_risk"] = "WRI_Aqueduct_4.0"
        else:
            logger.warning(
                "WRI Aqueduct lookup failed for (%.4f, %.4f): %s",
                lat, lon, wri.get("error"),
            )

    scores["_sources"] = sources
    logger.info("fill_missing_pillars result sources: %s", sources)
    return scores
