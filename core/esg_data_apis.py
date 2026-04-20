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
import time
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
_WBA_RATE_LIMIT_SECONDS = 1.05
_WBA_MAX_RETRIES = 3

# WBA endpoints
_WBA_BASE = "https://data.worldbenchmarkingalliance.org/api/data/v1"

# WRI Aqueduct 4.0 endpoint
_WRI_BASE = "https://api.resourcewatch.org/v1/aqueduct"

# WRI indicator → risk category mapping (Aqueduct 4.0, all 13 indicators)
_WRI_PHYSICAL: list[str] = ["bws", "bwd", "iav", "sev", "gtd", "rfr", "cfr", "drr"]
_WRI_REGULATORY: list[str] = ["udw", "usa", "cep"]
_WRI_REPUTATIONAL: list[str] = ["rri", "eco"]


# ===================================================================
# WBA — World Benchmarking Alliance
# ===================================================================

def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_score_0_100(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None

    # WBA fields appear in mixed scales across datasets.
    # Normalize common ranges into a 0-100 score for internal scoring.
    if parsed <= 1.0:
        parsed = parsed * 100.0
    elif parsed <= 5.0:
        parsed = parsed * 20.0

    return max(0.0, min(100.0, float(parsed)))


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _extract_http_detail(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None

    if isinstance(payload, dict):
        detail = payload.get("detail")
        return str(detail) if detail else None
    return None


def _extract_company_coordinates(company_row: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(company_row, dict):
        return None, None

    def _candidate_number(keys: tuple[str, ...]) -> float | None:
        for key, value in company_row.items():
            key_l = str(key).lower()
            if any(k in key_l for k in keys):
                parsed = _safe_float(value)
                if parsed is not None:
                    return parsed
        return None

    lat = _candidate_number(("lat", "latitude"))
    lon = _candidate_number(("lon", "lng", "longitude"))

    if lat is None or lon is None:
        return None, None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None
    return lat, lon


def _wba_dataset_get(
    client: httpx.Client,
    dataset_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    url = f"{_WBA_BASE}/datasets/{dataset_name}"
    last_error: Exception | None = None

    for attempt in range(_WBA_MAX_RETRIES):
        try:
            response = client.get(url, params=params)
            if response.status_code == 429:
                retry_after_raw = response.headers.get("Retry-After")
                retry_after = _safe_float(retry_after_raw) or _WBA_RATE_LIMIT_SECONDS
                wait_seconds = max(_WBA_RATE_LIMIT_SECONDS, retry_after)
                logger.warning(
                    "WBA rate limit hit for dataset '%s' (attempt %d/%d); sleeping %.2fs",
                    dataset_name,
                    attempt + 1,
                    _WBA_MAX_RETRIES,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list):
                return {"data": payload, "meta": {}}
            return {"data": [], "meta": {}}
        except httpx.TimeoutException as exc:
            last_error = exc
            if attempt + 1 < _WBA_MAX_RETRIES:
                time.sleep(_WBA_RATE_LIMIT_SECONDS)
                continue
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if 500 <= status < 600 and attempt + 1 < _WBA_MAX_RETRIES:
                backoff = _WBA_RATE_LIMIT_SECONDS * (attempt + 1)
                time.sleep(backoff)
                continue
            raise
        except (TypeError, ValueError) as exc:
            last_error = exc
            break

    if last_error is not None:
        raise last_error
    return {"data": [], "meta": {}}


def _pick_best_company_match(rows: list[dict[str, Any]], company_name: str) -> dict[str, Any] | None:
    if not rows:
        return None

    target = _normalize_name(company_name)

    def _rank(row: dict[str, Any]) -> tuple[int, int, int]:
        candidate_name = row.get("company_name") or row.get("name") or ""
        normalized_candidate = _normalize_name(candidate_name)

        is_exact = 0 if normalized_candidate == target else 1
        contains_target = 0 if target and target in normalized_candidate else 1
        length_distance = abs(len(normalized_candidate) - len(target))
        return (is_exact, contains_target, length_distance)

    return sorted(rows, key=_rank)[0]


def _extract_row_score(row: dict[str, Any]) -> float | None:
    preferred_keys = [
        "benchmark_score_numerical",
        "score_numerical",
        "indicator_score_numerical",
        "score",
        "value",
    ]
    for key in preferred_keys:
        if key in row:
            parsed = _normalize_score_0_100(row.get(key))
            if parsed is not None:
                return parsed

    for key, val in row.items():
        parsed = _normalize_score_0_100(val)
        if parsed is None:
            continue
        key_l = str(key).lower()
        if "score" in key_l or "value" in key_l:
            return parsed
    return None


def _collect_wba_scores(
    benchmark_rows: list[dict[str, Any]],
    indicator_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    pillar_values: dict[str, list[float]] = {
        "social": [],
        "governance": [],
        "environment": [],
    }
    indicators: dict[str, float] = {}

    pillar_keywords: dict[str, tuple[str, ...]] = {
        "social": ("social", "worker", "human rights", "inclusion", "community"),
        "governance": ("governance", "board", "ethic", "transparency", "corruption"),
        "environment": ("environment", "climate", "carbon", "emission", "energy", "water", "nature"),
    }

    total_candidates: list[float] = []

    for row in benchmark_rows + indicator_rows:
        if not isinstance(row, dict):
            continue

        row_score = _extract_row_score(row)
        row_context = " ".join(
            str(row.get(k, ""))
            for k in (
                "benchmark_name",
                "measurement_area_name",
                "indicator_name",
                "element_name",
                "attribute_name",
                "name",
            )
        ).lower()

        if row_score is not None:
            if "benchmark_score_numerical" in row:
                total_candidates.append(row_score)

            indicator_name = (
                row.get("indicator_name")
                or row.get("measurement_area_name")
                or row.get("element_name")
                or row.get("benchmark_name")
                or row.get("name")
            )
            if indicator_name:
                indicators[str(indicator_name)] = row_score

            for pillar, keywords in pillar_keywords.items():
                if any(keyword in row_context for keyword in keywords):
                    pillar_values[pillar].append(row_score)

        # Also map explicit numeric columns such as social_score/governance_score fields.
        for key, value in row.items():
            key_lower = str(key).lower()
            parsed = _normalize_score_0_100(value)
            if parsed is None:
                continue

            if "social" in key_lower:
                pillar_values["social"].append(parsed)
            if "governance" in key_lower:
                pillar_values["governance"].append(parsed)
            if "environment" in key_lower or "climate" in key_lower:
                pillar_values["environment"].append(parsed)
            if key_lower in ("benchmark_score_numerical", "total", "overall_score"):
                total_candidates.append(parsed)

    scores: dict[str, float] = {}
    for pillar, values in pillar_values.items():
        if values:
            scores[pillar] = round(sum(values) / len(values), 3)

    if total_candidates:
        scores["total"] = round(sum(total_candidates) / len(total_candidates), 3)
    elif indicators:
        scores["total"] = round(sum(indicators.values()) / len(indicators), 3)

    return scores, indicators


def get_wba_company_assessment(
    company_name: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch WBA company data via the WBA Data API datasets endpoints.

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
        "hq_coordinates": {"lat": None, "lon": None},
        "raw": None,
        "error": None,
    }

    api_key = (api_key or os.environ.get("WBA_API_KEY") or "").strip()
    if not api_key:
        result["error"] = "WBA_API_KEY not set"
        logger.warning("WBA: API key not available — skipping")
        return result

    headers = {
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=_TIMEOUT, headers=headers) as client:
            company_payload = _wba_dataset_get(
                client,
                "companies",
                {
                    "company_name__co": company_name,
                    "per_page": 50,
                    "page": 1,
                },
            )
            companies = company_payload.get("data") if isinstance(company_payload, dict) else []
            companies = [row for row in (companies or []) if isinstance(row, dict)]

            if not companies:
                result["error"] = f"No WBA company found for '{company_name}'"
                logger.info("WBA: no company rows found for '%s'", company_name)
                return result

            company = _pick_best_company_match(companies, company_name) or companies[0]
            company_id = company.get("company_id") or company.get("id")
            result["company_name"] = str(company.get("company_name") or company.get("name") or company_name)
            lat, lon = _extract_company_coordinates(company)
            result["hq_coordinates"] = {"lat": lat, "lon": lon}

            time.sleep(_WBA_RATE_LIMIT_SECONDS)

            benchmark_params: dict[str, Any] = {
                "per_page": 1000,
                "page": 1,
                "order_by": "methodology_year",
                "ascending": "false",
            }
            if company_id:
                benchmark_params["company_id"] = company_id
            else:
                benchmark_params["company_name__co"] = result["company_name"]

            benchmark_payload = _wba_dataset_get(client, "benchmarks", benchmark_params)
            benchmark_rows = benchmark_payload.get("data") if isinstance(benchmark_payload, dict) else []
            benchmark_rows = [row for row in (benchmark_rows or []) if isinstance(row, dict)]

            time.sleep(_WBA_RATE_LIMIT_SECONDS)

            indicator_params: dict[str, Any] = {
                "per_page": 1000,
                "page": 1,
            }
            if company_id:
                indicator_params["company_id"] = company_id
            else:
                indicator_params["company_name__co"] = result["company_name"]

            indicator_payload = _wba_dataset_get(client, "indicators", indicator_params)
            indicator_rows = indicator_payload.get("data") if isinstance(indicator_payload, dict) else []
            indicator_rows = [row for row in (indicator_rows or []) if isinstance(row, dict)]

            scores, indicators = _collect_wba_scores(benchmark_rows, indicator_rows)
            result["scores"].update(scores)
            result["indicators"].update(indicators)

            result["raw"] = {
                "company": company,
                "benchmarks_meta": benchmark_payload.get("meta", {}) if isinstance(benchmark_payload, dict) else {},
                "indicators_meta": indicator_payload.get("meta", {}) if isinstance(indicator_payload, dict) else {},
                "benchmarks_rows": benchmark_rows,
                "indicator_rows": indicator_rows,
            }

            result["found"] = bool(result["scores"] or result["indicators"])
            logger.info(
                "WBA: fetched data for '%s' — %d pillar scores, %d indicators",
                result["company_name"],
                len(result["scores"]),
                len(result["indicators"]),
            )

    except httpx.TimeoutException:
        result["error"] = f"WBA API timed out for '{company_name}'"
        logger.warning("WBA: request timed out for '%s'", company_name)
    except httpx.HTTPStatusError as exc:
        detail = _extract_http_detail(exc.response)
        status = exc.response.status_code
        if status == 401:
            result["error"] = "WBA API unauthorized (check WBA_API_KEY bearer token)"
        elif detail:
            result["error"] = f"WBA API HTTP {status}: {detail}"
        else:
            result["error"] = f"WBA API HTTP {status}"
        logger.warning("WBA: HTTP %d for '%s'", status, company_name)
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
    api_key: str | None = None,
    bearer_token: str | None = None,
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

    api_key = (
        api_key
        or os.environ.get("RESOURCE_WATCH_API_KEY")
        or os.environ.get("WRI_AQUEDUCT_API_KEY")
        or ""
    ).strip()
    bearer_token = (
        bearer_token
        or os.environ.get("RESOURCE_WATCH_TOKEN")
        or os.environ.get("WRI_AQUEDUCT_TOKEN")
        or ""
    ).strip()

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

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

    if not api_key:
        result["error"] = (
            "Resource Watch API key not set "
            "(RESOURCE_WATCH_API_KEY or WRI_AQUEDUCT_API_KEY)"
        )
        logger.warning("WRI: API key not available - skipping Aqueduct lookup")
        return result

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
            if isinstance(wba.get("hq_coordinates"), dict):
                scores["_wba_hq_coordinates"] = wba["hq_coordinates"]
        else:
            logger.warning(
                "WBA lookup failed for '%s': %s",
                company_name, wba.get("error"),
            )

    # --- WRI: fill water risk ---
    # Auto-derive HQ coordinates from WBA company data when caller doesn't provide them.
    if (lat is None or lon is None) and isinstance(scores.get("_wba_hq_coordinates"), dict):
        candidate = scores.get("_wba_hq_coordinates", {})
        lat = candidate.get("lat", lat)
        lon = candidate.get("lon", lon)

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
