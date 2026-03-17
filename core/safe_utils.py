"""
Safe utility functions for ESG report generation.
Prevents NoneType crashes through defensive dict access and type coercion.
"""

from typing import Any, Optional
from urllib.parse import urlparse


def safe_get(d: Any, *keys, default=None) -> Any:
    """Safely traverse nested dicts/objects.

    >>> safe_get({"a": {"b": 3}}, "a", "b")
    3
    >>> safe_get(None, "a", "b", default=0)
    0
    >>> safe_get({"a": None}, "a", "b", default="x")
    'x'
    """
    current = d
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key, None)
        else:
            return default
    return current if current is not None else default


def safe_number(val: Any, default: float = 0.0) -> float:
    """Coerce a value to float, returning *default* on failure.

    >>> safe_number("42.5")
    42.5
    >>> safe_number(None)
    0.0
    >>> safe_number("N/A", default=-1.0)
    -1.0
    """
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Coerce a value to int, returning *default* on failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Source name extraction
# ---------------------------------------------------------------------------

DOMAIN_PUBLISHER_MAP = {
    "cnbc.com": "CNBC",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "ft.com": "Financial Times",
    "wsj.com": "Wall Street Journal",
    "nytimes.com": "New York Times",
    "economictimes.indiatimes.com": "Economic Times",
    "business-standard.com": "Business Standard",
    "livemint.com": "Livemint",
    "thehindubusinessline.com": "The Hindu BusinessLine",
    "moneycontrol.com": "Moneycontrol",
    "financialpost.com": "Financial Post",
    "cdp.net": "CDP (Carbon Disclosure Project)",
    "wri-india.org": "World Resources Institute India",
    "downtoearth.org.in": "Down To Earth",
    "sebi.gov.in": "SEBI",
    "mca.gov.in": "Ministry of Corporate Affairs",
    "rbi.org.in": "Reserve Bank of India",
    "envfor.nic.in": "Ministry of Environment",
    "msn.com": "MSN",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "theguardian.com": "The Guardian",
    "forbes.com": "Forbes",
    "fortune.com": "Fortune",
    "sciencebasedtargets.org": "SBTi",
    "unpri.org": "UN PRI",
    "globalreporting.org": "GRI",
    "sec.gov": "SEC",
    "epa.gov": "EPA",
    "europa.eu": "European Commission",
    "ec.europa.eu": "European Commission",
    "bseindia.com": "BSE India",
    "nseindia.com": "NSE India",
    "yahoo.com": "Yahoo Finance",
    "google.com": "Google News",
}

# Reliability tiers: domain -> (tier_number, tier_score)
_TIER1_DOMAINS = {
    "cdp.net", "globalreporting.org", "sebi.gov.in", "mca.gov.in",
    "envfor.nic.in", "sec.gov", "epa.gov", "europa.eu", "ec.europa.eu",
    "bseindia.com", "nseindia.com", "sciencebasedtargets.org", "rbi.org.in",
}
_TIER2_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "nytimes.com",
    "economictimes.indiatimes.com", "business-standard.com", "livemint.com",
}
_TIER3_DOMAINS = {
    "cnbc.com", "thehindubusinessline.com", "moneycontrol.com",
    "bbc.com", "bbc.co.uk", "theguardian.com", "forbes.com", "fortune.com",
}
_EXCLUDED_DOMAINS = {
    "news.google.com",
}


def parse_source_name(url: str) -> str:
    """Extract human-readable publisher name from a URL.

    Falls back to 'Web source' — never returns 'Unknown'.

    >>> parse_source_name("https://reuters.com/article/xyz")
    'Reuters'
    >>> parse_source_name("https://unknown-site.org/page")
    'Web source'
    """
    if not url:
        return "Web source"

    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return "Web source"

    hostname = hostname.lower().lstrip("www.")

    # Direct match
    if hostname in DOMAIN_PUBLISHER_MAP:
        return DOMAIN_PUBLISHER_MAP[hostname]

    # Check subdomains (e.g. "in.reuters.com" -> reuters.com)
    for domain, name in DOMAIN_PUBLISHER_MAP.items():
        if hostname.endswith("." + domain) or hostname == domain:
            return name

    return "Web source"


def get_reliability_tier(url: str) -> int:
    """Return reliability tier (1=highest, 4=lowest, 0=excluded).

    >>> get_reliability_tier("https://cdp.net/report")
    1
    >>> get_reliability_tier("https://reuters.com/article")
    2
    >>> get_reliability_tier("https://random-blog.com/post")
    4
    """
    if not url:
        return 4

    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return 4

    hostname = hostname.lower().lstrip("www.")

    # Check excluded
    for d in _EXCLUDED_DOMAINS:
        if hostname == d or hostname.endswith("." + d):
            return 0

    # Check tiers
    for d in _TIER1_DOMAINS:
        if hostname == d or hostname.endswith("." + d):
            return 1
    for d in _TIER2_DOMAINS:
        if hostname == d or hostname.endswith("." + d):
            return 2
    for d in _TIER3_DOMAINS:
        if hostname == d or hostname.endswith("." + d):
            return 3

    return 4


def get_reliability_score(tier: int) -> float:
    """Convert tier number to 0-1 reliability score."""
    return {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.3, 0: 0.0}.get(tier, 0.3)
