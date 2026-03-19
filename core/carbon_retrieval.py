import asyncio
from core.carbon_validator import CarbonDataValidator

async def fetch_and_parse_carbon(url: str, source_name: str) -> dict:
    """Mock network request parser to simulate CDP / IR scraping."""
    # In a full implementation, this uses a scraping tool or API.
    # We mock failure by returning None, triggering the fallback chain fully.
    return None

async def fetch_regulatory_carbon(company: str, ticker: str, country: str) -> dict:
    """Mock regulatory filing search."""
    return None

def build_ir_urls(company: str, ticker: str, country: str) -> list[str]:
    """
    Build a prioritised list of IR/sustainability URLs to try.
    Extend this list as you add more companies.
    """
    company_slug = company.lower().replace(" ", "")
    urls = [
        f"https://www.{company_slug}.com/investors",
        f"https://www.{company_slug}.com/sustainability",
        f"https://www.{company_slug}.com/esg",
        f"https://www.{company_slug}.com/annual-report",
    ]
    # Country-specific regulatory filing URLs
    if country in ("GB", "UK"):
        urls.append(
            f"https://find-and-update.company-information.service.gov.uk/search?q={company}"
        )
    if country == "US":
        urls.append(
            f"https://www.sec.gov/cgi-bin/browse-edgar?company={company}&action=getcompany"
        )
    if country == "IN":
        urls.append(
            f"https://www.bseindia.com/corporates/ann.html?scrip={ticker}"
        )
    return urls

async def fetch_carbon_with_fallback(
    company: str,
    ticker: str, 
    industry: str,
    country: str,
    report_year: int
) -> dict:

    validator = CarbonDataValidator()
    sources_tried = []

    # SOURCE 1: CDP
    cdp_url = f"https://www.cdp.net/en/responses?queries%5Bname%5D={company}"
    cdp_data = await fetch_and_parse_carbon(cdp_url, source_name="CDP")
    sources_tried.append("CDP")
    if cdp_data:
        validated = validator.validate(cdp_data, company, industry, report_year)
        if validated["validation"]["passed"]:
            validated["source_chain"] = sources_tried
            return validated

    # SOURCE 2: Company IR page
    ir_urls = build_ir_urls(company, ticker, country)
    for ir_url in ir_urls:
        ir_data = await fetch_and_parse_carbon(ir_url, source_name="Company IR")
        # In the context of logging, we just log the domain or generic 'Company IR' to prevent explosive lists
        domain = ir_url.split('/')[2] if '//' in ir_url else ir_url
        sources_tried.append(f"Company IR ({domain})")
        if ir_data:
            validated = validator.validate(ir_data, company, industry, report_year)
            if validated["validation"]["passed"]:
                validated["source_chain"] = sources_tried
                return validated
        # Limiting loop slightly for mock
        break

    # SOURCE 3: Regulatory filings
    reg_data = await fetch_regulatory_carbon(company, ticker, country)
    sources_tried.append("Regulatory filing")
    if reg_data:
        validated = validator.validate(reg_data, company, industry, report_year)
        if validated["validation"]["passed"]:
            validated["source_chain"] = sources_tried
            return validated

    # SOURCE 4: Industry estimate fallback — always returns something
    floors = validator._get_floors(industry)
    fallback = validator._build_fallback(company, industry, floors)
    
    return {
        "scope1": None,
        "scope2": None,
        "scope3": None,
        "data_year": None,
        "data_quality": 0,
        "source": "Industry estimate (all primary sources failed)",
        "source_chain": sources_tried,
        "validation": {
            "passed": False,
            "fallback_triggered": True,
            "fallback_estimate": fallback,
            "rejection_reasons": ["No valid primary source found after exhausting fallback chain"],
            "validated_quality_score": 0
        }
    }
