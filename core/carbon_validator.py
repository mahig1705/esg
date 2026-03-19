import json
import os

class CarbonDataValidator:
    """
    Validates extracted carbon emissions data against industry-specific
    plausibility floors. Rejects bad data and triggers fallback retrieval.
    """

    MAX_ACCEPTABLE_DATA_AGE_YEARS = 3  # reject data older than this

    def validate(self, carbon_data: dict, company: str, 
                 industry: str, report_year: int) -> dict:
        """
        Main entry point. Returns a validated carbon_data dict with:
        - rejection flags
        - fallback estimates if rejected
        - quality score
        - data_age_years
        
        Input carbon_data format:
        {
          "scope1": float | None,
          "scope2": float | None,  
          "scope3": float | None,
          "data_year": int | None,
          "data_quality": int,          # 0-100
          "source": str
        }
        """
        
        result = carbon_data.copy()
        result["validation"] = {
            "passed": False,
            "rejection_reasons": [],
            "warnings": [],
            "data_age_years": None,
            "floor_used": None,
            "fallback_triggered": False,
            "fallback_estimate": None,
            "validated_quality_score": 0
        }
        
        floors = self._get_floors(industry)
        result["validation"]["floor_used"] = industry if industry in self._load_all_floors() else "Default"
        
        # CHECK 1: Data age
        if carbon_data.get("data_year"):
            age = report_year - carbon_data["data_year"]
            result["validation"]["data_age_years"] = age
            if age > self.MAX_ACCEPTABLE_DATA_AGE_YEARS:
                result["validation"]["rejection_reasons"].append(
                    f"Data is {age} years old (from {carbon_data['data_year']}). "
                    f"Maximum acceptable age is {self.MAX_ACCEPTABLE_DATA_AGE_YEARS} years."
                )
        else:
            result["validation"]["warnings"].append(
                "No data_year provided — cannot assess data age."
            )

        # CHECK 2: Scope 1 floor check (most critical)
        scope1 = carbon_data.get("scope1")
        if scope1 is None or scope1 == 0:
            result["validation"]["rejection_reasons"].append(
                f"Scope 1 is missing or zero. Cannot evaluate net-zero claim "
                f"without direct operational emissions."
            )
        elif scope1 < floors["scope1_min"]:
            result["validation"]["rejection_reasons"].append(
                f"Scope 1 = {scope1:,.0f} tCO2e is below the minimum plausible "
                f"floor of {floors['scope1_min']:,.0f} tCO2e for {industry} companies. "
                f"Likely a data extraction error or unit mismatch (check if value "
                f"is in MtCO2e instead of tCO2e)."
            )

        # CHECK 3: Unit mismatch detection
        # Common error: value is in MtCO2e but stored as tCO2e (off by 1M)
        # OR value is in ktCO2e (off by 1000)
        if scope1 and scope1 < floors["scope1_min"]:
            if scope1 * 1_000_000 >= floors["scope1_min"]:
                result["validation"]["warnings"].append(
                    f"POSSIBLE UNIT ERROR: {scope1} looks like it may be in "
                    f"MtCO2e. Multiply by 1,000,000 to get tCO2e. "
                    f"Corrected value would be {scope1 * 1_000_000:,.0f} tCO2e."
                )
            elif scope1 * 1_000 >= floors["scope1_min"]:
                result["validation"]["warnings"].append(
                    f"POSSIBLE UNIT ERROR: {scope1} looks like it may be in "
                    f"ktCO2e. Multiply by 1,000 to get tCO2e. "
                    f"Corrected value would be {scope1 * 1_000:,.0f} tCO2e."
                )

        # CHECK 4: Scope 3 completeness (for net-zero claims)
        scope3 = carbon_data.get("scope3")
        if scope3 and scope3 < (scope1 or 0):
            result["validation"]["warnings"].append(
                f"Scope 3 ({scope3:,.0f}) is LESS than Scope 1 ({scope1:,.0f}). "
                f"For most industries, Scope 3 is 5-20x larger than Scope 1. "
                f"Possible incomplete Scope 3 calculation."
            )

        # CHECK 5: Data quality score threshold
        if carbon_data.get("data_quality", 0) < 40:
            result["validation"]["rejection_reasons"].append(
                f"Data quality score is {carbon_data.get('data_quality', 0)}/100, "
                f"below the 40/100 minimum threshold for inclusion."
            )

        # DECISION: pass or reject
        if not result["validation"]["rejection_reasons"]:
            result["validation"]["passed"] = True
            result["validation"]["validated_quality_score"] = carbon_data.get(
                "data_quality", 50
            )
        else:
            result["validation"]["passed"] = False
            result["validation"]["fallback_triggered"] = True
            result["validation"]["fallback_estimate"] = self._build_fallback(
                company, industry, floors
            )
            result["validation"]["validated_quality_score"] = 0

        return result

    def _load_all_floors(self) -> dict:
        floors_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "emissions_floors.json"
        )
        with open(floors_path) as f:
            data = json.load(f)
        return data["emissions_floors"]

    def _get_floors(self, industry: str) -> dict:
        floors = self._load_all_floors()
        return floors.get(industry, floors["Default"])

    def _build_fallback(self, company: str, industry: str, floors: dict) -> dict:
        """
        Returns a clearly labelled estimated range when real data is rejected.
        This is shown in the report instead of the bad extracted value.
        """
        return {
            "type": "industry_estimate",
            "scope1_estimated_low": floors["scope1_typical_low"],
            "scope1_estimated_high": floors["scope1_typical_high"],
            "display_string": (
                f"Scope 1 data could not be verified from available sources. "
                f"Based on industry benchmarks for {industry} companies of "
                f"similar scale, estimated Scope 1 emissions range: "
                f"{floors['scope1_typical_low'] / 1_000_000:.0f}M – "
                f"{floors['scope1_typical_high'] / 1_000_000:.0f}M tCO2e/year. "
                f"Upload the company's latest annual report or CDP submission "
                f"to obtain verified figures."
            ),
            "confidence": "LOW",
            "source": "Industry benchmark estimate — not company-verified"
        }
