
import sys
import os

path = r'c:\Users\Admin\Downloads\Projects\ESG\agents\risk_scorer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = """            if g >= 70:
                positives.append("Strong governance signal (board/ethics/disclosure indicators)")
            if s >= 70:
                positives.append("Strong social signal (labor, safety, DEI, community indicators)")
            if e >= 70:
                positives.append("Strong environmental signal (climate, renewables, emissions indicators)")"""

replacement = """            if g >= 70:
                positives.append("Strong governance signal (board/ethics/disclosure indicators)")
            
            # SEC Metrics in Reasons
            external = analyses.get("external_benchmarks", {})
            sec = external.get("sec_metrics", {}) if isinstance(external, dict) else {}
            if sec:
                if sec.get("board_diversity_pct"):
                    positives.append(f"Documented board diversity ({sec['board_diversity_pct']}%) via SEC DEF 14A")
                if sec.get("executive_comp_esg_links"):
                    positives.append("Executive compensation formally linked to ESG targets")

            if s >= 70:
                positives.append("Strong social signal (labor, safety, DEI, community indicators)")
            if e >= 70:
                positives.append("Strong environmental signal (climate, renewables, emissions indicators)")"""

if target in content:
    new_content = content.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully updated risk_scorer.py (Reasons)")
else:
    print("Target not found in risk_scorer.py")
