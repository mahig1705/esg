
import sys
import os

path = r'c:\Users\Admin\Downloads\Projects\ESG\agents\risk_scorer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = """        if not sg_adequacy.get("governance", {}).get("is_adequate", False):
            # Pull governance score toward neutral when evidence is not decision-grade.
            governance_score = round((governance_score * 0.40) + (50.0 * 0.60), 1)
            print("   ⚠️ Governance pillar confidence-limited due to insufficient free-source evidence")"""

replacement = """        if not sg_adequacy.get("governance", {}).get("is_adequate", False):
            # Pull governance score toward neutral when evidence is not decision-grade.
            governance_score = round((governance_score * 0.40) + (50.0 * 0.60), 1)
            print("   ⚠️ Governance pillar confidence-limited due to insufficient free-source evidence")

        # SEC Metrics Integration
        external = all_analyses.get("external_benchmarks", {})
        sec_metrics = external.get("sec_metrics", {}) if isinstance(external, dict) else {}
        if sec_metrics:
            print(f"   🏛️ SEC Governance Metrics detected:")
            if sec_metrics.get("board_diversity_pct") is not None:
                div = sec_metrics["board_diversity_pct"]
                print(f"      - Board Diversity: {div}%")
                if div > 30: governance_score = min(100, governance_score + 5)
            if sec_metrics.get("executive_pay_ratio") is not None:
                ratio = sec_metrics["executive_pay_ratio"]
                print(f"      - CEO Pay Ratio: {ratio}:1")
                if ratio > 300: governance_score = max(0, governance_score - 5)
            if sec_metrics.get("executive_comp_esg_links"):
                print(f"      - Executive Compensation linked to ESG: YES")
                governance_score = min(100, governance_score + 3)
            
            if sec_metrics.get("conflict_minerals_human_rights"):
                print(f"      - Conflict Minerals Human Rights Controls: YES")
                social_score = min(100, social_score + 4)"""

if target in content:
    new_content = content.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully updated risk_scorer.py")
else:
    # Try with CRLF if standard failed
    target_crlf = target.replace('\\n', '\\r\\n')
    if target_crlf in content:
        new_content = content.replace(target_crlf, replacement.replace('\\n', '\\r\\n'))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Successfully updated risk_scorer.py (CRLF)")
    else:
        print("Target not found in risk_scorer.py")
        # Print a slice of content to see what's there
        idx = content.find('governance_score = round')
        if idx != -1:
            print("Found partial match at index", idx)
            print("Context:", repr(content[idx-100:idx+200]))
