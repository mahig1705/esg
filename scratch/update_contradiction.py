
import sys
import os

path = r'c:\Users\Admin\Downloads\Projects\ESG\agents\contradiction_analyzer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = """        try:
            llm_found = self._run_llm_contradiction_scan(claim=claim, evidence=evidence, temperature=0)"""

replacement = """        # Combine standard and historical evidence for LLM scan to detect temporal violations
        combined_evidence = list(evidence or []) + historical_evidence
        
        try:
            llm_found = self._run_llm_contradiction_scan(claim=claim, evidence=combined_evidence, temperature=0)"""

if target in content:
    new_content = content.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully updated contradiction_analyzer.py")
else:
    print("Target not found in contradiction_analyzer.py")
