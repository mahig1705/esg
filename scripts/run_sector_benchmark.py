"""
Run sector-wise benchmark evaluation on ground-truth dataset.

Usage:
    venv\\Scripts\\python.exe scripts\\run_sector_benchmark.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark_evaluator import evaluate_ground_truth_benchmark, save_benchmark_artifacts  # noqa: E402


def main() -> int:
    evaluation = evaluate_ground_truth_benchmark(
        dataset_path="data/ground_truth_dataset.csv",
        threshold=0.5,
    )
    paths = save_benchmark_artifacts(evaluation, output_dir="reports")
    print("Sector benchmark completed.")
    print(f"Dataset size: {evaluation.get('dataset_size')}")
    print(f"Artifacts: {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

