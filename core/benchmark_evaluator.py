"""
Benchmark evaluator for ground-truth ESG greenwashing dataset.

Produces overall and per-sector evaluation metrics for:
- Raw linguistic score
- Weighted-logistic recalibrated score
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import importlib.util
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss,
    confusion_matrix,
)

def _load_score_calibrator_module():
    module_path = Path(__file__).resolve().parent.parent / "ml_models" / "score_calibrator.py"
    spec = importlib.util.spec_from_file_location("score_calibrator_direct", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load score calibrator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_score_calibrator = _load_score_calibrator_module()
linguistic_greenwashing_score = _score_calibrator.linguistic_greenwashing_score
recalibrate_greenwashing_score = _score_calibrator.recalibrate_greenwashing_score


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    err = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        err += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return float(err)


def _threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    auc = _safe_auc(y_true, y_prob)
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "auc": auc,
        "brier": float(brier_score_loss(y_true, y_prob)) if len(y_true) else None,
        "ece": _ece(y_true, y_prob, n_bins=10),
        "threshold_metrics": _threshold_metrics(y_true, y_prob, threshold=threshold),
    }


def _small_sample_assessment(y_true: np.ndarray) -> Dict[str, Any]:
    n = int(len(y_true))
    positives = int(np.sum(y_true))
    negatives = int(n - positives)
    warnings: List[str] = []

    if n < 30:
        warnings.append("Sample size below 30; benchmark metrics should be treated as directional.")
    if positives < 10 or negatives < 10:
        warnings.append("Class counts are imbalanced at small absolute counts; interval estimates may be unstable.")
    if positives == 0 or negatives == 0:
        warnings.append("Single-class slice detected; discrimination metrics such as AUC are not informative.")

    if n < 15 or positives < 5 or negatives < 5:
        severity = "high"
    elif n < 30 or positives < 10 or negatives < 10:
        severity = "medium"
    else:
        severity = "low"

    return {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "warning_level": severity,
        "warnings": warnings,
    }


def _bootstrap_metric_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    threshold: float,
    iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any] | None:
    n = len(y_true)
    if n == 0:
        return None

    rng = np.random.default_rng(seed)
    samples: List[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        ps = y_prob[idx]
        if metric_name == "auc":
            value = _safe_auc(ys, ps)
        elif metric_name == "brier":
            value = float(brier_score_loss(ys, ps)) if len(ys) else None
        elif metric_name == "ece":
            value = _ece(ys, ps, n_bins=10)
        elif metric_name == "f1_at_threshold":
            value = _threshold_metrics(ys, ps, threshold=threshold).get("f1")
        else:
            value = None

        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        samples.append(float(value))

    if not samples:
        return None

    arr = np.array(samples, dtype=float)
    return {
        "iterations_requested": iterations,
        "iterations_used": int(len(arr)),
        "ci_method": "bootstrap_percentile_95",
        "lower": float(np.percentile(arr, 2.5)),
        "median": float(np.percentile(arr, 50)),
        "upper": float(np.percentile(arr, 97.5)),
    }


def _bootstrap_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    return {
        "auc": _bootstrap_metric_interval(y_true, y_prob, "auc", threshold, iterations=iterations, seed=seed),
        "brier": _bootstrap_metric_interval(y_true, y_prob, "brier", threshold, iterations=iterations, seed=seed + 1),
        "ece": _bootstrap_metric_interval(y_true, y_prob, "ece", threshold, iterations=iterations, seed=seed + 2),
        "f1_at_threshold": _bootstrap_metric_interval(
            y_true, y_prob, "f1_at_threshold", threshold, iterations=iterations, seed=seed + 3
        ),
    }


def _threshold_focus_summary(y_true: np.ndarray, y_prob: np.ndarray, center: float = 0.5, band: float = 0.1) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "band_center": center,
            "band_width": band,
            "cases_in_band": 0,
            "share_in_band": 0.0,
            "positive_rate_in_band": None,
        }

    low = center - band
    high = center + band
    mask = (y_prob >= low) & (y_prob <= high)
    cases = int(np.sum(mask))
    return {
        "band_center": center,
        "band_width": band,
        "cases_in_band": cases,
        "share_in_band": float(cases / len(y_true)),
        "positive_rate_in_band": float(np.mean(y_true[mask])) if cases else None,
    }


def _normalize_sector(sector: Any) -> str:
    text = str(sector or "").strip()
    if not text:
        return "Unknown"
    return text


def evaluate_ground_truth_benchmark(
    dataset_path: str = "data/ground_truth_dataset.csv",
    threshold: float = 0.5,
    bootstrap_iterations: int = 1000,
) -> Dict[str, Any]:
    df = pd.read_csv(dataset_path)
    required = {"company_name", "sector", "claim_text", "greenwashing_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df["sector"] = df["sector"].apply(_normalize_sector)
    df["label"] = df["greenwashing_label"].astype(int)

    raw_scores = []
    recal_scores = []
    for _, row in df.iterrows():
        claim = str(row["claim_text"])
        company = str(row["company_name"])
        sector = str(row["sector"])
        raw = float(linguistic_greenwashing_score(claim, company, sector))
        recal = float(recalibrate_greenwashing_score(raw, sector=sector).get("recalibrated_score", raw))
        raw_scores.append(raw / 100.0)
        recal_scores.append(recal / 100.0)

    df["raw_prob"] = np.array(raw_scores)
    df["recal_prob"] = np.array(recal_scores)

    y = df["label"].to_numpy()
    overall = {
        "raw": _compute_metrics(y, df["raw_prob"].to_numpy(), threshold=threshold),
        "recalibrated": _compute_metrics(y, df["recal_prob"].to_numpy(), threshold=threshold),
    }
    overall["raw"]["bootstrap_confidence_intervals"] = _bootstrap_intervals(
        y, df["raw_prob"].to_numpy(), threshold=threshold, iterations=bootstrap_iterations, seed=42
    )
    overall["recalibrated"]["bootstrap_confidence_intervals"] = _bootstrap_intervals(
        y, df["recal_prob"].to_numpy(), threshold=threshold, iterations=bootstrap_iterations, seed=142
    )
    overall["sample_assessment"] = _small_sample_assessment(y)
    overall["threshold_focus"] = {
        "raw": _threshold_focus_summary(y, df["raw_prob"].to_numpy(), center=threshold, band=0.1),
        "recalibrated": _threshold_focus_summary(y, df["recal_prob"].to_numpy(), center=threshold, band=0.1),
    }

    sector_rows: List[Dict[str, Any]] = []
    by_sector: Dict[str, Any] = {}
    for sector, g in df.groupby("sector"):
        ys = g["label"].to_numpy()
        raw_m = _compute_metrics(ys, g["raw_prob"].to_numpy(), threshold=threshold)
        rec_m = _compute_metrics(ys, g["recal_prob"].to_numpy(), threshold=threshold)
        raw_m["bootstrap_confidence_intervals"] = _bootstrap_intervals(
            ys, g["raw_prob"].to_numpy(), threshold=threshold, iterations=bootstrap_iterations, seed=42
        )
        rec_m["bootstrap_confidence_intervals"] = _bootstrap_intervals(
            ys, g["recal_prob"].to_numpy(), threshold=threshold, iterations=bootstrap_iterations, seed=142
        )
        sample_assessment = _small_sample_assessment(ys)
        threshold_focus = {
            "raw": _threshold_focus_summary(ys, g["raw_prob"].to_numpy(), center=threshold, band=0.1),
            "recalibrated": _threshold_focus_summary(ys, g["recal_prob"].to_numpy(), center=threshold, band=0.1),
        }
        by_sector[sector] = {
            "raw": raw_m,
            "recalibrated": rec_m,
            "sample_assessment": sample_assessment,
            "threshold_focus": threshold_focus,
        }

        sector_rows.append(
            {
                "sector": sector,
                "n": int(len(g)),
                "positive_rate": float(np.mean(ys)) if len(ys) else 0.0,
                "warning_level": sample_assessment.get("warning_level"),
                "raw_auc": raw_m.get("auc"),
                "raw_auc_ci_low": ((raw_m.get("bootstrap_confidence_intervals") or {}).get("auc") or {}).get("lower"),
                "raw_auc_ci_high": ((raw_m.get("bootstrap_confidence_intervals") or {}).get("auc") or {}).get("upper"),
                "recal_auc": rec_m.get("auc"),
                "recal_auc_ci_low": ((rec_m.get("bootstrap_confidence_intervals") or {}).get("auc") or {}).get("lower"),
                "recal_auc_ci_high": ((rec_m.get("bootstrap_confidence_intervals") or {}).get("auc") or {}).get("upper"),
                "raw_brier": raw_m.get("brier"),
                "recal_brier": rec_m.get("brier"),
                "raw_ece": raw_m.get("ece"),
                "recal_ece": rec_m.get("ece"),
                "raw_f1_at_50": raw_m.get("threshold_metrics", {}).get("f1"),
                "recal_f1_at_50": rec_m.get("threshold_metrics", {}).get("f1"),
                "raw_cases_near_threshold": threshold_focus["raw"].get("cases_in_band"),
                "recal_cases_near_threshold": threshold_focus["recalibrated"].get("cases_in_band"),
            }
        )

    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_size": int(len(df)),
        "threshold": threshold,
        "bootstrap_iterations": bootstrap_iterations,
        "overall": overall,
        "by_sector": by_sector,
        "sector_table": sector_rows,
    }
    return evaluation


def save_benchmark_artifacts(
    evaluation: Dict[str, Any],
    output_dir: str = "reports",
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "benchmark_sector_evaluation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    sector_df = pd.DataFrame(evaluation.get("sector_table", []))
    csv_path = out / "benchmark_sector_table.csv"
    sector_df.to_csv(csv_path, index=False)

    summary_path = out / "benchmark_summary.md"
    overall = evaluation.get("overall", {})
    raw = (overall.get("raw") or {})
    rec = (overall.get("recalibrated") or {})
    sample_assessment = overall.get("sample_assessment", {}) if isinstance(overall.get("sample_assessment"), dict) else {}
    threshold_focus = overall.get("threshold_focus", {}) if isinstance(overall.get("threshold_focus"), dict) else {}
    raw_ci = (raw.get("bootstrap_confidence_intervals") or {})
    rec_ci = (rec.get("bootstrap_confidence_intervals") or {})
    lines = [
        "# ESG Benchmark Summary",
        "",
        f"- Generated: {evaluation.get('generated_at')}",
        f"- Dataset size: {evaluation.get('dataset_size')}",
        f"- Threshold: {evaluation.get('threshold')}",
        f"- Bootstrap iterations: {evaluation.get('bootstrap_iterations')}",
        "",
        "## Overall",
        f"- Raw AUC: {raw.get('auc')}",
        f"- Raw AUC 95% CI: {((raw_ci.get('auc') or {}).get('lower'))} to {((raw_ci.get('auc') or {}).get('upper'))}",
        f"- Recalibrated AUC: {rec.get('auc')}",
        f"- Recalibrated AUC 95% CI: {((rec_ci.get('auc') or {}).get('lower'))} to {((rec_ci.get('auc') or {}).get('upper'))}",
        f"- Raw Brier: {raw.get('brier')}",
        f"- Recalibrated Brier: {rec.get('brier')}",
        f"- Raw ECE: {raw.get('ece')}",
        f"- Recalibrated ECE: {rec.get('ece')}",
        f"- Raw F1@50: {(raw.get('threshold_metrics') or {}).get('f1')}",
        f"- Recalibrated F1@50: {(rec.get('threshold_metrics') or {}).get('f1')}",
        "",
        "## Small-Sample Assessment",
        f"- Warning level: {sample_assessment.get('warning_level')}",
        f"- Positives: {sample_assessment.get('positives')}",
        f"- Negatives: {sample_assessment.get('negatives')}",
    ]
    for warning in sample_assessment.get("warnings", []) if isinstance(sample_assessment.get("warnings"), list) else []:
        lines.append(f"- Warning: {warning}")

    raw_threshold_focus = threshold_focus.get("raw", {}) if isinstance(threshold_focus.get("raw"), dict) else {}
    rec_threshold_focus = threshold_focus.get("recalibrated", {}) if isinstance(threshold_focus.get("recalibrated"), dict) else {}
    lines.extend(
        [
            "",
            "## Threshold Focus",
            f"- Raw cases near threshold (+/-0.10): {raw_threshold_focus.get('cases_in_band')}",
            f"- Recalibrated cases near threshold (+/-0.10): {rec_threshold_focus.get('cases_in_band')}",
            f"- Raw share near threshold: {raw_threshold_focus.get('share_in_band')}",
            f"- Recalibrated share near threshold: {rec_threshold_focus.get('share_in_band')}",
            "",
            "## Sector Table",
            "",
            "| Sector | n | Warn | PosRate | Raw AUC | Raw AUC 95% CI | Recal AUC | Recal AUC 95% CI | Raw Brier | Recal Brier | Raw ECE | Recal ECE | Raw F1@50 | Recal F1@50 | Near 50 (Raw) | Near 50 (Recal) |",
            "|---|---:|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evaluation.get("sector_table", []):
        lines.append(
            "| {sector} | {n} | {warning_level} | {positive_rate:.3f} | {raw_auc} | {raw_auc_ci_low} to {raw_auc_ci_high} | {recal_auc} | {recal_auc_ci_low} to {recal_auc_ci_high} | {raw_brier:.3f} | {recal_brier:.3f} | {raw_ece:.3f} | {recal_ece:.3f} | {raw_f1_at_50:.3f} | {recal_f1_at_50:.3f} | {raw_cases_near_threshold} | {recal_cases_near_threshold} |".format(
                sector=row.get("sector", "Unknown"),
                n=row.get("n", 0),
                warning_level=str(row.get("warning_level", "") or ""),
                positive_rate=float(row.get("positive_rate", 0.0) or 0.0),
                raw_auc=row.get("raw_auc"),
                raw_auc_ci_low=row.get("raw_auc_ci_low"),
                raw_auc_ci_high=row.get("raw_auc_ci_high"),
                recal_auc=row.get("recal_auc"),
                recal_auc_ci_low=row.get("recal_auc_ci_low"),
                recal_auc_ci_high=row.get("recal_auc_ci_high"),
                raw_brier=float(row.get("raw_brier", 0.0) or 0.0),
                recal_brier=float(row.get("recal_brier", 0.0) or 0.0),
                raw_ece=float(row.get("raw_ece", 0.0) or 0.0),
                recal_ece=float(row.get("recal_ece", 0.0) or 0.0),
                raw_f1_at_50=float(row.get("raw_f1_at_50", 0.0) or 0.0),
                recal_f1_at_50=float(row.get("recal_f1_at_50", 0.0) or 0.0),
                raw_cases_near_threshold=int(row.get("raw_cases_near_threshold", 0) or 0),
                recal_cases_near_threshold=int(row.get("recal_cases_near_threshold", 0) or 0),
            )
        )
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(summary_path),
    }
