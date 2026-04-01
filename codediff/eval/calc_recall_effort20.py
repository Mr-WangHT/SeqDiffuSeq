#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def to_int_label(v: str | int | float) -> int:
    if isinstance(v, int):
        return 1 if v != 0 else 0
    if isinstance(v, float):
        return 1 if v >= 0.5 else 0
    s = str(v).strip().lower()
    return 1 if s in {"1", "true", "t", "yes"} else 0


def calc_file_metrics(sorted_rows: List[Tuple[float, int]]) -> Optional[Tuple[float, float]]:
    """
    Return (Recall@20%LOC, Effort@20%Recall) for one file.
    sorted_rows must be sorted by score descending and contain (score, gt).
    """
    n = len(sorted_rows)
    if n == 0:
        return None

    total_true = sum(gt for _, gt in sorted_rows)
    if total_true == 0:
        return None

    # Recall@20%LOC
    k = max(1, int(round(n * 0.2)))
    recall20 = sum(gt for _, gt in sorted_rows[:k]) / total_true

    # Effort@20%Recall (aligned with DeepLineDP R logic)
    cum_true = 0
    count_prefix = 0
    for _, gt in sorted_rows:
        cum_true += gt
        recall = round(cum_true / total_true, 2)
        if recall <= 0.2:
            count_prefix += 1
    effort20 = count_prefix / n

    return recall20, effort20


def eval_release(csv_path: Path) -> Dict[str, float]:
    by_file: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            filename = row["filename"]
            score = float(row["prediction-prob"])
            gt = to_int_label(row["line-level-ground-truth"])
            by_file[filename].append((score, gt))

    recalls: List[float] = []
    efforts: List[float] = []

    for _, rows in by_file.items():
        rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
        m = calc_file_metrics(rows_sorted)
        if m is None:
            continue
        recalls.append(m[0])
        efforts.append(m[1])

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_effort = sum(efforts) / len(efforts) if efforts else 0.0

    return {
        "release": csv_path.stem,
        "Recall@20%LOC": mean_recall,
        "Effort@20%Recall": mean_effort,
        "num_files_with_bug": float(len(recalls)),
    }


def save_outputs(results: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "recall_effort20_within_release.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["release", "Recall@20%LOC", "Effort@20%Recall", "num_files_with_bug"])
        for r in results:
            writer.writerow(
                [
                    r["release"],
                    f"{r['Recall@20%LOC']:.6f}",
                    f"{r['Effort@20%Recall']:.6f}",
                    int(r["num_files_with_bug"]),
                ]
            )

    n = len(results)
    summary = {
        "num_releases": n,
        "mean_Recall@20%LOC": sum(r["Recall@20%LOC"] for r in results) / n if n else 0.0,
        "mean_Effort@20%Recall": sum(r["Effort@20%Recall"] for r in results) / n if n else 0.0,
        "total_files_with_bug": int(sum(r["num_files_with_bug"] for r in results)),
    }

    json_path = out_dir / "recall_effort20_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Recall@20%LOC and Effort@20%Recall from within-release prediction CSVs.")
    parser.add_argument(
        "--prediction-dir",
        default="codediff/output/prediction/CodeDiff/within-release",
        help="Directory containing per-release prediction CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="CodeDiff/eval_result",
        help="Directory to save metric files.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.prediction_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    csv_files = sorted(pred_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {pred_dir}")

    results = [eval_release(fp) for fp in csv_files]

    for r in results:
        print(
            f"{r['release']}: Recall@20%LOC={r['Recall@20%LOC']:.6f}, "
            f"Effort@20%Recall={r['Effort@20%Recall']:.6f}, "
            f"num_files_with_bug={int(r['num_files_with_bug'])}"
        )

    save_outputs(results, Path(args.out_dir))
    print(f"Saved detailed metrics to: {Path(args.out_dir) / 'recall_effort20_within_release.csv'}")
    print(f"Saved summary metrics to: {Path(args.out_dir) / 'recall_effort20_summary.json'}")


if __name__ == "__main__":
    main()
