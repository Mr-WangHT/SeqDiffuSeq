#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def to_int_label(v: str | int | float) -> int:
    if isinstance(v, int):
        return 1 if v != 0 else 0
    if isinstance(v, float):
        return 1 if v >= 0.5 else 0
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes"}:
        return 1
    return 0


def to_float(v: str | int | float) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).strip())


def auc_roc(y_true: List[int], y_score: List[float]) -> float:
    # Rank-based AUC without external dependencies.
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5

    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        num_pos_in_tie = sum(label for _, label in pairs[i : j + 1])
        rank_sum_pos += avg_rank * num_pos_in_tie
        rank += (j - i + 1)
        i = j + 1

    return (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)


def confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def balanced_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    return 0.5 * (tpr + tnr)


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return safe_div(num, den)


@dataclass
class EvalResult:
    release: str
    auc: float
    mcc: float
    balanced_accuracy: float
    line_precision: float
    line_recall: float
    line_f1: float
    line_acc: float
    n_files: int
    n_lines: int


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


def build_file_level_from_line_level(rows: Iterable[Dict[str, str]]) -> List[Dict[str, float]]:
    # R script uses file-level GT/prob/pred directly.
    # CodeDiff within-release output is line-level, so we aggregate by filename:
    # - file ground truth: any defective line -> 1
    # - file probability: max line probability
    # - file prediction: threshold on file probability (>=0.5)
    by_file: Dict[str, Dict[str, float]] = {}
    for r in rows:
        filename = r["filename"]
        gt = to_int_label(r["line-level-ground-truth"])
        prob = to_float(r["prediction-prob"])
        entry = by_file.get(filename)
        if entry is None:
            by_file[filename] = {"gt": float(gt), "prob": prob}
        else:
            entry["gt"] = float(max(int(entry["gt"]), gt))
            entry["prob"] = max(float(entry["prob"]), prob)

    file_rows: List[Dict[str, float]] = []
    for filename, v in by_file.items():
        p = float(v["prob"])
        file_rows.append(
            {
                "filename": filename,
                "file_gt": float(v["gt"]),
                "file_prob": p,
                "file_pred": 1.0 if p >= 0.5 else 0.0,
            }
        )
    return file_rows


def eval_release(csv_path: Path) -> EvalResult:
    rows = read_csv_rows(csv_path)

    # Line-level metrics
    y_true_line = [to_int_label(r["line-level-ground-truth"]) for r in rows]
    y_pred_line = [to_int_label(r["prediction-label"]) for r in rows]
    tp_l, fp_l, fn_l, tn_l = confusion_counts(y_true_line, y_pred_line)
    p_l = safe_div(tp_l, tp_l + fp_l)
    r_l = safe_div(tp_l, tp_l + fn_l)
    f1_l = safe_div(2 * p_l * r_l, p_l + r_l)
    acc_l = safe_div(tp_l + tn_l, len(y_true_line))

    # File-level metrics (mapped from line-level output)
    file_rows = build_file_level_from_line_level(rows)
    y_true_file = [int(fr["file_gt"]) for fr in file_rows]
    y_prob_file = [float(fr["file_prob"]) for fr in file_rows]
    y_pred_file = [int(fr["file_pred"]) for fr in file_rows]

    tp, fp, fn, tn = confusion_counts(y_true_file, y_pred_file)
    result = EvalResult(
        release=csv_path.stem,
        auc=auc_roc(y_true_file, y_prob_file),
        mcc=mcc(tp, fp, fn, tn),
        balanced_accuracy=balanced_accuracy(tp, fp, fn, tn),
        line_precision=p_l,
        line_recall=r_l,
        line_f1=f1_l,
        line_acc=acc_l,
        n_files=len(file_rows),
        n_lines=len(rows),
    )
    return result


def save_results(results: List[EvalResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "within_release_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "release",
                "auc",
                "mcc",
                "balanced_accuracy",
                "line_precision",
                "line_recall",
                "line_f1",
                "line_acc",
                "n_files",
                "n_lines",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.release,
                    f"{r.auc:.6f}",
                    f"{r.mcc:.6f}",
                    f"{r.balanced_accuracy:.6f}",
                    f"{r.line_precision:.6f}",
                    f"{r.line_recall:.6f}",
                    f"{r.line_f1:.6f}",
                    f"{r.line_acc:.6f}",
                    r.n_files,
                    r.n_lines,
                ]
            )

    mean_dict = {
        "num_releases": len(results),
        "mean_auc": safe_div(sum(r.auc for r in results), len(results)),
        "mean_mcc": safe_div(sum(r.mcc for r in results), len(results)),
        "mean_balanced_accuracy": safe_div(sum(r.balanced_accuracy for r in results), len(results)),
        "mean_line_precision": safe_div(sum(r.line_precision for r in results), len(results)),
        "mean_line_recall": safe_div(sum(r.line_recall for r in results), len(results)),
        "mean_line_f1": safe_div(sum(r.line_f1 for r in results), len(results)),
        "mean_line_acc": safe_div(sum(r.line_acc for r in results), len(results)),
    }

    json_path = out_dir / "within_release_metrics_summary.json"
    json_path.write_text(json.dumps(mean_dict, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Python translation (core evaluation part) of DeepLineDP get_evaluation_result.R, "
            "adapted for CodeDiff within-release prediction files."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        default="codediff/output/prediction/CodeDiff/within-release",
        help="Directory containing per-release prediction CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="CodeDiff/eval_result",
        help="Directory to save computed metrics.",
    )
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    csv_files = sorted(prediction_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {prediction_dir}")

    results: List[EvalResult] = []
    for csv_file in csv_files:
        r = eval_release(csv_file)
        results.append(r)
        print(
            f"{r.release}: "
            f"AUC={r.auc:.4f}, MCC={r.mcc:.4f}, BalAcc={r.balanced_accuracy:.4f}, "
            f"LineF1={r.line_f1:.4f}, LineAcc={r.line_acc:.4f}"
        )

    save_results(results, Path(args.out_dir))
    print(f"Saved detailed metrics to: {Path(args.out_dir) / 'within_release_metrics.csv'}")
    print(f"Saved summary metrics to: {Path(args.out_dir) / 'within_release_metrics_summary.json'}")


if __name__ == "__main__":
    main()
