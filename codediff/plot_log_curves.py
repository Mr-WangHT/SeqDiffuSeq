#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(
    r"epoch\s*=\s*(?P<epoch>\d+)\s+"
    r"train_loss\s*=\s*(?P<train_loss>[0-9eE+\-.]+)\s+"
    r"train_mse\s*=\s*(?P<train_mse>[0-9eE+\-.]+)\s+"
    r"train_cls\s*=\s*(?P<train_cls>[0-9eE+\-.]+)\s+"
    r"valid_loss\s*=\s*(?P<valid_loss>[0-9eE+\-.]+)\s+"
    r"valid_mse\s*=\s*(?P<valid_mse>[0-9eE+\-.]+)\s+"
    r"valid_cls\s*=\s*(?P<valid_cls>[0-9eE+\-.]+)\s+"
    r"valid_p\s*=\s*(?P<valid_p>[0-9eE+\-.]+)\s+"
    r"valid_r\s*=\s*(?P<valid_r>[0-9eE+\-.]+)\s+"
    r"valid_f1\s*=\s*(?P<valid_f1>[0-9eE+\-.]+)\s+"
    r"valid_acc\s*=\s*(?P<valid_acc>[0-9eE+\-.]+)",
    re.MULTILINE,
)


def parse_epoch_metrics(log_text: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for m in EPOCH_PATTERN.finditer(log_text):
        row = {
            "epoch": int(m.group("epoch")),
            "train_loss": float(m.group("train_loss")),
            "train_mse": float(m.group("train_mse")),
            "train_cls": float(m.group("train_cls")),
            "valid_loss": float(m.group("valid_loss")),
            "valid_mse": float(m.group("valid_mse")),
            "valid_cls": float(m.group("valid_cls")),
            "valid_p": float(m.group("valid_p")),
            "valid_r": float(m.group("valid_r")),
            "valid_f1": float(m.group("valid_f1")),
            "valid_acc": float(m.group("valid_acc")),
        }
        rows.append(row)

    # Keep last record for each epoch in case duplicated logs are appended.
    dedup: Dict[int, Dict[str, float]] = {}
    for row in rows:
        dedup[int(row["epoch"])] = row

    return [dedup[e] for e in sorted(dedup.keys())]


def save_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_mse",
        "train_cls",
        "valid_loss",
        "valid_mse",
        "valid_cls",
        "valid_p",
        "valid_r",
        "valid_f1",
        "valid_acc",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_loss_curves(rows: List[Dict[str, float]], out_path: Path) -> None:
    epochs = [int(r["epoch"]) for r in rows]
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, [r["train_loss"] for r in rows], marker="o", label="train_loss")
    plt.plot(epochs, [r["valid_loss"] for r in rows], marker="o", label="valid_loss")
    plt.plot(epochs, [r["train_mse"] for r in rows], marker=".", linestyle="--", label="train_mse")
    plt.plot(epochs, [r["train_cls"] for r in rows], marker=".", linestyle="--", label="train_cls")
    plt.plot(epochs, [r["valid_mse"] for r in rows], marker=".", linestyle=":", label="valid_mse")
    plt.plot(epochs, [r["valid_cls"] for r in rows], marker=".", linestyle=":", label="valid_cls")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_valid_metrics(rows: List[Dict[str, float]], out_path: Path) -> None:
    epochs = [int(r["epoch"]) for r in rows]
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, [r["valid_p"] for r in rows], marker="o", label="valid_precision")
    plt.plot(epochs, [r["valid_r"] for r in rows], marker="o", label="valid_recall")
    plt.plot(epochs, [r["valid_f1"] for r in rows], marker="o", label="valid_f1")
    plt.plot(epochs, [r["valid_acc"] for r in rows], marker="o", label="valid_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss and validation metric curves from CodeDiff training log.")
    parser.add_argument("--log-file", required=True, help="Path to training log file.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for figures and parsed CSV. Default: <log_dir>/figures",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (log_path.parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = parse_epoch_metrics(text)
    if not rows:
        raise ValueError("No epoch summary lines found in log. Expected lines like: epoch=... train_loss=... valid_acc=...")

    parsed_csv = out_dir / "epoch_metrics_parsed.csv"
    loss_png = out_dir / "loss_curves.png"
    valid_png = out_dir / "valid_metrics_curves.png"

    save_csv(rows, parsed_csv)
    plot_loss_curves(rows, loss_png)
    plot_valid_metrics(rows, valid_png)

    print(f"Parsed epochs: {len(rows)}")
    print(f"Saved CSV: {parsed_csv}")
    print(f"Saved loss figure: {loss_png}")
    print(f"Saved valid figure: {valid_png}")


if __name__ == "__main__":
    main()
