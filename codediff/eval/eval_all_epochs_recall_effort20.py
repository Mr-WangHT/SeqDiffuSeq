#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codediff.eval.calc_recall_effort20 import eval_release
from codediff.model_utils import create_model_and_diffusion
from codediff.main import generate_release_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all epoch checkpoints with Recall@20 and Effort@20, then plot curves."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Checkpoint directory, e.g. codediff/output/model/CodeDiff/activemq/<exp_name>",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Default: CodeDiff/eval_result/<exp_name>_all_epochs_eval",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Override device. Default uses run_config device.",
    )
    parser.add_argument(
        "--skip-prediction-csv",
        action="store_true",
        help="If set, do not keep per-release prediction CSVs (faster disk cleanup).",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Start evaluating from this epoch (inclusive).",
    )
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=-1,
        help="End evaluating at this epoch (inclusive). -1 means last checkpoint.",
    )
    parser.add_argument(
        "--reuse-existing-predictions",
        action="store_true",
        help="Reuse existing prediction CSVs under output-dir/predictions_by_epoch when available.",
    )
    return parser.parse_args()


def find_epochs(model_dir: Path) -> List[int]:
    pattern = re.compile(r"^checkpoint_(\d+)epochs\.pth$")
    epochs: List[int] = []
    for p in model_dir.glob("checkpoint_*epochs.pth"):
        m = pattern.match(p.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def ensure_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_run_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found under {model_dir}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def choose_output_dir(model_dir: Path, user_out: str) -> Path:
    if user_out:
        return Path(user_out)
    exp_name = model_dir.name
    return Path("CodeDiff/eval_result") / f"{exp_name}_all_epochs_eval"


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    cfg = load_run_config(model_dir)
    epochs = find_epochs(model_dir)
    if not epochs:
        raise FileNotFoundError(f"No checkpoint_*epochs.pth found under: {model_dir}")

    start_epoch = max(1, int(args.start_epoch))
    end_epoch = epochs[-1] if int(args.end_epoch) < 0 else int(args.end_epoch)
    epochs = [e for e in epochs if start_epoch <= e <= end_epoch]
    if not epochs:
        raise ValueError(
            f"No checkpoints in requested range: start={start_epoch}, end={end_epoch}"
        )

    output_dir = choose_output_dir(model_dir, args.output_dir)
    ensure_parent(output_dir)

    preds_root = output_dir / "predictions_by_epoch"
    if not args.skip_prediction_csv:
        ensure_parent(preds_root)

    dataset = cfg["dataset"]
    data_dir = Path(cfg["data_dir"])
    tokenizer_json = cfg["tokenizer_json"]

    tokenizer = Tokenizer.from_file(tokenizer_json)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    device = torch.device(args.device if args.device else cfg.get("device", "cpu"))

    model, diffusion = create_model_and_diffusion(
        vocab_size=vocab_size,
        pad_id=pad_id,
        diffusion_steps=int(cfg["diffusion_steps"]),
        line_encoder_type=cfg.get("line_encoder", "lstm"),
        codebert_local_path=(cfg.get("codebert_local_path") or None),
        freeze_codebert=bool(cfg.get("freeze_codebert", False)),
    )
    model.to(device)
    model.eval()

    train_release = cfg["train_release"]
    test_releases = list(cfg["test_releases"])

    max_tokens_per_line = int(cfg["max_tokens_per_line"])
    max_lines_per_file = int(cfg["max_lines_per_file"])
    line_window_size = int(cfg["line_window_size"])
    line_window_stride = int(cfg["line_window_stride"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    drop_comment = bool(cfg.get("drop_comment", False))
    drop_blank = bool(cfg.get("drop_blank", False))

    per_release_rows: List[Dict] = []
    per_epoch_rows: List[Dict] = []

    print(f"[info] dataset={dataset} epochs={epochs[0]}..{epochs[-1]} count={len(epochs)}")
    print(f"[info] model_dir={model_dir}")
    print(f"[info] output_dir={output_dir}")

    for epoch in epochs:
        ckpt_path = model_dir / f"checkpoint_{epoch}epochs.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        recalls: List[float] = []
        efforts: List[float] = []

        print(f"[epoch {epoch}] evaluating {len(test_releases)} releases ...")
        for release in test_releases:
            epoch_dir = preds_root / f"epoch_{epoch}" if not args.skip_prediction_csv else output_dir / ".tmp"
            ensure_parent(epoch_dir)
            pred_csv = epoch_dir / f"{release}.csv"

            if not (args.reuse_existing_predictions and pred_csv.exists()):
                generate_release_predictions(
                    model=model,
                    diffusion=diffusion,
                    tokenizer=tokenizer,
                    data_dir=data_dir,
                    dataset_name=dataset,
                    train_release=train_release,
                    target_release=release,
                    output_csv=pred_csv,
                    max_tokens_per_line=max_tokens_per_line,
                    max_lines_per_file=max_lines_per_file,
                    line_window_size=line_window_size,
                    line_window_stride=line_window_stride,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_comment=drop_comment,
                    drop_blank=drop_blank,
                    device=device,
                )

            metrics = eval_release(pred_csv)
            recalls.append(float(metrics["Recall@20%LOC"]))
            efforts.append(float(metrics["Effort@20%Recall"]))

            per_release_rows.append(
                {
                    "epoch": epoch,
                    "release": release,
                    "Recall@20%LOC": float(metrics["Recall@20%LOC"]),
                    "Effort@20%Recall": float(metrics["Effort@20%Recall"]),
                    "num_files_with_bug": int(metrics["num_files_with_bug"]),
                    "prediction_csv": str(pred_csv) if not args.skip_prediction_csv else "",
                }
            )

        epoch_mean_recall = sum(recalls) / len(recalls)
        epoch_mean_effort = sum(efforts) / len(efforts)
        per_epoch_rows.append(
            {
                "epoch": epoch,
                "mean_Recall@20%LOC": epoch_mean_recall,
                "mean_Effort@20%Recall": epoch_mean_effort,
            }
        )

        print(
            f"[epoch {epoch}] mean Recall@20%LOC={epoch_mean_recall:.6f}, "
            f"mean Effort@20%Recall={epoch_mean_effort:.6f}"
        )

    per_release_csv = output_dir / "recall_effort20_per_release_by_epoch.csv"
    with per_release_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "epoch",
                "release",
                "Recall@20%LOC",
                "Effort@20%Recall",
                "num_files_with_bug",
                "prediction_csv",
            ],
        )
        writer.writeheader()
        for row in per_release_rows:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "release": row["release"],
                    "Recall@20%LOC": f"{row['Recall@20%LOC']:.6f}",
                    "Effort@20%Recall": f"{row['Effort@20%Recall']:.6f}",
                    "num_files_with_bug": row["num_files_with_bug"],
                    "prediction_csv": row["prediction_csv"],
                }
            )

    per_epoch_csv = output_dir / "recall_effort20_mean_by_epoch.csv"
    with per_epoch_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["epoch", "mean_Recall@20%LOC", "mean_Effort@20%Recall"],
        )
        writer.writeheader()
        for row in per_epoch_rows:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "mean_Recall@20%LOC": f"{row['mean_Recall@20%LOC']:.6f}",
                    "mean_Effort@20%Recall": f"{row['mean_Effort@20%Recall']:.6f}",
                }
            )

    best_recall = max(per_epoch_rows, key=lambda x: x["mean_Recall@20%LOC"])
    best_effort = min(per_epoch_rows, key=lambda x: x["mean_Effort@20%Recall"])

    summary = {
        "model_dir": str(model_dir),
        "num_epochs_evaluated": len(per_epoch_rows),
        "best_recall_epoch": int(best_recall["epoch"]),
        "best_recall_value": float(best_recall["mean_Recall@20%LOC"]),
        "best_effort_epoch": int(best_effort["epoch"]),
        "best_effort_value": float(best_effort["mean_Effort@20%Recall"]),
        "per_epoch_csv": str(per_epoch_csv),
        "per_release_csv": str(per_release_csv),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        epochs_x = [int(row["epoch"]) for row in per_epoch_rows]
        recalls_y = [float(row["mean_Recall@20%LOC"]) for row in per_epoch_rows]
        efforts_y = [float(row["mean_Effort@20%Recall"]) for row in per_epoch_rows]

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        axes[0].plot(epochs_x, recalls_y, marker="o", linewidth=1.8)
        axes[0].set_title("Mean Recall@20%LOC by Epoch")
        axes[0].set_ylabel("Recall@20%LOC")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_x, efforts_y, marker="o", linewidth=1.8, color="#d62728")
        axes[1].set_title("Mean Effort@20%Recall by Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Effort@20%Recall")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"{dataset} all-epoch evaluation")
        fig.tight_layout()

        plot_path = output_dir / "recall_effort20_curves.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"[done] saved plot: {plot_path}")
    except Exception as exc:
        print(f"[warn] failed to plot curves: {exc}")

    print(f"[done] per-epoch csv: {per_epoch_csv}")
    print(f"[done] per-release csv: {per_release_csv}")
    print(f"[done] summary json: {summary_path}")


if __name__ == "__main__":
    main()
