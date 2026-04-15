#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EPOCH_CKPT_RE = re.compile(r"^checkpoint_(\d+)epochs\.pth$")


# metric_name -> (key_in_log, maximize)
DEFAULT_METRICS: Dict[str, Tuple[str, bool]] = {
    "f1": ("valid_f1", True),
    "auc": ("valid_auc", True),
    "recall20": ("recall20", True),
    "effort20": ("effort20", False),
    "rank_score": ("rank_score", True),
    "ifa": ("ifa", False),
    "valid_loss": ("valid_loss", False),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prune per-epoch checkpoints: keep only epochs that are best for chosen metrics "
            "for each experiment under output/model."
        )
    )
    p.add_argument("--model-root", default="codediffpro/output/model", help="Root directory of model outputs.")
    p.add_argument("--loss-root", default="codediffpro/output/loss", help="Root directory of loss logs.")
    p.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS.keys()),
        help=(
            "Comma-separated metrics to use when selecting checkpoints. "
            f"Supported: {', '.join(DEFAULT_METRICS.keys())}"
        ),
    )
    p.add_argument(
        "--keep-alias-checkpoints",
        action="store_true",
        help="Also keep alias files like checkpoint_best.pth / checkpoint_best_f1.pth / checkpoint_best_auc.pth.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only print what would be deleted.",
    )
    return p.parse_args()


def parse_epoch_metrics_from_log(log_path: Path) -> Dict[int, Dict[str, float]]:
    epoch_metrics: Dict[int, Dict[str, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("epoch="):
                continue
            fields: Dict[str, str] = {}
            for token in line.split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                fields[k] = v
            if "epoch" not in fields:
                continue
            try:
                epoch = int(fields["epoch"])
            except ValueError:
                continue

            row: Dict[str, float] = {}
            for k, v in fields.items():
                if k == "epoch":
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    continue
            if row:
                # Later log lines overwrite older values for the same epoch (useful for resumed runs).
                epoch_metrics[epoch] = row
    return epoch_metrics


def collect_experiment_metrics(loss_dir: Path) -> Dict[int, Dict[str, float]]:
    logs = sorted(loss_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    merged: Dict[int, Dict[str, float]] = {}
    for log_path in logs:
        parsed = parse_epoch_metrics_from_log(log_path)
        for epoch, vals in parsed.items():
            merged[epoch] = vals
    return merged


def best_epoch_for_metric(
    epoch_metrics: Dict[int, Dict[str, float]], metric_key: str, maximize: bool
) -> Optional[Tuple[int, float]]:
    best_epoch: Optional[int] = None
    best_val: Optional[float] = None
    for epoch in sorted(epoch_metrics.keys()):
        vals = epoch_metrics[epoch]
        if metric_key not in vals:
            continue
        val = vals[metric_key]
        if best_val is None:
            best_epoch = epoch
            best_val = val
            continue
        better = val > best_val if maximize else val < best_val
        if better:
            best_epoch = epoch
            best_val = val
    if best_epoch is None or best_val is None:
        return None
    return best_epoch, best_val


def list_epoch_checkpoints(exp_model_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in exp_model_dir.glob("checkpoint_*epochs.pth"):
        m = EPOCH_CKPT_RE.match(p.name)
        if not m:
            continue
        out[int(m.group(1))] = p
    return out


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root)
    loss_root = Path(args.loss_root)

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metric_names if m not in DEFAULT_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}. Supported: {', '.join(DEFAULT_METRICS.keys())}"
        )

    exp_dirs: List[Path] = []
    for dataset_dir in sorted(model_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for exp_dir in sorted(dataset_dir.iterdir()):
            if exp_dir.is_dir():
                exp_dirs.append(exp_dir)

    total_delete = 0
    total_keep = 0
    total_exp = 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] model_root={model_root} loss_root={loss_root} metrics={metric_names}")

    for exp_model_dir in exp_dirs:
        exp_name = exp_model_dir.name
        loss_dir = loss_root / exp_name
        if not loss_dir.exists():
            continue

        epoch_metrics = collect_experiment_metrics(loss_dir)
        if not epoch_metrics:
            continue

        epoch_ckpts = list_epoch_checkpoints(exp_model_dir)
        if not epoch_ckpts:
            continue

        best_epochs: Dict[str, int] = {}
        for metric_name in metric_names:
            metric_key, maximize = DEFAULT_METRICS[metric_name]
            picked = best_epoch_for_metric(epoch_metrics, metric_key=metric_key, maximize=maximize)
            if picked is None:
                continue
            best_epochs[metric_name] = picked[0]

        if not best_epochs:
            continue

        keep_epochs = sorted(set(best_epochs.values()))
        keep_files: List[Path] = [epoch_ckpts[e] for e in keep_epochs if e in epoch_ckpts]
        delete_files: List[Path] = [p for e, p in sorted(epoch_ckpts.items()) if e not in set(keep_epochs)]

        if args.keep_alias_checkpoints:
            for alias in ("checkpoint_best.pth", "checkpoint_best_f1.pth", "checkpoint_best_auc.pth"):
                alias_path = exp_model_dir / alias
                if alias_path.exists():
                    keep_files.append(alias_path)

        total_exp += 1
        total_keep += len(keep_files)
        total_delete += len(delete_files)

        best_info = ", ".join([f"{k}:{v}" for k, v in sorted(best_epochs.items())])
        print(f"\n[exp] {exp_name}")
        print(f"  best_epochs -> {best_info}")
        print(f"  keep_epoch_ckpts -> {[p.name for p in keep_files if EPOCH_CKPT_RE.match(p.name)]}")
        print(f"  delete_count -> {len(delete_files)}")

        if args.apply:
            for p in delete_files:
                p.unlink(missing_ok=True)

    print(f"\n[{mode}] experiments={total_exp} keep_files={total_keep} delete_files={total_delete}")


if __name__ == "__main__":
    main()
