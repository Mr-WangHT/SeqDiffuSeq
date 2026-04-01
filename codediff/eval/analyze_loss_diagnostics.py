#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze learning-rate suitability and loss component balance.")
    parser.add_argument("--loss-csv", required=True, help="Path to codediff loss_record csv")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate used in training")
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--disable-cls-head", action="store_true")
    parser.add_argument("--out-json", default="", help="Optional output json path")
    return parser.parse_args()


def safe_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    v = str(row.get(key, "")).strip()
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def judge_lr(train_loss: List[float], lr: float) -> Dict[str, str]:
    if len(train_loss) < 3:
        return {"level": "unknown", "reason": "too few epochs for LR diagnosis"}

    first = train_loss[0]
    last = train_loss[-1]
    drop_ratio = (first - last) / (abs(first) + 1e-12)

    diffs = [train_loss[i] - train_loss[i - 1] for i in range(1, len(train_loss))]
    mean_diff = mean(diffs)
    diff_std = pstdev(diffs) if len(diffs) > 1 else 0.0
    osc_ratio = diff_std / (abs(mean_diff) + 1e-12)

    if mean_diff > 0:
        return {
            "level": "too_high_or_unstable",
            "reason": f"train loss increasing on average (mean diff={mean_diff:.6f})",
        }

    if drop_ratio < 0.05:
        return {
            "level": "likely_too_low",
            "reason": f"loss drop is too small (drop_ratio={drop_ratio:.2%})",
        }

    if osc_ratio > 4.0:
        return {
            "level": "possibly_high",
            "reason": f"loss trend is noisy (osc_ratio={osc_ratio:.2f})",
        }

    if 0.05 <= drop_ratio <= 0.95:
        return {
            "level": "reasonable",
            "reason": f"loss decreases stably (drop_ratio={drop_ratio:.2%}, osc_ratio={osc_ratio:.2f})",
        }

    return {
        "level": "aggressive_but_working",
        "reason": f"loss drops very fast (drop_ratio={drop_ratio:.2%}, osc_ratio={osc_ratio:.2f})",
    }


def judge_component_balance(
    train_loss: List[float],
    train_mse: List[float],
    train_cls: List[float],
    train_consistency: List[float],
    cls_weight: float,
    consistency_weight: float,
    disable_cls_head: bool,
) -> Dict[str, object]:
    ratios_mse: List[float] = []
    ratios_cls: List[float] = []
    ratios_cons: List[float] = []

    eff_cls_weight = 0.0 if disable_cls_head else cls_weight

    for l, m, c, k in zip(train_loss, train_mse, train_cls, train_consistency):
        denom = max(l, 1e-12)
        ratios_mse.append(m / denom)
        ratios_cls.append((eff_cls_weight * c) / denom)
        ratios_cons.append((consistency_weight * k) / denom)

    mean_mse = mean(ratios_mse) if ratios_mse else 0.0
    mean_cls = mean(ratios_cls) if ratios_cls else 0.0
    mean_cons = mean(ratios_cons) if ratios_cons else 0.0

    notes: List[str] = []
    if disable_cls_head and mean_cls > 1e-4:
        notes.append("classification term is not near zero while cls head is disabled")

    if mean_mse > 0.95:
        notes.append("MSE dominates total loss; auxiliary terms may be too weak")
    if mean_cons > 0.5:
        notes.append("consistency term is very strong; may over-regularize")
    if 0.02 <= mean_cons <= 0.30:
        notes.append("consistency contribution is in a moderate range")

    if not notes:
        notes.append("loss component balance looks acceptable")

    return {
        "mean_ratio_mse": mean_mse,
        "mean_ratio_cls": mean_cls,
        "mean_ratio_consistency": mean_cons,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    loss_csv = Path(args.loss_csv)
    if not loss_csv.exists():
        raise FileNotFoundError(f"loss csv not found: {loss_csv}")

    rows: List[Dict[str, str]] = []
    with loss_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        rows.extend(reader)

    train_loss = [safe_float(r, "train_loss") for r in rows]
    train_mse = [safe_float(r, "train_mse") for r in rows]
    train_cls = [safe_float(r, "train_cls") for r in rows]
    train_consistency = [safe_float(r, "train_consistency") for r in rows]

    lr_judgement = judge_lr(train_loss, args.lr)
    balance = judge_component_balance(
        train_loss=train_loss,
        train_mse=train_mse,
        train_cls=train_cls,
        train_consistency=train_consistency,
        cls_weight=args.cls_weight,
        consistency_weight=args.consistency_weight,
        disable_cls_head=args.disable_cls_head,
    )

    result = {
        "loss_csv": str(loss_csv),
        "num_epochs": len(rows),
        "lr": args.lr,
        "cls_weight": args.cls_weight,
        "consistency_weight": args.consistency_weight,
        "disable_cls_head": args.disable_cls_head,
        "lr_judgement": lr_judgement,
        "loss_component_balance": balance,
        "first_train_loss": train_loss[0] if train_loss else None,
        "last_train_loss": train_loss[-1] if train_loss else None,
    }

    print(json.dumps(result, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved diagnostics -> {out_path}")


if __name__ == "__main__":
    main()
