from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codediffpro.train_minimal import build_concat_file_dataloader
from codediffpro.modeling import CodeSeqDiffuProtoModel, LabelGaussianDiffusion


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    try:
        return Tokenizer.from_file(str(tokenizer_path))
    except Exception:
        vocab_path = tokenizer_path.with_name("vocab.json")
        merges_path = tokenizer_path.with_name("merges.txt")
        tokenizer = Tokenizer(models.BPE.from_file(str(vocab_path), str(merges_path), unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
        return tokenizer


def resolve_checkpoint_path(model_dir: Path, checkpoint: str) -> Path:
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_absolute():
        return ckpt_path
    return model_dir / ckpt_path


def calc_ifa(sorted_rows: List[Tuple[float, int]]) -> Optional[int]:
    if not sorted_rows:
        return None
    for idx, (_, gt) in enumerate(sorted_rows, start=1):
        if gt == 1:
            return idx
    return None


def calc_file_metrics_at_ratio(sorted_rows: List[Tuple[float, int]], ratio: float) -> Optional[Tuple[float, float]]:
    n = len(sorted_rows)
    if n == 0:
        return None
    total_true = sum(gt for _, gt in sorted_rows)
    if total_true == 0:
        return None

    k = max(1, int(round(n * ratio)))
    recall_ratio_loc = sum(gt for _, gt in sorted_rows[:k]) / total_true

    cum_true = 0
    count_prefix = 0
    for _, gt in sorted_rows:
        cum_true += gt
        recall = round(cum_true / total_true, 2)
        if recall <= ratio:
            count_prefix += 1
    effort_ratio_recall = count_prefix / n
    return recall_ratio_loc, effort_ratio_recall


def evaluate_prediction_csv(prediction_csv: Path) -> Dict[str, float]:
    line_by_file: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    line_items: List[Tuple[float, int]] = []

    with prediction_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            filename = row["filename"]
            prob = float(row["prediction-prob"])
            gt = int(row["line-level-ground-truth"])
            line_by_file[filename].append((prob, gt))
            line_items.append((prob, gt))

    ifa_vals: List[int] = []
    recall20_vals: List[float] = []
    effort20_vals: List[float] = []

    for rows in line_by_file.values():
        rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
        ifa = calc_ifa(rows_sorted)
        if ifa is not None:
            ifa_vals.append(ifa)
        metrics = calc_file_metrics_at_ratio(rows_sorted, ratio=0.2)
        if metrics is None:
            continue
        recall20_vals.append(metrics[0])
        effort20_vals.append(metrics[1])

    if line_items:
        scores = [s for s, _ in line_items]
        labels = [y for _, y in line_items]
        candidates = [min(scores) - 1e-12] + sorted(set(scores))
        best_f1 = -1.0
        best_tau = candidates[0]
        best_precision = 0.0
        best_recall = 0.0
        for tau in candidates:
            tp = fp = fn = tn = 0
            for score, label in line_items:
                pred = int(score > tau)
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1
                else:
                    tn += 1
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
                best_precision = precision
                best_recall = recall

        paired = sorted(zip(scores, labels), key=lambda x: x[0])
        pos = sum(labels)
        neg = len(labels) - pos
        if pos == 0 or neg == 0:
            auc = 0.5
        else:
            ranks: List[float] = []
            i = 0
            n = len(paired)
            while i < n:
                j = i + 1
                while j < n and paired[j][0] == paired[i][0]:
                    j += 1
                avg_rank = (i + 1 + j) / 2.0
                ranks.extend([avg_rank] * (j - i))
                i = j
            sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, paired) if y == 1)
            auc = (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    else:
        auc = 0.0
        best_f1 = 0.0
        best_tau = 0.0
        best_precision = 0.0
        best_recall = 0.0

    return {
        "line_auc": float(max(0.0, min(1.0, auc))),
        "line_best_f1": float(best_f1),
        "line_best_tau": float(best_tau),
        "line_precision_at_best_f1": float(best_precision),
        "line_recall_at_best_f1": float(best_recall),
        "ifa_mean": float(sum(ifa_vals) / len(ifa_vals)) if ifa_vals else 0.0,
        "recall20": float(sum(recall20_vals) / len(recall20_vals)) if recall20_vals else 0.0,
        "effort20": float(sum(effort20_vals) / len(effort20_vals)) if effort20_vals else 0.0,
        "num_buggy_files": float(len(ifa_vals)),
    }


def infer_one_release(
    model,
    diffusion,
    tokenizer,
    data_dir: Path,
    dataset_name: str,
    train_release: str,
    target_release: str,
    output_csv: Path,
    max_tokens_per_line: int,
    max_lines_per_file: int,
    line_window_size: int,
    line_window_stride: int,
    batch_size: int,
    num_workers: int,
    drop_comment: bool,
    drop_blank: bool,
    tau: float,
    proto_temperature: float,
    device: torch.device,
) -> None:
    model.eval()

    loader = build_concat_file_dataloader(
        csv_paths=[data_dir / f"{target_release}.csv"],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_tokens_per_line=max_tokens_per_line,
        max_lines_per_file=max_lines_per_file,
        line_window_size=line_window_size,
        line_window_stride=line_window_stride,
        shuffle=False,
        num_workers=num_workers,
        drop_comment=drop_comment,
        drop_blank=drop_blank,
        balanced_sampling=False,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        fields = [
            "project",
            "train",
            "test",
            "filename",
            "line-number",
            "line-level-ground-truth",
            "prediction-prob",
            "prediction-label",
            "margin",
        ]
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        agg: Dict[Tuple[str, int], Dict[str, object]] = {}

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                token_mask = batch["token_mask"].to(device)
                line_labels = batch["line_labels"].to(device)
                line_mask = batch["line_mask"].to(device)

                bsz, num_lines = line_labels.shape
                sampled_emb = diffusion.p_sample_loop(
                    model=model,
                    shape=(bsz, num_lines, model.label_emb_dim),
                    model_kwargs={
                        "input_ids": input_ids,
                        "token_mask": token_mask,
                        "line_mask": line_mask,
                    },
                    device=device,
                )
                logits = diffusion.prototype_logits(sampled_emb, model.label_embedding.weight, temperature=proto_temperature)
                probs = torch.softmax(logits, dim=-1)[..., 1]
                margins = (logits[..., 1] - logits[..., 0])

                filenames = batch["filenames"]
                line_numbers = batch["line_numbers"]

                for b in range(bsz):
                    for l in range(num_lines):
                        if int(line_mask[b, l].item()) == 0:
                            continue
                        key = (filenames[b], int(line_numbers[b, l].item()))
                        prob = float(probs[b, l].item())
                        margin = float(margins[b, l].item())
                        label = int(line_labels[b, l].item())
                        entry = agg.get(key)
                        if entry is None:
                            agg[key] = {
                                "project": dataset_name,
                                "train": train_release,
                                "test": target_release,
                                "filename": filenames[b],
                                "line-number": int(line_numbers[b, l].item()),
                                "line-level-ground-truth": label,
                                "prob_sum": prob,
                                "margin_sum": margin,
                                "count": 1,
                            }
                        else:
                            entry["prob_sum"] = float(entry["prob_sum"]) + prob
                            entry["margin_sum"] = float(entry["margin_sum"]) + margin
                            entry["count"] = int(entry["count"]) + 1

        for key in sorted(agg.keys(), key=lambda x: (x[0], x[1])):
            row = agg[key]
            c = max(1, int(row["count"]))
            avg_prob = float(row["prob_sum"]) / c
            avg_margin = float(row["margin_sum"]) / c
            writer.writerow(
                {
                    "project": row["project"],
                    "train": row["train"],
                    "test": row["test"],
                    "filename": row["filename"],
                    "line-number": row["line-number"],
                    "line-level-ground-truth": row["line-level-ground-truth"],
                    "prediction-prob": avg_prob,
                    "prediction-label": int(avg_margin > tau),
                    "margin": avg_margin,
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prototype-only inference for codediffpro minimal model")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--checkpoint", default="checkpoint_best.pth")
    p.add_argument("--release", required=True)
    p.add_argument("--tau", default="auto")
    p.add_argument("--output", default="")
    p.add_argument("--metrics-output", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))

    tokenizer = load_tokenizer(Path(cfg["tokenizer_json"]))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    model = CodeSeqDiffuProtoModel(
        vocab_size=tokenizer.get_vocab_size(),
        pad_id=pad_id,
        token_emb_dim=int(cfg.get("token_emb_dim", 128)),
        line_hidden_dim=int(cfg.get("line_hidden_dim", 128)),
        cond_dim=int(cfg.get("cond_dim", 256)),
        label_emb_dim=int(cfg.get("label_emb_dim", 256)),
        num_cond_layers=int(cfg.get("num_cond_layers", 2)),
        num_denoise_layers=int(cfg.get("num_denoise_layers", 2)),
        nhead=int(cfg.get("nhead", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    diffusion = LabelGaussianDiffusion(num_timesteps=int(cfg["diffusion_steps"]))

    device = torch.device(cfg.get("device", "cpu"))
    ckpt_path = resolve_checkpoint_path(model_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if str(args.tau).lower() == "auto":
        tau = 0.0
        if "best_tau" in ckpt:
            tau = float(ckpt["best_tau"])
        elif (model_dir / "selected_tau.txt").exists():
            tau = float((model_dir / "selected_tau.txt").read_text(encoding="utf-8").strip())
    else:
        tau = float(args.tau)

    out_path = Path(args.output) if args.output else Path("codediffpro/output/prediction") / cfg["exp_name"] / f"{args.release}.csv"

    infer_one_release(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        data_dir=Path(cfg["data_dir"]),
        dataset_name=cfg["dataset"],
        train_release=cfg["train_release"],
        target_release=args.release,
        output_csv=out_path,
        max_tokens_per_line=int(cfg["max_tokens_per_line"]),
        max_lines_per_file=int(cfg["max_lines_per_file"]),
        line_window_size=int(cfg["line_window_size"]),
        line_window_stride=int(cfg["line_window_stride"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        drop_comment=bool(cfg["drop_comment"]),
        drop_blank=bool(cfg["drop_blank"]),
        tau=tau,
        proto_temperature=float(cfg.get("proto_temperature", 0.1)),
        device=device,
    )

    metrics = evaluate_prediction_csv(out_path)
    metrics["release"] = args.release
    metrics["checkpoint"] = str(ckpt_path)
    metrics["tau_used"] = float(tau)

    metrics_path = Path(args.metrics_output) if args.metrics_output else out_path.with_name(f"{out_path.stem}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"saved -> {out_path} (tau={tau:.6f})")
    print(
        f"IFA={metrics['ifa_mean']:.6f} Recall20={metrics['recall20']:.6f} Effort20={metrics['effort20']:.6f} "
        f"LineAUC={metrics['line_auc']:.6f} LineBestF1={metrics['line_best_f1']:.6f}"
    )
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
