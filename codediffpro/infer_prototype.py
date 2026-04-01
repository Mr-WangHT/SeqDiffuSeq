from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codediffpro.train_minimal import build_concat_file_dataloader
from codediffpro.modeling import CodeSeqDiffuProtoModel, LabelGaussianDiffusion


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))

    tokenizer = Tokenizer.from_file(cfg["tokenizer_json"])
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
    ckpt = torch.load(model_dir / args.checkpoint, map_location=device)
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
    print(f"saved -> {out_path} (tau={tau:.6f})")


if __name__ == "__main__":
    main()
