from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codediff.main import build_concat_file_dataloader, get_project_releases
from codediff.model_utils import create_model_and_diffusion


def load_run_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


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
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        fieldnames = [
            "project",
            "train",
            "test",
            "filename",
            "line-number",
            "line-level-ground-truth",
            "prediction-prob",
            "prediction-label",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        aggregated_rows: Dict[Tuple[str, int], Dict[str, object]] = {}

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                token_mask = batch["token_mask"].to(device)
                line_labels = batch["line_labels"].to(device)
                line_mask = batch["line_mask"].to(device)

                bsz, num_lines = line_labels.shape
                x_t = torch.randn(bsz, num_lines, model.label_emb_dim, device=device)
                t = torch.full((bsz,), diffusion.num_timesteps - 1, dtype=torch.long, device=device)
                _, pred_x0 = diffusion.p_sample_step(
                    model=model,
                    x_t=x_t,
                    t=t,
                    model_kwargs={
                        "input_ids": input_ids,
                        "token_mask": token_mask,
                        "line_mask": line_mask,
                    },
                )

                logits = model.classify_from_embeddings(pred_x0)
                probs = torch.softmax(logits, dim=-1)[..., 1]

                filenames = batch["filenames"]
                line_numbers = batch["line_numbers"]

                for b in range(bsz):
                    for l in range(num_lines):
                        if int(line_mask[b, l].item()) == 0:
                            continue
                        filename = filenames[b]
                        line_no = int(line_numbers[b, l].item())
                        key = (filename, line_no)

                        prob = float(probs[b, l].item())
                        label = int(line_labels[b, l].item())
                        entry = aggregated_rows.get(key)
                        if entry is None:
                            aggregated_rows[key] = {
                                "project": dataset_name,
                                "train": train_release,
                                "test": target_release,
                                "filename": filename,
                                "line-number": line_no,
                                "line-level-ground-truth": label,
                                "prob_sum": prob,
                                "count": 1,
                            }
                        else:
                            entry["prob_sum"] = float(entry["prob_sum"]) + prob
                            entry["count"] = int(entry["count"]) + 1

        for key in sorted(aggregated_rows.keys(), key=lambda x: (x[0], x[1])):
            row = aggregated_rows[key]
            avg_prob = float(row["prob_sum"]) / max(1, int(row["count"]))
            writer.writerow(
                {
                    "project": row["project"],
                    "train": row["train"],
                    "test": row["test"],
                    "filename": row["filename"],
                    "line-number": row["line-number"],
                    "line-level-ground-truth": row["line-level-ground-truth"],
                    "prediction-prob": avg_prob,
                    "prediction-label": int(avg_prob >= 0.5),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-step inference for CodeDiff checkpoint")
    parser.add_argument("--model-dir", required=True, help="Path to experiment model dir containing run_config.json")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pth",
        help="Checkpoint filename under model-dir (default: checkpoint_best.pth)",
    )
    parser.add_argument(
        "--output-subdir",
        default="one-step",
        help="Subdir under prediction experiment folder (default: one-step)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    cfg = load_run_config(model_dir)
    dataset = cfg["dataset"]
    data_dir = Path(cfg["data_dir"])
    tokenizer_json = cfg["tokenizer_json"]
    exp_name = cfg.get("exp_name", "")

    tokenizer = Tokenizer.from_file(tokenizer_json)

    train_release, _, test_releases = get_project_releases(dataset, data_dir)

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    model, diffusion = create_model_and_diffusion(
        vocab_size=vocab_size,
        pad_id=pad_id,
        diffusion_steps=int(cfg["diffusion_steps"]),
        line_encoder_type=cfg.get("line_encoder", "lstm"),
        codebert_local_path=(cfg.get("codebert_local_path") or None),
        freeze_codebert=bool(cfg.get("freeze_codebert", False)),
    )

    device = torch.device(cfg.get("device", "cpu"))
    model.to(device)

    ckpt_path = model_dir / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    prediction_dir = Path("codediff/output/prediction/CodeDiff")
    if exp_name:
        prediction_dir = prediction_dir / exp_name
    prediction_dir = prediction_dir / args.output_subdir
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for release in test_releases:
        out_csv = prediction_dir / f"{release}.csv"
        infer_one_release(
            model=model,
            diffusion=diffusion,
            tokenizer=tokenizer,
            data_dir=data_dir,
            dataset_name=dataset,
            train_release=train_release,
            target_release=release,
            output_csv=out_csv,
            max_tokens_per_line=int(cfg["max_tokens_per_line"]),
            max_lines_per_file=int(cfg["max_lines_per_file"]),
            line_window_size=int(cfg["line_window_size"]),
            line_window_stride=int(cfg["line_window_stride"]),
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["num_workers"]),
            drop_comment=bool(cfg["drop_comment"]),
            drop_blank=bool(cfg["drop_blank"]),
            device=device,
        )
        print(f"saved one-step predictions -> {out_csv}")


if __name__ == "__main__":
    main()
