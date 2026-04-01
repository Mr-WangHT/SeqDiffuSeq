from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tokenizers import Tokenizer
from torch.utils.data import ConcatDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codediff.dataloader_utils import FileSequenceDataset
from codediff.model_utils import create_model_and_diffusion


ALL_TRAIN_RELEASES = {
    "activemq": "activemq-5.0.0",
    "camel": "camel-1.4.0",
    "derby": "derby-10.2.1.6",
    "groovy": "groovy-1_5_7",
    "hbase": "hbase-0.94.0",
    "hive": "hive-0.9.0",
    "jruby": "jruby-1.1",
    "lucene": "lucene-2.3.0",
    "wicket": "wicket-1.3.0-incubating-beta-1",
}

ALL_EVAL_RELEASES = {
    "activemq": ["activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
    "camel": ["camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
    "derby": ["derby-10.3.1.4", "derby-10.5.1.1"],
    "groovy": ["groovy-1_6_BETA_1", "groovy-1_6_BETA_2"],
    "hbase": ["hbase-0.95.0", "hbase-0.95.2"],
    "hive": ["hive-0.10.0", "hive-0.12.0"],
    "jruby": ["jruby-1.4.0", "jruby-1.5.0", "jruby-1.7.0.preview1"],
    "lucene": ["lucene-2.9.0", "lucene-3.0.0", "lucene-3.1"],
    "wicket": ["wicket-1.3.0-beta2", "wicket-1.5.3"],
}


def _natural_key(version: str):
    parts = re.split(r"([0-9]+)", version.lower())
    key = []
    for p in parts:
        if p == "":
            continue
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p))
    return key


def discover_release_map(data_dir: Path) -> Dict[str, List[str]]:
    by_project: Dict[str, List[Tuple[str, str]]] = {}
    for file_path in sorted(data_dir.glob("*.csv")):
        stem = file_path.stem
        if "-" not in stem:
            continue
        project, version = stem.split("-", 1)
        by_project.setdefault(project, []).append((version, stem))

    release_map: Dict[str, List[str]] = {}
    for project, items in by_project.items():
        items_sorted = sorted(items, key=lambda x: _natural_key(x[0]))
        release_map[project] = [name for _, name in items_sorted]
    return release_map


def split_releases_for_project(releases: Sequence[str]) -> Tuple[str, str, List[str]]:
    if len(releases) < 2:
        raise ValueError("Each project must contain at least two versions (train + val/test)")
    train_release = releases[0]
    val_release = releases[1]
    test_releases = list(releases[1:])
    return train_release, val_release, test_releases


def get_project_releases(dataset: str, data_dir: Path) -> Tuple[str, str, List[str]]:
    if dataset in ALL_TRAIN_RELEASES and dataset in ALL_EVAL_RELEASES:
        train_release = ALL_TRAIN_RELEASES[dataset]
        eval_releases = list(ALL_EVAL_RELEASES[dataset])
        if not eval_releases:
            raise ValueError(f"No eval releases configured for dataset '{dataset}'")
        val_release = eval_releases[0]
        test_releases = eval_releases
        return train_release, val_release, test_releases

    release_map = discover_release_map(data_dir)
    if dataset not in release_map:
        raise ValueError(f"dataset '{dataset}' not found under {data_dir}. available: {sorted(release_map.keys())}")
    return split_releases_for_project(release_map[dataset])


def build_concat_file_dataloader(
    csv_paths: Sequence[Path],
    tokenizer,
    batch_size: int,
    max_tokens_per_line: int,
    max_lines_per_file: int,
    line_window_size: int,
    line_window_stride: int,
    shuffle: bool,
    num_workers: int,
    drop_comment: bool,
    drop_blank: bool,
) -> DataLoader:
    datasets = [
        FileSequenceDataset(
            csv_path=str(path),
            tokenizer=tokenizer,
            max_tokens_per_line=max_tokens_per_line,
            max_lines_per_file=max_lines_per_file,
            line_window_size=(line_window_size if line_window_size > 0 else None),
            line_window_stride=(line_window_stride if line_window_stride > 0 else None),
            split="all",
            drop_comment=drop_comment,
            drop_blank=drop_blank,
        )
        for path in csv_paths
    ]

    concat = ConcatDataset(datasets)
    collate_fn = datasets[0].collate_fn
    return DataLoader(
        concat,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def save_loss_records(loss_csv_path: Path, records: List[Dict]) -> None:
    loss_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(loss_csv_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "epoch",
                "train_loss",
                "valid_loss",
                "train_mse",
                "train_cls",
                "train_consistency",
                "valid_mse",
                "valid_cls",
                "valid_consistency",
                "valid_precision",
                "valid_recall",
                "valid_f1",
                "valid_acc",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def parse_float_or_default(value: str, default: float = 0.0) -> float:
    if value is None:
        return default
    s = str(value).strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def run_validation(
    model,
    diffusion,
    val_loader,
    device,
    cls_weight: float,
    consistency_weight: float,
    use_cls_head: bool,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    mse_sum = 0.0
    cls_sum = 0.0
    consistency_sum = 0.0
    steps = 0
    aggregated_rows: Dict[Tuple[str, int], Dict[str, float]] = {}
    with torch.no_grad():
        for batch in val_loader:
            batch_t = {
                "input_ids": batch["input_ids"].to(device),
                "token_mask": batch["token_mask"].to(device),
                "line_labels": batch["line_labels"].to(device),
                "line_mask": batch["line_mask"].to(device),
            }
            losses = diffusion.training_losses(
                model=model,
                batch=batch_t,
                cls_weight=cls_weight,
                consistency_weight=consistency_weight,
                use_cls_head=use_cls_head,
            )
            loss_sum += float(losses.loss.item())
            mse_sum += float(losses.mse.item())
            cls_sum += float(losses.cls.item())
            consistency_sum += float(losses.consistency.item())
            steps += 1

            if use_cls_head:
                sampled_emb = diffusion.p_sample_loop(
                    model=model,
                    shape=(
                        batch_t["line_labels"].shape[0],
                        batch_t["line_labels"].shape[1],
                        model.label_emb_dim,
                    ),
                    model_kwargs={
                        "input_ids": batch_t["input_ids"],
                        "token_mask": batch_t["token_mask"],
                        "line_mask": batch_t["line_mask"],
                    },
                    device=device,
                )
                logits = model.classify_from_embeddings(sampled_emb)
                probs = torch.softmax(logits, dim=-1)[..., 1]

                labels = batch_t["line_labels"]
                mask = batch_t["line_mask"].bool()
                filenames = batch["filenames"]
                line_numbers = batch["line_numbers"]

                bsz, num_lines = labels.shape
                for b in range(bsz):
                    for l in range(num_lines):
                        if not bool(mask[b, l].item()):
                            continue
                        key = (filenames[b], int(line_numbers[b, l].item()))
                        prob = float(probs[b, l].item())
                        label = int(labels[b, l].item())

                        entry = aggregated_rows.get(key)
                        if entry is None:
                            aggregated_rows[key] = {
                                "prob_sum": prob,
                                "count": 1.0,
                                "label": float(label),
                            }
                        else:
                            entry["prob_sum"] += prob
                            entry["count"] += 1.0

    if steps == 0:
        return {
            "loss": 0.0,
            "mse": 0.0,
            "cls": 0.0,
            "consistency": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "acc": 0.0,
        }

    if not use_cls_head:
        return {
            "loss": loss_sum / steps,
            "mse": mse_sum / steps,
            "cls": cls_sum / steps,
            "consistency": consistency_sum / steps,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "acc": 0.0,
        }

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for row in aggregated_rows.values():
        avg_prob = float(row["prob_sum"]) / max(1.0, float(row["count"]))
        pred = 1 if avg_prob >= 0.5 else 0
        label = int(row["label"])
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
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        "loss": loss_sum / steps,
        "mse": mse_sum / steps,
        "cls": cls_sum / steps,
        "consistency": consistency_sum / steps,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(acc),
    }


def generate_release_predictions(
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
                logits = model.classify_from_embeddings(sampled_emb)
                probs = torch.softmax(logits, dim=-1)[..., 1]
                preds = (probs >= 0.5).long()

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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodeDiff training entry (DeepLineDP-style experiment settings)")
    parser.add_argument("--dataset", required=True, help="project name, e.g., activemq")
    parser.add_argument("--data-dir", default="codediff/data/lineDP_dataset")
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--exp-name", default="")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-tokens-per-line", type=int, default=64)
    parser.add_argument("--max-lines-per-file", type=int, default=256, help="Max lines per sample; <=0 disables this cap")
    parser.add_argument(
        "--line-window-size",
        type=int,
        default=256,
        help="Sliding window size on lines per file. Use 256 to cover full files by chunks.",
    )
    parser.add_argument(
        "--line-window-stride",
        type=int,
        default=256,
        help="Sliding window stride on lines per file. 256 means non-overlap chunking.",
    )
    parser.add_argument("--drop-comment", action="store_true")
    parser.add_argument("--drop-blank", action="store_true")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--cls-weight", type=float, default=0.1)
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Weight for sampling-path consistency loss between adjacent reverse steps",
    )
    parser.add_argument(
        "--disable-cls-head",
        action="store_true",
        help="Disable classification-head loss in training/validation (cls loss term set to 0)",
    )
    parser.add_argument(
        "--line-encoder",
        choices=["lstm", "codebert"],
        default="lstm",
        help="Line encoder type: lstm or codebert",
    )
    parser.add_argument(
        "--codebert-local-path",
        default="",
        help="Local CodeBERT directory path; required when --line-encoder codebert",
    )
    parser.add_argument(
        "--freeze-codebert",
        action="store_true",
        help="Freeze CodeBERT backbone parameters when using codebert encoder",
    )
    parser.add_argument(
        "--codebert-unfreeze-epoch",
        type=int,
        default=-1,
        help=(
            "Epoch to start fine-tuning CodeBERT backbone when using codebert encoder. "
            "-1 means auto: first half frozen, second half trainable. "
            "0 means trainable from epoch 1. Ignored when --freeze-codebert is set."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path. Default: codediff/output/loss/CodeDiff/<exp_name>/<dataset>_train_<timestamp>.log",
    )

    parser.add_argument("--do-predict", action="store_true", help="Generate prediction CSVs after training")
    parser.add_argument("--predict-epoch", type=int, default=-1, help="Checkpoint epoch for prediction, -1 means latest epoch")
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs. Final epoch is always validated.",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Save only the best checkpoint (based on --best-metric) instead of saving every epoch checkpoint.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["valid_f1", "valid_acc", "valid_loss"],
        default="valid_f1",
        help="Metric used to decide best checkpoint when --save-best-only is enabled.",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=20,
        help="Print batch-level training diagnostics every N batches (<=0 disables)",
    )
    parser.add_argument(
        "--log-first-batch-shape",
        action="store_true",
        help="Always print tensor shapes and mask stats for the first batch of each epoch",
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=20,
        help="Window size (in batches) used to estimate recent loss trend",
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=1e-4,
        help="Absolute slope threshold to classify trend as stable",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    tokenizer = Tokenizer.from_file(args.tokenizer_json)

    train_release, val_release, test_releases = get_project_releases(args.dataset, data_dir)

    train_csv = data_dir / f"{train_release}.csv"
    val_csv = data_dir / f"{val_release}.csv"

    output_root = Path("codediff/output")
    model_dir = output_root / "model" / "CodeDiff" / args.dataset
    if args.exp_name:
        model_dir = model_dir / args.exp_name
    loss_dir = output_root / "loss" / "CodeDiff"
    if args.exp_name:
        loss_dir = loss_dir / args.exp_name
    prediction_dir = output_root / "prediction" / "CodeDiff"
    if args.exp_name:
        prediction_dir = prediction_dir / args.exp_name
    prediction_dir = prediction_dir / "within-release"

    model_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file:
        log_file_path = Path(args.log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_path = loss_dir / f"{args.dataset}_train_{timestamp}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_fout = open(log_file_path, "a", encoding="utf-8")

    def log_message(message: str) -> None:
        print(message)
        log_fout.write(message + "\n")
        log_fout.flush()

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    train_loader = build_concat_file_dataloader(
        csv_paths=[train_csv],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_tokens_per_line=args.max_tokens_per_line,
        max_lines_per_file=args.max_lines_per_file,
        line_window_size=args.line_window_size,
        line_window_stride=args.line_window_stride,
        shuffle=True,
        num_workers=args.num_workers,
        drop_comment=args.drop_comment,
        drop_blank=args.drop_blank,
    )
    val_loader = build_concat_file_dataloader(
        csv_paths=[val_csv],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_tokens_per_line=args.max_tokens_per_line,
        max_lines_per_file=args.max_lines_per_file,
        line_window_size=args.line_window_size,
        line_window_stride=args.line_window_stride,
        shuffle=False,
        num_workers=args.num_workers,
        drop_comment=args.drop_comment,
        drop_blank=args.drop_blank,
    )

    model, diffusion = create_model_and_diffusion(
        vocab_size=vocab_size,
        pad_id=pad_id,
        diffusion_steps=args.diffusion_steps,
        line_encoder_type=args.line_encoder,
        codebert_local_path=(args.codebert_local_path if args.codebert_local_path else None),
        freeze_codebert=args.freeze_codebert,
    )
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    effective_cls_weight = 0.0 if args.disable_cls_head else args.cls_weight
    log_message(
        f"[config] lr={args.lr} cls_weight={args.cls_weight} effective_cls_weight={effective_cls_weight} "
        f"consistency_weight={args.consistency_weight} disable_cls_head={args.disable_cls_head}"
    )

    codebert_unfreeze_epoch = None

    cfg = {
        **vars(args),
        "train_release": train_release,
        "val_release": val_release,
        "test_releases": test_releases,
        "resolved_log_file": str(log_file_path),
        "resolved_prediction_dir": str(prediction_dir),
    }
    with open(model_dir / "run_config.json", "w", encoding="utf-8") as fout:
        json.dump(cfg, fout, indent=2)

    checkpoint_pattern = re.compile(r"^checkpoint_(\d+)epochs\.pth$")
    existing_epochs: List[int] = []
    for p in model_dir.glob("checkpoint_*epochs.pth"):
        m = checkpoint_pattern.match(p.name)
        if m:
            existing_epochs.append(int(m.group(1)))

    train_records: List[Dict] = []
    best_epoch: Optional[int] = None
    best_metric_value: Optional[float] = None
    start_epoch = 1
    if existing_epochs:
        latest_epoch = max(existing_epochs)
        ckpt = torch.load(model_dir / f"checkpoint_{latest_epoch}epochs.pth", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = latest_epoch + 1

        loss_csv_path = loss_dir / f"{args.dataset}-loss_record.csv"
        if loss_csv_path.exists():
            with open(loss_csv_path, "r", encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    train_records.append(
                        {
                            "epoch": int(row["epoch"]),
                            "train_loss": float(row["train_loss"]),
                            "valid_loss": parse_float_or_default(row.get("valid_loss", ""), 0.0),
                            "train_mse": float(row["train_mse"]),
                            "train_cls": float(row["train_cls"]),
                            "train_consistency": parse_float_or_default(row.get("train_consistency", ""), 0.0),
                            "valid_mse": parse_float_or_default(row.get("valid_mse", ""), 0.0),
                            "valid_cls": parse_float_or_default(row.get("valid_cls", ""), 0.0),
                            "valid_consistency": parse_float_or_default(row.get("valid_consistency", ""), 0.0),
                            "valid_precision": parse_float_or_default(row.get("valid_precision", ""), 0.0),
                            "valid_recall": parse_float_or_default(row.get("valid_recall", ""), 0.0),
                            "valid_f1": parse_float_or_default(row.get("valid_f1", ""), 0.0),
                            "valid_acc": parse_float_or_default(row.get("valid_acc", ""), 0.0),
                        }
                    )

    if args.line_encoder == "codebert":
        if args.freeze_codebert:
            model.set_codebert_trainable(False)
            log_message("[codebert] freeze mode enabled by --freeze-codebert (backbone stays frozen for all epochs)")
        else:
            if args.codebert_unfreeze_epoch < 0:
                codebert_unfreeze_epoch = (args.num_epochs // 2) + 1
            elif args.codebert_unfreeze_epoch == 0:
                codebert_unfreeze_epoch = 1
            else:
                codebert_unfreeze_epoch = args.codebert_unfreeze_epoch

            initial_trainable = start_epoch >= codebert_unfreeze_epoch
            model.set_codebert_trainable(initial_trainable)
            log_message(
                f"[codebert] staged fine-tuning enabled: unfreeze_epoch={codebert_unfreeze_epoch}, "
                f"current_trainable={model.is_codebert_trainable()}"
            )

    for epoch in range(start_epoch, args.num_epochs + 1):
        if args.line_encoder == "codebert" and (not args.freeze_codebert) and codebert_unfreeze_epoch is not None:
            should_train_codebert = epoch >= codebert_unfreeze_epoch
            if should_train_codebert != model.is_codebert_trainable():
                model.set_codebert_trainable(should_train_codebert)
                state = "unfrozen (trainable)" if should_train_codebert else "frozen"
                log_message(f"[codebert] epoch={epoch}: backbone switched to {state}")

        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_cls_sum = 0.0
        train_consistency_sum = 0.0
        steps = 0
        ema_loss = None
        ema_mse = None
        ema_cls = None
        ema_consistency = None
        recent_loss_hist: List[float] = []

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_t = {
                "input_ids": batch["input_ids"].to(device),
                "token_mask": batch["token_mask"].to(device),
                "line_labels": batch["line_labels"].to(device),
                "line_mask": batch["line_mask"].to(device),
            }

            losses = diffusion.training_losses(
                model=model,
                batch=batch_t,
                cls_weight=args.cls_weight,
                consistency_weight=args.consistency_weight,
                use_cls_head=(not args.disable_cls_head),
            )

            optimizer.zero_grad(set_to_none=True)
            losses.loss.backward()
            optimizer.step()

            train_loss_sum += float(losses.loss.item())
            train_mse_sum += float(losses.mse.item())
            train_cls_sum += float(losses.cls.item())
            train_consistency_sum += float(losses.consistency.item())
            steps += 1

            cur_loss = float(losses.loss.item())
            cur_mse = float(losses.mse.item())
            cur_cls = float(losses.cls.item())
            cur_consistency = float(losses.consistency.item())

            if ema_loss is None:
                ema_loss = cur_loss
                ema_mse = cur_mse
                ema_cls = cur_cls
                ema_consistency = cur_consistency
            else:
                ema_decay = 0.98
                ema_loss = ema_decay * ema_loss + (1.0 - ema_decay) * cur_loss
                ema_mse = ema_decay * ema_mse + (1.0 - ema_decay) * cur_mse
                ema_cls = ema_decay * ema_cls + (1.0 - ema_decay) * cur_cls
                ema_consistency = ema_decay * ema_consistency + (1.0 - ema_decay) * cur_consistency

            should_log = args.log_every_batches > 0 and (batch_idx % args.log_every_batches == 0)
            should_log_shape = args.log_first_batch_shape and batch_idx == 1
            if should_log or should_log_shape:
                input_ids_shape = tuple(batch_t["input_ids"].shape)
                token_mask_shape = tuple(batch_t["token_mask"].shape)
                line_labels_shape = tuple(batch_t["line_labels"].shape)
                line_mask_shape = tuple(batch_t["line_mask"].shape)

                valid_line_ratio = float(batch_t["line_mask"].float().mean().item())
                valid_token_ratio = float(batch_t["token_mask"].float().mean().item())

                valid_lines = batch_t["line_mask"].bool()
                if valid_lines.any():
                    pos_ratio = float(batch_t["line_labels"][valid_lines].float().mean().item())
                else:
                    pos_ratio = 0.0

                recent_loss_hist.append(cur_loss)
                if args.trend_window > 0 and len(recent_loss_hist) > args.trend_window:
                    recent_loss_hist = recent_loss_hist[-args.trend_window :]

                slope = 0.0
                trend = "→"
                if len(recent_loss_hist) >= max(3, args.trend_window // 2):
                    n = len(recent_loss_hist)
                    x_mean = (n - 1) / 2.0
                    y_mean = sum(recent_loss_hist) / n
                    denom = 0.0
                    numer = 0.0
                    for i, y in enumerate(recent_loss_hist):
                        dx = i - x_mean
                        denom += dx * dx
                        numer += dx * (y - y_mean)
                    slope = numer / (denom + 1e-12)
                    if slope < -abs(args.trend_threshold):
                        trend = "↓"
                    elif slope > abs(args.trend_threshold):
                        trend = "↑"

                start_loss = recent_loss_hist[0]
                end_loss = recent_loss_hist[-1]
                rel_change = (end_loss - start_loss) / (abs(start_loss) + 1e-12)
                weighted_cls = effective_cls_weight * cur_cls
                weighted_consistency = args.consistency_weight * cur_consistency
                denom = max(cur_loss, 1e-12)
                mse_ratio = cur_mse / denom
                cls_ratio = weighted_cls / denom
                consistency_ratio = weighted_consistency / denom

                log_message(
                    f"[train][epoch {epoch}][batch {batch_idx}] "
                    f"shape input_ids={input_ids_shape} token_mask={token_mask_shape} "
                    f"line_labels={line_labels_shape} line_mask={line_mask_shape} | "
                    f"valid_line_ratio={valid_line_ratio:.4f} valid_token_ratio={valid_token_ratio:.4f} pos_ratio={pos_ratio:.4f} | "
                    f"loss={cur_loss:.6f} mse={cur_mse:.6f} cls={cur_cls:.6f} consistency={cur_consistency:.6f} | "
                    f"ema_loss={ema_loss:.6f} ema_mse={ema_mse:.6f} ema_cls={ema_cls:.6f} ema_consistency={ema_consistency:.6f} | "
                    f"parts(mse/cls/cons)={mse_ratio:.2%}/{cls_ratio:.2%}/{consistency_ratio:.2%} | "
                    f"trend(win={len(recent_loss_hist)})={trend} slope={slope:.6e} rel_change={rel_change:.2%}"
                )

        if steps == 0:
            train_stats = {"loss": 0.0, "mse": 0.0, "cls": 0.0, "consistency": 0.0}
        else:
            train_stats = {
                "loss": train_loss_sum / steps,
                "mse": train_mse_sum / steps,
                "cls": train_cls_sum / steps,
                "consistency": train_consistency_sum / steps,
            }

        should_validate = (epoch % max(1, args.eval_every_epochs) == 0) or (epoch == args.num_epochs)
        val_stats = None
        if should_validate:
            val_stats = run_validation(
                model=model,
                diffusion=diffusion,
                val_loader=val_loader,
                device=device,
                cls_weight=args.cls_weight,
                consistency_weight=args.consistency_weight,
                use_cls_head=(not args.disable_cls_head),
            )

        if val_stats is not None:
            log_message(
                f"epoch={epoch} "
                f"train_loss={train_stats['loss']:.6f} train_mse={train_stats['mse']:.6f} "
                f"train_cls={train_stats['cls']:.6f} train_consistency={train_stats['consistency']:.6f} "
                f"valid_loss={val_stats['loss']:.6f} valid_mse={val_stats['mse']:.6f} "
                f"valid_cls={val_stats['cls']:.6f} valid_consistency={val_stats['consistency']:.6f} "
                f"valid_p={val_stats['precision']:.6f} valid_r={val_stats['recall']:.6f} "
                f"valid_f1={val_stats['f1']:.6f} valid_acc={val_stats['acc']:.6f}"
            )
        else:
            log_message(
                f"epoch={epoch} "
                f"train_loss={train_stats['loss']:.6f} train_mse={train_stats['mse']:.6f} "
                f"train_cls={train_stats['cls']:.6f} train_consistency={train_stats['consistency']:.6f} "
                f"valid=SKIPPED(eval_every={args.eval_every_epochs})"
            )

        if args.save_best_only:
            if val_stats is not None:
                metric_value = None
                if args.best_metric == "valid_f1":
                    metric_value = val_stats["f1"]
                    improved = best_metric_value is None or metric_value > best_metric_value
                elif args.best_metric == "valid_acc":
                    metric_value = val_stats["acc"]
                    improved = best_metric_value is None or metric_value > best_metric_value
                else:
                    metric_value = val_stats["loss"]
                    improved = best_metric_value is None or metric_value < best_metric_value

                if improved:
                    best_metric_value = float(metric_value)
                    best_epoch = epoch
                    ckpt_path = model_dir / "checkpoint_best.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_metric": args.best_metric,
                            "best_metric_value": best_metric_value,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        ckpt_path,
                    )
                    log_message(
                        f"[best] updated: epoch={epoch} {args.best_metric}={best_metric_value:.6f} -> {ckpt_path}"
                    )
        else:
            if epoch % args.save_every_epochs == 0:
                ckpt_path = model_dir / f"checkpoint_{epoch}epochs.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )

        train_records.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "valid_loss": (val_stats["loss"] if val_stats is not None else ""),
                "train_mse": train_stats["mse"],
                "train_cls": train_stats["cls"],
                "train_consistency": train_stats["consistency"],
                "valid_mse": (val_stats["mse"] if val_stats is not None else ""),
                "valid_cls": (val_stats["cls"] if val_stats is not None else ""),
                "valid_consistency": (val_stats["consistency"] if val_stats is not None else ""),
                "valid_precision": (val_stats["precision"] if val_stats is not None else ""),
                "valid_recall": (val_stats["recall"] if val_stats is not None else ""),
                "valid_f1": (val_stats["f1"] if val_stats is not None else ""),
                "valid_acc": (val_stats["acc"] if val_stats is not None else ""),
            }
        )
        save_loss_records(loss_dir / f"{args.dataset}-loss_record.csv", train_records)

    if args.do_predict:
        predict_epoch = args.predict_epoch
        if predict_epoch < 0:
            if args.save_best_only:
                ckpt_path = model_dir / "checkpoint_best.pth"
            else:
                predict_epoch = args.num_epochs
                ckpt_path = model_dir / f"checkpoint_{predict_epoch}epochs.pth"
        else:
            ckpt_path = model_dir / f"checkpoint_{predict_epoch}epochs.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"predict checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        predict_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        for release in test_releases:
            out_csv = prediction_dir / f"{release}_{predict_timestamp}.csv"
            generate_release_predictions(
                model=model,
                diffusion=diffusion,
                tokenizer=tokenizer,
                data_dir=data_dir,
                dataset_name=args.dataset,
                train_release=train_release,
                target_release=release,
                output_csv=out_csv,
                max_tokens_per_line=args.max_tokens_per_line,
                max_lines_per_file=args.max_lines_per_file,
                line_window_size=args.line_window_size,
                line_window_stride=args.line_window_stride,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_comment=args.drop_comment,
                drop_blank=args.drop_blank,
                device=device,
            )
            log_message(f"saved predictions -> {out_csv}")

    log_fout.close()


if __name__ == "__main__":
    main()
