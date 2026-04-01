from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tokenizers import Tokenizer
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODEDIFF_ROOT = PROJECT_ROOT / "codediff"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CODEDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEDIFF_ROOT))

try:
    from codediff.dataloader_utils import FileSequenceDataset
except ModuleNotFoundError:
    from dataloader_utils import FileSequenceDataset
from codediffpro.modeling import CodeSeqDiffuProtoModel, LabelGaussianDiffusion

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


def get_project_releases(dataset: str) -> Tuple[str, str, List[str]]:
    train_release = ALL_TRAIN_RELEASES[dataset]
    eval_releases = list(ALL_EVAL_RELEASES[dataset])
    val_release = eval_releases[0]
    # Keep the first eval release for validation, and start test releases from
    # the third project version onward.
    test_releases = eval_releases[1:]
    return train_release, val_release, test_releases


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
    balanced_sampling: bool,
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

    if not balanced_sampling:
        return DataLoader(concat, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    # Lightweight balancing by file-level positivity.
    file_labels = []
    for ds in datasets:
        for item in ds.files:
            has_pos = any(int(ex.label) == 1 for ex in item["lines"])
            file_labels.append(1 if has_pos else 0)

    pos_cnt = max(1, sum(file_labels))
    neg_cnt = max(1, len(file_labels) - pos_cnt)
    w_pos = 0.5 / pos_cnt
    w_neg = 0.5 / neg_cnt
    weights = [w_pos if y == 1 else w_neg for y in file_labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    return DataLoader(concat, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)


def _metrics_from_margin(m: torch.Tensor, y: torch.Tensor, tau: float) -> Dict[str, float]:
    pred = (m > float(tau)).long()
    tp = int(((pred == 1) & (y == 1)).sum().item())
    fp = int(((pred == 1) & (y == 0)).sum().item())
    fn = int(((pred == 0) & (y == 1)).sum().item())
    tn = int(((pred == 0) & (y == 0)).sum().item())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    pred_pos_rate = (tp + fp) / max(1, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "pred_pos_rate": pred_pos_rate,
    }


def calc_binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if scores.numel() == 0 or labels.numel() == 0:
        return 0.5
    y = labels.long()
    pos_cnt = int((y == 1).sum().item())
    neg_cnt = int((y == 0).sum().item())
    if pos_cnt == 0 or neg_cnt == 0:
        return 0.5

    # Rank-based AUC (equivalent to Mann-Whitney U) with tie handling.
    sorted_idx = torch.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = y[sorted_idx]

    n = int(sorted_scores.numel())
    ranks = torch.empty(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and float(sorted_scores[j].item()) == float(sorted_scores[i].item()):
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[i:j] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[sorted_labels == 1].sum().item())
    auc = (sum_pos_ranks - pos_cnt * (pos_cnt + 1) / 2.0) / (pos_cnt * neg_cnt)
    return float(max(0.0, min(1.0, auc)))


def select_tau_by_f1(m: torch.Tensor, y: torch.Tensor, candidate_count: int = 61) -> float:
    if m.numel() == 0:
        return 0.0
    lo = float(m.min().item())
    hi = float(m.max().item())
    if hi <= lo:
        return lo
    taus = torch.linspace(lo, hi, steps=max(5, int(candidate_count)), device=m.device)
    best_tau = float(taus[0].item())
    best_f1 = -1.0
    best_pred_pos = 1.0
    for tau in taus:
        metrics = _metrics_from_margin(m, y, float(tau.item()))
        f1 = float(metrics["f1"])
        pred_pos = float(metrics["pred_pos_rate"])
        if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and pred_pos < best_pred_pos):
            best_f1 = f1
            best_pred_pos = pred_pos
            best_tau = float(tau.item())
    return best_tau


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


def evaluate(
    model,
    diffusion,
    loader,
    device,
    proto_temperature: float,
    tau: float,
    tau_mode: str = "fixed",
    tau_candidates: int = 61,
) -> Dict[str, float]:
    model.eval()
    margins_y0: List[float] = []
    margins_y1: List[float] = []
    all_margins: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    by_file_scores: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    loss_sum = mse_sum = cons_sum = proto_sum = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            batch_t = {
                "input_ids": batch["input_ids"].to(device),
                "token_mask": batch["token_mask"].to(device),
                "line_labels": batch["line_labels"].to(device),
                "line_mask": batch["line_mask"].to(device),
            }
            losses = diffusion.training_losses(
                model=model,
                batch=batch_t,
                consistency_weight=0.0,
                proto_weight=1.0,
                proto_temperature=proto_temperature,
                proto_margin=0.0,
            )
            loss_sum += float(losses.loss.item())
            mse_sum += float(losses.mse.item())
            cons_sum += float(losses.consistency.item())
            proto_sum += float(losses.proto.item())
            steps += 1

            bsz, num_lines = batch_t["line_labels"].shape
            x_hat = diffusion.p_sample_loop(
                model=model,
                shape=(bsz, num_lines, model.label_emb_dim),
                model_kwargs={
                    "input_ids": batch_t["input_ids"],
                    "token_mask": batch_t["token_mask"],
                    "line_mask": batch_t["line_mask"],
                },
                device=device,
            )
            logits = diffusion.prototype_logits(x_hat, model.label_embedding.weight, temperature=proto_temperature)
            margin = (logits[..., 1] - logits[..., 0])
            probs = torch.softmax(logits, dim=-1)[..., 1]

            labels = batch_t["line_labels"]
            valid = batch_t["line_mask"].bool()
            m = margin[valid]
            y = labels[valid]
            all_margins.append(m.detach().cpu())
            all_labels.append(y.detach().cpu())

            if (y == 0).any():
                margins_y0.extend(m[y == 0].detach().cpu().tolist())
            if (y == 1).any():
                margins_y1.extend(m[y == 1].detach().cpu().tolist())

            filenames = batch["filenames"]
            line_numbers = batch["line_numbers"]
            for b in range(bsz):
                fname = filenames[b]
                for l in range(num_lines):
                    if int(batch_t["line_mask"][b, l].item()) == 0:
                        continue
                    ln = int(line_numbers[b, l].item())
                    prob = float(probs[b, l].item())
                    gt = int(batch_t["line_labels"][b, l].item())
                    entry = by_file_scores[fname].get(ln)
                    if entry is None:
                        by_file_scores[fname][ln] = {"prob_sum": prob, "cnt": 1.0, "gt": float(gt)}
                    else:
                        entry["prob_sum"] += prob
                        entry["cnt"] += 1.0

    if all_margins:
        m_all = torch.cat(all_margins, dim=0)
        y_all = torch.cat(all_labels, dim=0)
    else:
        m_all = torch.empty(0)
        y_all = torch.empty(0, dtype=torch.long)

    tau_used = float(tau)
    if tau_mode == "auto":
        tau_used = select_tau_by_f1(m_all, y_all, candidate_count=tau_candidates)
    cls_metrics = _metrics_from_margin(m_all, y_all, tau_used)
    auc = calc_binary_roc_auc(m_all, y_all)

    m0 = sum(margins_y0) / max(1, len(margins_y0))
    m1 = sum(margins_y1) / max(1, len(margins_y1))

    recall20_vals: List[float] = []
    effort20_vals: List[float] = []
    for _, line_map in by_file_scores.items():
        rows: List[Tuple[float, int]] = []
        for _, item in line_map.items():
            avg_prob = float(item["prob_sum"]) / max(1.0, float(item["cnt"]))
            gt = int(item["gt"])
            rows.append((avg_prob, gt))
        rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
        file_metrics = calc_file_metrics_at_ratio(rows_sorted, ratio=0.2)
        if file_metrics is None:
            continue
        recall20_vals.append(file_metrics[0])
        effort20_vals.append(file_metrics[1])

    recall20 = sum(recall20_vals) / len(recall20_vals) if recall20_vals else 0.0
    effort20 = sum(effort20_vals) / len(effort20_vals) if effort20_vals else 0.0
    rank_score = 2.0 * recall20 * effort20 / (recall20 + effort20 + 1e-12)

    return {
        "loss": loss_sum / max(1, steps),
        "mse": mse_sum / max(1, steps),
        "consistency": cons_sum / max(1, steps),
        "proto": proto_sum / max(1, steps),
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "acc": cls_metrics["acc"],
        "auc": auc,
        "pred_pos_rate": cls_metrics["pred_pos_rate"],
        "margin_y0_mean": m0,
        "margin_y1_mean": m1,
        "margin_gap": m1 - m0,
        "recall20": recall20,
        "effort20": effort20,
        "rank_score": rank_score,
        "tau": tau_used,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="codediffpro minimal prototype-only diffusion training")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", default="codediff/data/lineDP_dataset")
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--exp-name", required=True)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--max-tokens-per-line", type=int, default=48)
    p.add_argument("--max-lines-per-file", type=int, default=-1)
    p.add_argument("--line-window-size", type=int, default=256)
    p.add_argument("--line-window-stride", type=int, default=64)
    p.add_argument("--drop-comment", action="store_true")
    p.add_argument("--drop-blank", action="store_true")
    p.add_argument("--balanced-sampling", action="store_true")

    p.add_argument("--diffusion-steps", type=int, default=20)
    p.add_argument("--token-emb-dim", type=int, default=128)
    p.add_argument("--line-hidden-dim", type=int, default=128)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--label-emb-dim", type=int, default=256)
    p.add_argument("--num-cond-layers", type=int, default=2)
    p.add_argument("--num-denoise-layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--consistency-weight", type=float, default=0.1)
    p.add_argument("--proto-weight", type=float, default=0.2)
    p.add_argument("--proto-temperature", type=float, default=0.1)
    p.add_argument("--proto-margin", type=float, default=0.05)
    p.add_argument("--proto-pos-weight", type=float, default=1.0)
    p.add_argument("--proto-neg-weight", type=float, default=1.0)
    p.add_argument("--ranking-weight", type=float, default=0.0)
    p.add_argument("--ranking-margin", type=float, default=0.2)
    p.add_argument("--ranking-focal-gamma", type=float, default=0.0)
    p.add_argument("--ranking-hard-mix", type=float, default=1.0)
    p.add_argument("--ranking-loss-type", choices=["hinge", "softplus"], default="hinge")
    p.add_argument("--ranking-softplus-temp", type=float, default=0.1)
    p.add_argument("--hard-negative-ratio", type=float, default=0.0)
    p.add_argument("--hard-negative-weight", type=float, default=0.0)
    p.add_argument("--hard-negative-margin", type=float, default=0.0)
    p.add_argument("--positive-margin-weight", type=float, default=0.0)
    p.add_argument("--positive-margin-target", type=float, default=0.3)
    p.add_argument(
        "--loss-mode",
        choices=["classify_focus", "rank_focus", "hybrid", "reclassify", "rerank"],
        default="hybrid",
        help=(
            "Loss composition mode: classify_focus uses mse/consistency/proto only; "
            "rank_focus uses mse/consistency plus ranking/hard-negative (proto scaled down); "
            "hybrid uses all configured weights as-is."
        ),
    )
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--tau-mode", choices=["fixed", "auto"], default="fixed")
    p.add_argument("--tau-candidates", type=int, default=61)
    p.add_argument(
        "--best-model-metric",
        choices=["f1", "recall20", "effort20", "rank_score"],
        default="f1",
        help="Metric used to save checkpoint_best and selected_tau.",
    )
    p.add_argument(
        "--best-model-alpha",
        type=float,
        default=0.5,
        help="Weight for recall20 when best-model-metric=rank_score (rest goes to effort20).",
    )
    p.add_argument("--dynamic-aux-balance", action="store_true")
    p.add_argument("--dynamic-target-aux-ratio", type=float, default=0.35)
    p.add_argument("--dynamic-scale-min", type=float, default=0.1)
    p.add_argument("--dynamic-scale-max", type=float, default=100.0)
    p.add_argument("--dynamic-ema-momentum", type=float, default=0.9)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-all-checkpoints", action="store_true")
    p.add_argument("--no-save-all-checkpoints", action="store_false", dest="save_all_checkpoints")
    p.set_defaults(save_all_checkpoints=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    train_release, val_release, test_releases = get_project_releases(args.dataset)
    data_dir = Path(args.data_dir)

    train_csv = data_dir / f"{train_release}.csv"
    val_csv = data_dir / f"{val_release}.csv"

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
        balanced_sampling=args.balanced_sampling,
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
        balanced_sampling=False,
    )

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    model = CodeSeqDiffuProtoModel(
        vocab_size=tokenizer.get_vocab_size(),
        pad_id=pad_id,
        token_emb_dim=args.token_emb_dim,
        line_hidden_dim=args.line_hidden_dim,
        cond_dim=args.cond_dim,
        label_emb_dim=args.label_emb_dim,
        num_cond_layers=args.num_cond_layers,
        num_denoise_layers=args.num_denoise_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    )
    diffusion = LabelGaussianDiffusion(num_timesteps=args.diffusion_steps)

    device = torch.device(args.device)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_dir = Path("codediffpro/output/model") / args.dataset / args.exp_name
    loss_dir = Path("codediffpro/output/loss") / args.exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {**vars(args), "train_release": train_release, "val_release": val_release, "test_releases": test_releases}
    (model_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = loss_dir / f"{args.dataset}_train_{timestamp}.log"

    records: List[Dict[str, float]] = []
    best_f1 = -1.0
    best_auc = -1.0
    best_primary = -1.0
    best_tau = float(args.tau)

    eval_every = max(1, int(args.eval_every))
    save_every = max(1, int(args.save_every))

    metric_name = str(args.best_model_metric)
    metric_alpha = float(max(0.0, min(1.0, args.best_model_alpha)))

    def _primary_metric(v: Dict[str, float]) -> float:
        if metric_name == "f1":
            return float(v["f1"])
        if metric_name == "recall20":
            return float(v["recall20"])
        if metric_name == "effort20":
            return float(v["effort20"])
        return metric_alpha * float(v["recall20"]) + (1.0 - metric_alpha) * float(v["effort20"])

    loss_mode = str(args.loss_mode)
    if loss_mode == "reclassify":
        print("[warn] loss-mode 'reclassify' is deprecated, use 'classify_focus' instead.")
        loss_mode = "classify_focus"
    elif loss_mode == "rerank":
        print("[warn] loss-mode 'rerank' is deprecated, use 'rank_focus' instead.")
        loss_mode = "rank_focus"

    eff_consistency_weight = float(args.consistency_weight)
    eff_proto_weight = float(args.proto_weight)
    eff_ranking_weight = float(args.ranking_weight)
    eff_hn_weight = float(args.hard_negative_weight)
    eff_pos_weight = float(args.positive_margin_weight)
    eff_rank_hard_mix = float(args.ranking_hard_mix)
    if loss_mode == "classify_focus":
        eff_ranking_weight = 0.0
        eff_hn_weight = 0.0
        eff_pos_weight = 0.0
    elif loss_mode == "rank_focus":
        # Bias strongly toward ranking quality while keeping weak stabilizers.
        eff_consistency_weight = float(args.consistency_weight) * 0.5
        eff_proto_weight = float(args.proto_weight) * 0.08
        eff_ranking_weight = float(args.ranking_weight) * 2.2
        eff_hn_weight = float(args.hard_negative_weight) * 1.6
        eff_pos_weight = float(args.positive_margin_weight) * 1.5
        eff_rank_hard_mix = max(0.0, min(1.0, 0.7 * float(args.ranking_hard_mix)))

    with log_path.open("w", encoding="utf-8") as logf:
        def log(msg: str) -> None:
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

        log(
            f"[config] proto-only lr={args.lr} proto_weight={args.proto_weight} proto_temperature={args.proto_temperature} "
            f"proto_margin={args.proto_margin} ranking_weight={args.ranking_weight} hard_negative_weight={args.hard_negative_weight} "
            f"tau={args.tau} tau_mode={args.tau_mode} balanced_sampling={args.balanced_sampling} "
            f"loss_mode={loss_mode} effective(cons={eff_consistency_weight},proto={eff_proto_weight},rank={eff_ranking_weight},hn={eff_hn_weight},pos={eff_pos_weight},mix={eff_rank_hard_mix}) "
            f"best_model_metric={metric_name} alpha={metric_alpha:.3f} "
            f"dynamic_aux_balance={args.dynamic_aux_balance} target_aux_ratio={args.dynamic_target_aux_ratio} "
            f"scale_range=[{args.dynamic_scale_min},{args.dynamic_scale_max}] ema={args.dynamic_ema_momentum}"
        )

        ema_aux_scale = 1.0

        for epoch in range(1, args.num_epochs + 1):
            model.train()
            s_loss = s_mse = s_cons = s_proto = s_rank = s_hn = s_pos = s_aux_scale = s_aux_ratio = 0.0
            steps = 0

            for batch in train_loader:
                batch_t = {
                    "input_ids": batch["input_ids"].to(device),
                    "token_mask": batch["token_mask"].to(device),
                    "line_labels": batch["line_labels"].to(device),
                    "line_mask": batch["line_mask"].to(device),
                }
                losses = diffusion.training_losses(
                    model=model,
                    batch=batch_t,
                    consistency_weight=eff_consistency_weight,
                    proto_weight=eff_proto_weight,
                    proto_temperature=args.proto_temperature,
                    proto_margin=args.proto_margin,
                    proto_pos_weight=args.proto_pos_weight,
                    proto_neg_weight=args.proto_neg_weight,
                    ranking_weight=eff_ranking_weight,
                    ranking_margin=args.ranking_margin,
                    ranking_focal_gamma=args.ranking_focal_gamma,
                    ranking_hard_mix=eff_rank_hard_mix,
                    ranking_loss_type=args.ranking_loss_type,
                    ranking_softplus_temp=args.ranking_softplus_temp,
                    hard_negative_ratio=args.hard_negative_ratio,
                    hard_negative_weight=eff_hn_weight,
                    hard_negative_margin=args.hard_negative_margin,
                    positive_margin_weight=eff_pos_weight,
                    positive_margin_target=args.positive_margin_target,
                )
                mse_term = losses.mse
                aux_term = (
                    float(eff_consistency_weight) * losses.consistency
                    + float(eff_proto_weight) * losses.proto
                    + float(eff_ranking_weight) * losses.ranking
                    + float(eff_hn_weight) * losses.hard_negative
                    + float(eff_pos_weight) * losses.positive_margin
                )

                if args.dynamic_aux_balance:
                    target_ratio = float(max(0.01, min(0.95, args.dynamic_target_aux_ratio)))
                    desired_aux = mse_term.detach() * (target_ratio / max(1e-6, 1.0 - target_ratio))
                    aux_abs = float(aux_term.detach().abs().item())
                    if aux_abs > 0.0:
                        raw_scale = float((desired_aux / (aux_term.detach().abs() + 1e-12)).item())
                    else:
                        raw_scale = 1.0
                    raw_scale = max(float(args.dynamic_scale_min), min(float(args.dynamic_scale_max), raw_scale))
                    m = float(max(0.0, min(0.999, args.dynamic_ema_momentum)))
                    ema_aux_scale = m * ema_aux_scale + (1.0 - m) * raw_scale
                    aux_scale = ema_aux_scale
                else:
                    aux_scale = 1.0

                total_loss = mse_term + float(aux_scale) * aux_term
                aux_ratio = float((float(aux_scale) * aux_term.detach().item()) / (total_loss.detach().item() + 1e-12))

                optim.zero_grad(set_to_none=True)
                total_loss.backward()
                optim.step()

                s_loss += float(total_loss.item())
                s_mse += float(losses.mse.item())
                s_cons += float(losses.consistency.item())
                s_proto += float(losses.proto.item())
                s_rank += float(losses.ranking.item())
                s_hn += float(losses.hard_negative.item())
                s_pos += float(losses.positive_margin.item())
                s_aux_scale += float(aux_scale)
                s_aux_ratio += float(aux_ratio)
                steps += 1

            val = evaluate(
                model=model,
                diffusion=diffusion,
                loader=val_loader,
                device=device,
                proto_temperature=args.proto_temperature,
                tau=args.tau,
                tau_mode=args.tau_mode,
                tau_candidates=args.tau_candidates,
            )
            tr_loss = s_loss / max(1, steps)
            tr_mse = s_mse / max(1, steps)
            tr_cons = s_cons / max(1, steps)
            tr_proto = s_proto / max(1, steps)
            tr_rank = s_rank / max(1, steps)
            tr_hn = s_hn / max(1, steps)
            tr_pos = s_pos / max(1, steps)
            tr_aux_scale = s_aux_scale / max(1, steps)
            tr_aux_ratio = s_aux_ratio / max(1, steps)

            if epoch % eval_every != 0:
                log(
                    f"epoch={epoch} train_loss={tr_loss:.6f} train_mse={tr_mse:.6f} train_consistency={tr_cons:.6f} "
                    f"train_proto={tr_proto:.6f} train_rank={tr_rank:.6f} train_hn={tr_hn:.6f} train_pos={tr_pos:.6f} "
                    f"train_aux_scale={tr_aux_scale:.6f} train_aux_ratio={tr_aux_ratio:.6f} [skip_valid]"
                )
                continue

            log(
                f"epoch={epoch} train_loss={tr_loss:.6f} train_mse={tr_mse:.6f} train_consistency={tr_cons:.6f} train_proto={tr_proto:.6f} "
                f"train_rank={tr_rank:.6f} train_hn={tr_hn:.6f} train_pos={tr_pos:.6f} "
                f"train_aux_scale={tr_aux_scale:.6f} train_aux_ratio={tr_aux_ratio:.6f} "
                f"valid_loss={val['loss']:.6f} valid_f1={val['f1']:.6f} valid_auc={val['auc']:.6f} "
                f"valid_p={val['precision']:.6f} valid_r={val['recall']:.6f} "
                f"valid_pred_pos={val['pred_pos_rate']:.6f} margin_gap={val['margin_gap']:.6f} "
                f"recall20={val['recall20']:.6f} effort20={val['effort20']:.6f} rank_score={val['rank_score']:.6f} tau_used={val['tau']:.6f}"
            )

            if args.save_all_checkpoints and (epoch % save_every == 0):
                torch.save(
                    {
                        "epoch": epoch,
                        "best_tau": float(val["tau"]),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                    },
                    model_dir / f"checkpoint_{epoch}epochs.pth",
                )

            records.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_mse": tr_mse,
                    "train_consistency": tr_cons,
                    "train_proto": tr_proto,
                    "train_ranking": tr_rank,
                    "train_hard_negative": tr_hn,
                    "train_positive_margin": tr_pos,
                    "train_aux_scale": tr_aux_scale,
                    "train_aux_ratio": tr_aux_ratio,
                    "valid_loss": val["loss"],
                    "valid_precision": val["precision"],
                    "valid_recall": val["recall"],
                    "valid_f1": val["f1"],
                    "valid_auc": val["auc"],
                    "valid_pred_pos_rate": val["pred_pos_rate"],
                    "valid_margin_y0_mean": val["margin_y0_mean"],
                    "valid_margin_y1_mean": val["margin_y1_mean"],
                    "valid_margin_gap": val["margin_gap"],
                    "valid_recall20": val["recall20"],
                    "valid_effort20": val["effort20"],
                    "valid_rank_score": val["rank_score"],
                    "valid_tau": val["tau"],
                }
            )

            primary = _primary_metric(val)
            if primary > best_primary:
                best_primary = primary
                best_tau = float(val["tau"])
                torch.save(
                    {
                        "epoch": epoch,
                        "best_tau": best_tau,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                    },
                    model_dir / "checkpoint_best.pth",
                )
                log(
                    f"[best] epoch={epoch} metric={metric_name} score={best_primary:.6f} "
                    f"recall20={val['recall20']:.6f} effort20={val['effort20']:.6f} tau={best_tau:.6f}"
                )

            if val["f1"] > best_f1:
                best_f1 = val["f1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "best_tau": float(val["tau"]),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                    },
                    model_dir / "checkpoint_best_f1.pth",
                )
                log(f"[best_f1] epoch={epoch} valid_f1={best_f1:.6f} tau={float(val['tau']):.6f}")

            if val["auc"] > best_auc:
                best_auc = float(val["auc"])
                torch.save(
                    {
                        "epoch": epoch,
                        "best_tau": float(val["tau"]),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                    },
                    model_dir / "checkpoint_best_auc.pth",
                )
                log(f"[best_auc] epoch={epoch} valid_auc={best_auc:.6f} tau={float(val['tau']):.6f}")

        (model_dir / "selected_tau.txt").write_text(f"{best_tau:.8f}\n", encoding="utf-8")
        log(f"[done] best_tau={best_tau:.6f}")

    with (loss_dir / f"{args.dataset}-loss_record.csv").open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(records[0].keys()) if records else ["epoch"])
        writer.writeheader()
        for row in records:
            writer.writerow(row)


if __name__ == "__main__":
    main()
