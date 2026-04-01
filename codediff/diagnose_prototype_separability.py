from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codediff.main import build_concat_file_dataloader
from codediff.model_utils import create_model_and_diffusion


def load_run_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def prototype_logits(label_embeddings: torch.Tensor, prototypes: torch.Tensor, sim: str) -> torch.Tensor:
    if sim == "cosine":
        z = F.normalize(label_embeddings, dim=-1)
        p = F.normalize(prototypes, dim=-1)
        return torch.einsum("bld,cd->blc", z, p)
    return torch.einsum("bld,cd->blc", label_embeddings, prototypes)


def binary_metrics_from_probs(labels: List[int], probs: List[float], threshold: float = 0.5) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for y, p in zip(labels, probs):
        y_hat = 1 if p >= threshold else 0
        if y_hat == 1 and y == 1:
            tp += 1
        elif y_hat == 1 and y == 0:
            fp += 1
        elif y_hat == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    idx = int(max(0, min(len(values) - 1, round(q * (len(values) - 1)))))
    return values[idx]


def parse_release_list(text: str | None, fallback: Iterable[str]) -> List[str]:
    if text is None or text.strip() == "":
        return list(fallback)
    items = [x.strip() for x in text.split(",") if x.strip()]
    return items if items else list(fallback)


def parse_t_points(text: str, num_steps: int) -> List[int]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        items = ["0", str(num_steps // 2), str(num_steps - 1)]
    points = []
    for s in items:
        v = int(s)
        v = max(0, min(num_steps - 1, v))
        points.append(v)
    # keep order but unique
    seen = set()
    out = []
    for p in points:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose prototype separability of a trained diffusion model")
    parser.add_argument("--model-dir", required=True, help="Model experiment dir containing run_config.json")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--sim", choices=["dot", "cosine"], default="cosine")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--releases",
        default="",
        help="Comma-separated releases to analyze. Empty means run_config.test_releases",
    )
    parser.add_argument(
        "--t-points",
        default="0,10,19",
        help="Comma-separated diffusion t for diagnostics (clamped to valid range)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    cfg = load_run_config(model_dir)

    dataset = cfg["dataset"]
    data_dir = Path(cfg["data_dir"])
    tokenizer = Tokenizer.from_file(cfg["tokenizer_json"])
    exp_name = cfg.get("exp_name", "")

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
    model.eval()

    releases = parse_release_list(args.releases, cfg.get("test_releases", []))
    if not releases:
        raise ValueError("No releases provided and run_config.test_releases is empty")

    t_points = parse_t_points(args.t_points, int(cfg["diffusion_steps"]))

    diag_root = Path("CodeDiff/eval_result/diagnostics")
    if exp_name:
        diag_root = diag_root / exp_name
    diag_root.mkdir(parents=True, exist_ok=True)

    per_release_rows: List[Dict[str, object]] = []
    t_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        prototypes = model.label_embedding.weight

        for release in releases:
            loader = build_concat_file_dataloader(
                csv_paths=[data_dir / f"{release}.csv"],
                tokenizer=tokenizer,
                batch_size=int(cfg["batch_size"]),
                max_tokens_per_line=int(cfg["max_tokens_per_line"]),
                max_lines_per_file=int(cfg["max_lines_per_file"]),
                line_window_size=int(cfg["line_window_size"]),
                line_window_stride=int(cfg["line_window_stride"]),
                shuffle=False,
                num_workers=int(cfg["num_workers"]),
                drop_comment=bool(cfg["drop_comment"]),
                drop_blank=bool(cfg["drop_blank"]),
            )

            labels_full: List[int] = []
            probs_full: List[float] = []
            margins_full: List[float] = []

            # t-wise stats containers
            t_labels: Dict[int, List[int]] = {t: [] for t in t_points}
            t_probs: Dict[int, List[float]] = {t: [] for t in t_points}
            t_margins_pos: Dict[int, List[float]] = {t: [] for t in t_points}
            t_margins_neg: Dict[int, List[float]] = {t: [] for t in t_points}

            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                token_mask = batch["token_mask"].to(device)
                line_labels = batch["line_labels"].to(device)
                line_mask = batch["line_mask"].to(device)
                valid_mask = line_mask.bool()

                bsz, num_lines = line_labels.shape

                # Full diffusion sampling diagnosis.
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
                logits = prototype_logits(sampled_emb, prototypes, args.sim)
                probs = torch.softmax(logits, dim=-1)[..., 1]
                margins = (logits[..., 1] - logits[..., 0])

                y = line_labels[valid_mask].detach().cpu().tolist()
                p = probs[valid_mask].detach().cpu().tolist()
                m = margins[valid_mask].detach().cpu().tolist()
                labels_full.extend(int(v) for v in y)
                probs_full.extend(float(v) for v in p)
                margins_full.extend(float(v) for v in m)

                # t-wise x0_hat separability diagnosis.
                x_start = model.get_label_embeddings(line_labels)
                for t in t_points:
                    noise = torch.randn_like(x_start)
                    t_batch = torch.full((bsz,), t, dtype=torch.long, device=device)
                    x_t = diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
                    pred_noise = model(
                        noisy_label_emb=x_t,
                        timesteps=t_batch,
                        input_ids=input_ids,
                        token_mask=token_mask,
                        line_mask=line_mask,
                    )
                    x0_hat = diffusion.predict_x0_from_eps(x_t=x_t, t=t_batch, eps=pred_noise)
                    logits_t = prototype_logits(x0_hat, prototypes, args.sim)
                    probs_t = torch.softmax(logits_t, dim=-1)[..., 1]
                    margins_t = (logits_t[..., 1] - logits_t[..., 0])

                    y_t = line_labels[valid_mask].detach().cpu().tolist()
                    p_t = probs_t[valid_mask].detach().cpu().tolist()
                    m_t = margins_t[valid_mask].detach().cpu().tolist()

                    t_labels[t].extend(int(v) for v in y_t)
                    t_probs[t].extend(float(v) for v in p_t)
                    for yy, mm in zip(y_t, m_t):
                        if int(yy) == 1:
                            t_margins_pos[t].append(float(mm))
                        else:
                            t_margins_neg[t].append(float(mm))

            if not labels_full:
                continue

            probs_sorted = sorted(probs_full)
            metrics_full = binary_metrics_from_probs(labels_full, probs_full, threshold=args.threshold)
            per_release_rows.append(
                {
                    "release": release,
                    "n": len(labels_full),
                    "gt_pos_rate": sum(labels_full) / max(1, len(labels_full)),
                    "pred_pos_rate": sum(1 for v in probs_full if v >= args.threshold) / max(1, len(probs_full)),
                    "prob_q10": quantile(probs_sorted, 0.10),
                    "prob_q50": quantile(probs_sorted, 0.50),
                    "prob_q90": quantile(probs_sorted, 0.90),
                    **metrics_full,
                }
            )

            for t in t_points:
                if not t_labels[t]:
                    continue
                m_pos = t_margins_pos[t]
                m_neg = t_margins_neg[t]
                metrics_t = binary_metrics_from_probs(t_labels[t], t_probs[t], threshold=args.threshold)
                t_rows.append(
                    {
                        "release": release,
                        "t": t,
                        "n": len(t_labels[t]),
                        "pos_margin_mean": (sum(m_pos) / max(1, len(m_pos))),
                        "neg_margin_mean": (sum(m_neg) / max(1, len(m_neg))),
                        "margin_gap": (sum(m_pos) / max(1, len(m_pos))) - (sum(m_neg) / max(1, len(m_neg))),
                        **metrics_t,
                    }
                )

    # Save outputs.
    release_csv = diag_root / "prototype_full_diffusion_release_stats.csv"
    with release_csv.open("w", encoding="utf-8", newline="") as fout:
        fields = [
            "release",
            "n",
            "gt_pos_rate",
            "pred_pos_rate",
            "prob_q10",
            "prob_q50",
            "prob_q90",
            "precision",
            "recall",
            "f1",
            "acc",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in per_release_rows:
            out = dict(row)
            for k in [
                "gt_pos_rate",
                "pred_pos_rate",
                "prob_q10",
                "prob_q50",
                "prob_q90",
                "precision",
                "recall",
                "f1",
                "acc",
            ]:
                out[k] = f"{float(out[k]):.6f}"
            writer.writerow(out)

    t_csv = diag_root / "prototype_t_separability_stats.csv"
    with t_csv.open("w", encoding="utf-8", newline="") as fout:
        fields = [
            "release",
            "t",
            "n",
            "pos_margin_mean",
            "neg_margin_mean",
            "margin_gap",
            "precision",
            "recall",
            "f1",
            "acc",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in t_rows:
            out = dict(row)
            for k in [
                "pos_margin_mean",
                "neg_margin_mean",
                "margin_gap",
                "precision",
                "recall",
                "f1",
                "acc",
            ]:
                out[k] = f"{float(out[k]):.6f}"
            writer.writerow(out)

    summary_json = diag_root / "prototype_diagnosis_summary.json"
    summary = {
        "model_dir": str(model_dir),
        "checkpoint": str(args.checkpoint),
        "sim": args.sim,
        "threshold": args.threshold,
        "releases": releases,
        "t_points": t_points,
        "release_stats_csv": str(release_csv),
        "t_stats_csv": str(t_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved {release_csv}")
    print(f"saved {t_csv}")
    print(f"saved {summary_json}")


if __name__ == "__main__":
    main()
