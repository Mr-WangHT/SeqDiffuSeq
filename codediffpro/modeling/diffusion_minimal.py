from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


def _extract(arr: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    out = arr.to(t.device)[t].float()
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out.expand(x_shape)


@dataclass
class DiffusionLosses:
    loss: torch.Tensor
    mse: torch.Tensor
    consistency: torch.Tensor
    proto: torch.Tensor
    ranking: torch.Tensor
    hard_negative: torch.Tensor
    positive_margin: torch.Tensor


class LabelGaussianDiffusion:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_timesteps = num_timesteps
        self.betas = linear_beta_schedule(num_timesteps, beta_start=beta_start, beta_end=beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=self.alphas_cumprod.dtype), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * eps

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, pred_eps: torch.Tensor):
        pred_x0 = self.predict_x0_from_eps(x_t=x_t, t=t, eps=pred_eps)
        posterior_mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0 + _extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_var = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var = _extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_var, posterior_log_var, pred_x0

    def p_sample_step(self, model, x_t: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, torch.Tensor]):
        pred_eps = model(
            noisy_label_emb=x_t,
            timesteps=t,
            input_ids=model_kwargs["input_ids"],
            token_mask=model_kwargs["token_mask"],
            line_mask=model_kwargs["line_mask"],
        )
        mean, _, log_var, pred_x0 = self.p_mean_variance(x_t=x_t, t=t, pred_eps=pred_eps)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.dim() - 1)))
        x_prev = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return x_prev, pred_x0

    def p_sample_loop(self, model, shape, model_kwargs: Dict[str, torch.Tensor], device: torch.device):
        x_t = torch.randn(*shape, device=device)
        pred_x0 = None
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            x_t, pred_x0 = self.p_sample_step(model=model, x_t=x_t, t=t, model_kwargs=model_kwargs)
        return pred_x0 if pred_x0 is not None else x_t

    @staticmethod
    def prototype_cosine_logits(x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        z = F.normalize(x, dim=-1)
        p = F.normalize(prototypes, dim=-1)
        logits = torch.einsum("bld,cd->blc", z, p)
        return logits

    @staticmethod
    def prototype_logits(x: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        logits = LabelGaussianDiffusion.prototype_cosine_logits(x=x, prototypes=prototypes)
        return logits / max(float(temperature), 1e-6)

    def training_losses(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        consistency_weight: float = 0.1,
        proto_weight: float = 0.2,
        proto_temperature: float = 0.1,
        proto_margin: float = 0.0,
        proto_pos_weight: float = 1.0,
        proto_neg_weight: float = 1.0,
        ranking_weight: float = 0.0,
        ranking_margin: float = 0.2,
        ranking_focal_gamma: float = 0.0,
        ranking_hard_mix: float = 1.0,
        ranking_loss_type: str = "hinge",
        ranking_softplus_temp: float = 0.1,
        hard_negative_ratio: float = 0.0,
        hard_negative_weight: float = 0.0,
        hard_negative_margin: float = 0.0,
        positive_margin_weight: float = 0.0,
        positive_margin_target: float = 0.3,
    ) -> DiffusionLosses:
        input_ids = batch["input_ids"]
        token_mask = batch["token_mask"]
        line_labels = batch["line_labels"]
        line_mask = batch["line_mask"].float()

        bsz = input_ids.size(0)
        t = torch.randint(0, self.num_timesteps, (bsz,), device=input_ids.device).long()

        x_start = model.get_label_embeddings(line_labels)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        pred_noise = model(
            noisy_label_emb=x_t,
            timesteps=t,
            input_ids=input_ids,
            token_mask=token_mask,
            line_mask=line_mask,
        )

        mse_per_pos = ((pred_noise - noise) ** 2).mean(dim=-1)
        mse = (mse_per_pos * line_mask).sum() / line_mask.sum().clamp(min=1.0)

        x0_pred = self.predict_x0_from_eps(x_t=x_t, t=t, eps=pred_noise)
        prototypes = model.label_embedding.weight
        raw_logits = self.prototype_cosine_logits(x0_pred, prototypes)
        logits = raw_logits / max(float(proto_temperature), 1e-6)

        if proto_margin > 0.0:
            flat_logits = logits.view(-1, logits.size(-1)).clone()
            flat_labels = line_labels.view(-1)
            row_idx = torch.arange(flat_labels.shape[0], device=flat_labels.device)
            flat_logits[row_idx, flat_labels] = flat_logits[row_idx, flat_labels] - float(proto_margin)
            logits = flat_logits.view_as(logits)

        proto_per_pos = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            line_labels.view(-1),
            reduction="none",
        ).view_as(line_labels)
        if proto_pos_weight != 1.0 or proto_neg_weight != 1.0:
            cls_w = torch.where(line_labels == 1, float(proto_pos_weight), float(proto_neg_weight)).float()
            proto_per_pos = proto_per_pos * cls_w
        proto = (proto_per_pos * line_mask).sum() / line_mask.sum().clamp(min=1.0)

        margin = raw_logits[..., 1] - raw_logits[..., 0]

        # Probe the hardest denoising setting (largest t) for auxiliary constraints,
        # avoiding trivial satisfaction on teacher-forced states.
        if ranking_weight > 0.0 or hard_negative_weight > 0.0:
            t_hard = torch.full_like(t, self.num_timesteps - 1)
            x_hard = torch.randn_like(x_start)
            pred_noise_hard = model(
                noisy_label_emb=x_hard,
                timesteps=t_hard,
                input_ids=input_ids,
                token_mask=token_mask,
                line_mask=line_mask,
            )
            x0_hard = self.predict_x0_from_eps(x_t=x_hard, t=t_hard, eps=pred_noise_hard)
            raw_logits_hard = self.prototype_cosine_logits(x0_hard, prototypes)
            margin_hard = raw_logits_hard[..., 1] - raw_logits_hard[..., 0]
            mix = float(max(0.0, min(1.0, ranking_hard_mix)))
            margin = mix * margin_hard + (1.0 - mix) * margin

        valid = line_mask > 0
        valid_y = line_labels[valid]
        valid_m = margin[valid]
        pos_m = valid_m[valid_y == 1]
        neg_m = valid_m[valid_y == 0]

        ranking = torch.zeros((), device=input_ids.device, dtype=mse.dtype)
        hard_negative = torch.zeros((), device=input_ids.device, dtype=mse.dtype)
        positive_margin = torch.zeros((), device=input_ids.device, dtype=mse.dtype)

        if pos_m.numel() > 0 and neg_m.numel() > 0 and (ranking_weight > 0.0 or hard_negative_weight > 0.0):
            selected_neg = neg_m
            if hard_negative_ratio > 0.0 and neg_m.numel() > 1:
                k = int(max(1, round(neg_m.numel() * float(hard_negative_ratio))))
                k = min(k, neg_m.numel())
                selected_neg = torch.topk(neg_m, k=k, largest=True).values

            if ranking_weight > 0.0:
                # Encourage positive margins to stay above hard negatives by a fixed margin.
                pair_gap = pos_m.unsqueeze(1) - selected_neg.unsqueeze(0)
                if ranking_loss_type == "softplus":
                    temp = max(1e-4, float(ranking_softplus_temp))
                    pair_violation = F.softplus((float(ranking_margin) - pair_gap) / temp) * temp
                else:
                    pair_violation = F.relu(float(ranking_margin) - pair_gap)
                if ranking_focal_gamma > 0.0:
                    hardness = (pair_violation.detach() + 1e-6).pow(float(ranking_focal_gamma))
                    ranking = (pair_violation * hardness).mean()
                else:
                    ranking = pair_violation.mean()

            if hard_negative_weight > 0.0:
                # Penalize negatives with overly high positive-class margin.
                hard_negative = F.relu(selected_neg + float(hard_negative_margin)).mean()

            if positive_margin_weight > 0.0:
                # Push positive samples to a safe margin above zero for better top-ranked recall.
                positive_margin = F.relu(float(positive_margin_target) - pos_m).mean()

        consistency = torch.zeros((), device=input_ids.device, dtype=mse.dtype)
        if consistency_weight > 0.0:
            x_prev_mean, _, _, _ = self.p_mean_variance(x_t=x_t, t=t, pred_eps=pred_noise)
            t_prev = torch.clamp(t - 1, min=0)
            pred_noise_prev = model(
                noisy_label_emb=x_prev_mean.detach(),
                timesteps=t_prev,
                input_ids=input_ids,
                token_mask=token_mask,
                line_mask=line_mask,
            )
            x0_pred_prev = self.predict_x0_from_eps(x_t=x_prev_mean.detach(), t=t_prev, eps=pred_noise_prev)
            cons_per_pos = ((x0_pred_prev - x0_pred.detach()) ** 2).mean(dim=-1)
            valid_t_mask = (t > 0).float().unsqueeze(-1)
            cons_mask = line_mask * valid_t_mask
            consistency = (cons_per_pos * cons_mask).sum() / cons_mask.sum().clamp(min=1.0)

        total = (
            mse
            + float(consistency_weight) * consistency
            + float(proto_weight) * proto
            + float(ranking_weight) * ranking
            + float(hard_negative_weight) * hard_negative
            + float(positive_margin_weight) * positive_margin
        )
        return DiffusionLosses(
            loss=total,
            mse=mse,
            consistency=consistency,
            proto=proto,
            ranking=ranking,
            hard_negative=hard_negative,
            positive_margin=positive_margin,
        )
