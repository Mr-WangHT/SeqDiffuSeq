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
	cls: torch.Tensor
	consistency: torch.Tensor


class LabelGaussianDiffusion:
	def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
		self.num_timesteps = num_timesteps
		self.betas = linear_beta_schedule(num_timesteps, beta_start=beta_start, beta_end=beta_end)
		self.alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
		self.alphas_cumprod_prev = torch.cat(
			[torch.ones(1, dtype=self.alphas_cumprod.dtype), self.alphas_cumprod[:-1]], dim=0
		)
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
		self.posterior_log_variance = torch.log(self.posterior_variance)

		self.posterior_mean_coef1 = (
			self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_mean_coef2 = (
			(1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
		)

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

	def training_losses(
		self,
		model,
		batch: Dict[str, torch.Tensor],
		cls_weight: float = 1.0,
		consistency_weight: float = 0.0,
		use_cls_head: bool = True,
	) -> DiffusionLosses:
		"""
		batch expects:
		  - input_ids: [B, L, T]
		  - token_mask: [B, L, T]
		  - line_labels: [B, L] (0/1)
		  - line_mask: [B, L] (1 valid)
		"""
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
		cls = torch.zeros((), device=input_ids.device, dtype=mse.dtype)
		if use_cls_head:
			logits = model.classify_from_embeddings(x0_pred)
			cls_loss_per_pos = F.cross_entropy(
				logits.view(-1, logits.size(-1)),
				line_labels.view(-1),
				reduction="none",
			).view_as(line_labels)
			cls = (cls_loss_per_pos * line_mask).sum() / line_mask.sum().clamp(min=1.0)

		consistency = torch.zeros((), device=input_ids.device, dtype=mse.dtype)
		if consistency_weight > 0.0:
			# One deterministic reverse step (posterior mean only) to align adjacent-step x0 predictions.
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

		effective_cls_weight = cls_weight if use_cls_head else 0.0
		total = mse + effective_cls_weight * cls + consistency_weight * consistency
		return DiffusionLosses(loss=total, mse=mse, cls=cls, consistency=consistency)

