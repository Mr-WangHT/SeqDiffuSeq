from __future__ import annotations

from codediff.src.modeling.diffusion.gaussian_diffusion import LabelGaussianDiffusion
from codediff.src.modeling.predictor.transformer_model import CodeSeqDiffuModel


def create_model_and_diffusion(
	*,
	vocab_size: int,
	pad_id: int = 0,
	token_emb_dim: int = 128,
	line_hidden_dim: int = 128,
	cond_dim: int = 256,
	label_emb_dim: int = 256,
	num_cond_layers: int = 2,
	num_denoise_layers: int = 2,
	nhead: int = 8,
	dropout: float = 0.1,
	diffusion_steps: int = 1000,
	line_encoder_type: str = "lstm",
	codebert_local_path: str | None = None,
	freeze_codebert: bool = False,
):
	model = CodeSeqDiffuModel(
		vocab_size=vocab_size,
		pad_id=pad_id,
		token_emb_dim=token_emb_dim,
		line_hidden_dim=line_hidden_dim,
		cond_dim=cond_dim,
		label_emb_dim=label_emb_dim,
		num_cond_layers=num_cond_layers,
		num_denoise_layers=num_denoise_layers,
		nhead=nhead,
		dropout=dropout,
		line_encoder_type=line_encoder_type,
		codebert_local_path=codebert_local_path,
		freeze_codebert=freeze_codebert,
	)
	diffusion = LabelGaussianDiffusion(num_timesteps=diffusion_steps)
	return model, diffusion

