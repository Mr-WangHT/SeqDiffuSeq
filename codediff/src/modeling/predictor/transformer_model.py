from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
	half = dim // 2
	device = timesteps.device
	freqs = torch.exp(
		-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / max(half, 1)
	)
	args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
	emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
	if dim % 2 == 1:
		emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
	return emb


class CodeLineBiLSTMEncoder(nn.Module):
	def __init__(self, vocab_size: int, token_emb_dim: int, line_hidden_dim: int, pad_id: int = 0, dropout: float = 0.1):
		super().__init__()
		self.pad_id = pad_id
		self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_id)
		self.lstm = nn.LSTM(
			input_size=token_emb_dim,
			hidden_size=line_hidden_dim,
			num_layers=1,
			batch_first=True,
			bidirectional=True,
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
		# token_ids: [N, T], token_mask: [N, T]
		lengths = token_mask.long().sum(dim=1).clamp(min=1)
		embedded = self.dropout(self.token_embedding(token_ids))
		packed = nn.utils.rnn.pack_padded_sequence(
			embedded,
			lengths.cpu(),
			batch_first=True,
			enforce_sorted=False,
		)
		_, (h_n, _) = self.lstm(packed)
		# h_n: [2, N, H]
		line_repr = torch.cat([h_n[0], h_n[1]], dim=-1)  # [N, 2H]
		return line_repr


class CodeLineCodeBERTEncoder(nn.Module):
	def __init__(self, model_path: str, dropout: float = 0.1, freeze_backbone: bool = False):
		super().__init__()
		try:
			self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
		except Exception:
			# Fallback for old torch versions that do not support transformers' weights_only torch.load call.
			config = AutoConfig.from_pretrained(model_path, local_files_only=True)
			self.backbone = AutoModel.from_config(config)
			weight_path = Path(model_path) / "pytorch_model.bin"
			state = torch.load(str(weight_path), map_location="cpu")
			if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
				state = state["state_dict"]
			self.backbone.load_state_dict(state, strict=False)
		self.output_dim = int(self.backbone.config.hidden_size)
		self.dropout = nn.Dropout(dropout)
		if freeze_backbone:
			for p in self.backbone.parameters():
				p.requires_grad = False

	def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
		# token_ids: [N, T], token_mask: [N, T]
		attn_mask = token_mask.long()
		outputs = self.backbone(input_ids=token_ids.long(), attention_mask=attn_mask)
		hidden = outputs.last_hidden_state  # [N, T, H]
		mask = attn_mask.unsqueeze(-1).float()
		summed = (hidden * mask).sum(dim=1)
		denom = mask.sum(dim=1).clamp(min=1.0)
		return self.dropout(summed / denom)


class FileConditionEncoder(nn.Module):
	def __init__(self, in_dim: int, cond_dim: int, num_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
		super().__init__()
		self.in_proj = nn.Linear(in_dim, cond_dim)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=cond_dim,
			nhead=nhead,
			dim_feedforward=cond_dim * 4,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

	def forward(self, line_embeddings: torch.Tensor, line_mask: torch.Tensor) -> torch.Tensor:
		# line_embeddings: [B, L, D], line_mask: [B, L] (1 valid)
		x = self.in_proj(line_embeddings)
		key_padding_mask = line_mask == 0
		return self.encoder(x, src_key_padding_mask=key_padding_mask)


class CodeSeqDiffuModel(nn.Module):
	"""
	Pipeline:
	  code lines -> BiLSTM line embedding -> file sequence encoder (condition)
	  noisy label embedding + condition -> transformer decoder -> predicted noise
	"""

	def __init__(
		self,
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
		line_encoder_type: str = "lstm",
		codebert_local_path: Optional[str] = None,
		freeze_codebert: bool = False,
	):
		super().__init__()
		self.label_emb_dim = label_emb_dim
		self.line_encoder_type = line_encoder_type.lower()

		if self.line_encoder_type == "lstm":
			self.line_encoder = CodeLineBiLSTMEncoder(
				vocab_size=vocab_size,
				token_emb_dim=token_emb_dim,
				line_hidden_dim=line_hidden_dim,
				pad_id=pad_id,
				dropout=dropout,
			)
			line_repr_dim = line_hidden_dim * 2
		elif self.line_encoder_type == "codebert":
			if not codebert_local_path:
				raise ValueError("codebert_local_path is required when line_encoder_type='codebert'")
			self.line_encoder = CodeLineCodeBERTEncoder(
				model_path=codebert_local_path,
				dropout=dropout,
				freeze_backbone=freeze_codebert,
			)
			line_repr_dim = self.line_encoder.output_dim
		else:
			raise ValueError(f"Unsupported line_encoder_type: {line_encoder_type}")

		self.cond_encoder = FileConditionEncoder(
			in_dim=line_repr_dim,
			cond_dim=cond_dim,
			num_layers=num_cond_layers,
			nhead=nhead,
			dropout=dropout,
		)

		self.cond_to_label_dim = nn.Linear(cond_dim, label_emb_dim)
		self.label_embedding = nn.Embedding(2, label_emb_dim)
		self.time_mlp = nn.Sequential(
			nn.Linear(label_emb_dim, label_emb_dim),
			nn.SiLU(),
			nn.Linear(label_emb_dim, label_emb_dim),
		)

		decoder_layer = nn.TransformerDecoderLayer(
			d_model=label_emb_dim,
			nhead=nhead,
			dim_feedforward=label_emb_dim * 4,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.denoise_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_denoise_layers)

		self.noise_head = nn.Linear(label_emb_dim, label_emb_dim)
		self.classifier = nn.Linear(label_emb_dim, 2)

	def encode_lines(self, input_ids: torch.Tensor, token_mask: torch.Tensor, line_mask: torch.Tensor) -> torch.Tensor:
		# input_ids/token_mask: [B, L, T], line_mask: [B, L]
		bsz, num_lines, num_tokens = input_ids.shape
		flat_ids = input_ids.view(bsz * num_lines, num_tokens)
		flat_mask = token_mask.view(bsz * num_lines, num_tokens)

		line_repr = self.line_encoder(flat_ids, flat_mask)  # [B*L, 2H]
		line_repr = line_repr.view(bsz, num_lines, -1)
		line_repr = line_repr * line_mask.unsqueeze(-1).float()
		return line_repr

	def build_condition(self, input_ids: torch.Tensor, token_mask: torch.Tensor, line_mask: torch.Tensor) -> torch.Tensor:
		line_repr = self.encode_lines(input_ids, token_mask, line_mask)
		cond = self.cond_encoder(line_repr, line_mask)
		return self.cond_to_label_dim(cond)

	def get_label_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
		return self.label_embedding(labels.long())

	def forward(
		self,
		noisy_label_emb: torch.Tensor,
		timesteps: torch.Tensor,
		input_ids: torch.Tensor,
		token_mask: torch.Tensor,
		line_mask: torch.Tensor,
	) -> torch.Tensor:
		# noisy_label_emb: [B, L, D]
		cond = self.build_condition(input_ids=input_ids, token_mask=token_mask, line_mask=line_mask)

		t_emb = timestep_embedding(timesteps, self.label_emb_dim)
		t_emb = self.time_mlp(t_emb).unsqueeze(1)
		x = noisy_label_emb + t_emb

		memory_key_padding_mask = line_mask == 0
		pred_hidden = self.denoise_decoder(
			tgt=x,
			memory=cond,
			tgt_key_padding_mask=memory_key_padding_mask,
			memory_key_padding_mask=memory_key_padding_mask,
		)
		return self.noise_head(pred_hidden)

	def classify_from_embeddings(self, label_embeddings: torch.Tensor) -> torch.Tensor:
		return self.classifier(label_embeddings)

	def set_codebert_trainable(self, trainable: bool) -> None:
		if self.line_encoder_type != "codebert":
			return
		for p in self.line_encoder.backbone.parameters():
			p.requires_grad = trainable

	def is_codebert_trainable(self) -> bool:
		if self.line_encoder_type != "codebert":
			return False
		return any(p.requires_grad for p in self.line_encoder.backbone.parameters())

