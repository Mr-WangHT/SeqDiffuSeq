from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / max(half, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def line_position_embedding(num_lines: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(num_lines, dtype=torch.float32, device=device).unsqueeze(1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / max(half, 1))
    args = pos * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((num_lines, 1), device=device)], dim=1)
    return emb


class CodeLineBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, token_emb_dim: int, line_hidden_dim: int, pad_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=token_emb_dim,
            hidden_size=line_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        lengths = token_mask.long().sum(dim=1).clamp(min=1)
        embedded = self.dropout(self.token_embedding(token_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return torch.cat([h_n[0], h_n[1]], dim=-1)


class FileConditionEncoder(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, num_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, cond_dim)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        layer = nn.TransformerEncoderLayer(
            d_model=cond_dim,
            nhead=nhead,
            dim_feedforward=cond_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, line_embeddings: torch.Tensor, line_mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(line_embeddings)
        pos = line_position_embedding(x.size(1), x.size(2), x.device).unsqueeze(0)
        x = x + self.pos_scale * pos
        key_padding_mask = line_mask == 0
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class CodeSeqDiffuProtoModel(nn.Module):
    """Minimal model for prototype-only diffusion training (no linear classifier head)."""

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
    ) -> None:
        super().__init__()
        self.label_emb_dim = label_emb_dim

        self.line_encoder = CodeLineBiLSTMEncoder(
            vocab_size=vocab_size,
            token_emb_dim=token_emb_dim,
            line_hidden_dim=line_hidden_dim,
            pad_id=pad_id,
            dropout=dropout,
        )
        self.cond_encoder = FileConditionEncoder(
            in_dim=line_hidden_dim * 2,
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

    def encode_lines(self, input_ids: torch.Tensor, token_mask: torch.Tensor, line_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_lines, num_tokens = input_ids.shape
        flat_ids = input_ids.view(bsz * num_lines, num_tokens)
        flat_mask = token_mask.view(bsz * num_lines, num_tokens)

        line_repr = self.line_encoder(flat_ids, flat_mask)
        line_repr = line_repr.view(bsz, num_lines, -1)
        return line_repr * line_mask.unsqueeze(-1).float()

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
        cond = self.build_condition(input_ids=input_ids, token_mask=token_mask, line_mask=line_mask)

        t_emb = timestep_embedding(timesteps, self.label_emb_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        x = noisy_label_emb + t_emb

        key_padding_mask = line_mask == 0
        pred_hidden = self.denoise_decoder(
            tgt=x,
            memory=cond,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )
        return self.noise_head(pred_hidden)
