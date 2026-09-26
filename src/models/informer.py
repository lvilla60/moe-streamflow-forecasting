"""Compact Informer adaptation with simplified ProbSparse attention."""

import math

import torch
from torch import nn

from .common import SinusoidalPositionalEncoding, validate_sequence


class ProbSparseAttention(nn.Module):
    """Attend fully only for queries with the largest sampled sparsity scores."""

    def __init__(self, d_model, n_heads, dropout=0.0, factor=5, sparse=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.factor = factor
        self.sparse = sparse
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)

    def forward(self, queries, keys, values, attn_mask=None):
        batch, query_length, _ = queries.shape
        key_length = keys.size(1)
        q = self._heads(self.query(queries))
        k = self._heads(self.key(keys))
        v = self._heads(self.value(values))

        # Global top-query selection depends on later queries/keys. Use full
        # attention for masked self-attention and decoder cross-attention so
        # each decoder position depends only on its prefix and encoder memory.
        if attn_mask is not None or not self.sparse:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
            if attn_mask is not None:
                if attn_mask.shape != (query_length, key_length):
                    raise ValueError("attention mask has incompatible shape")
                mask = attn_mask.to(device=queries.device, dtype=torch.bool)
                if mask.all(dim=-1).any():
                    raise ValueError("each query must have at least one unmasked key")
                scores = scores.masked_fill(mask, float("-inf"))
            context = torch.matmul(self.dropout(torch.softmax(scores, dim=-1)), v)
            context = context.transpose(1, 2).contiguous().view(batch, query_length, self.d_model)
            return self.output(context)

        sample_count = min(
            key_length, max(1, self.factor * math.ceil(math.log(key_length + 1)))
        )
        top_count = min(
            query_length, max(1, self.factor * math.ceil(math.log(query_length + 1)))
        )
        sample_indices = torch.linspace(
            0, key_length - 1, sample_count, device=queries.device
        ).long()
        sampled_k = k.index_select(2, sample_indices)
        sampled_scores = torch.matmul(q, sampled_k.transpose(-2, -1))
        sparsity = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)
        top_indices = sparsity.topk(top_count, dim=-1, sorted=False).indices
        selected_q = torch.gather(
            q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_size)
        )
        scores = torch.matmul(selected_q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        attention = self.dropout(torch.softmax(scores, dim=-1))
        selected_context = torch.matmul(attention, v)
        context = v.mean(dim=2, keepdim=True).expand(-1, -1, query_length, -1).clone()
        context = context.scatter(
            2, top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_size),
            selected_context,
        )
        context = context.transpose(1, 2).contiguous().view(batch, query_length, self.d_model)
        return self.output(context)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, factor):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout, factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x, x, x)))
        return self.norm2(x + self.feed_forward(x))


class DistillationLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.convolution = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.activation(self.convolution(x.transpose(1, 2)))).transpose(1, 2)


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, factor):
        super().__init__()
        self.self_attention = ProbSparseAttention(d_model, n_heads, dropout, factor)
        self.cross_attention = ProbSparseAttention(d_model, n_heads, dropout, factor, sparse=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, causal_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, memory)))
        return self.norm3(x + self.feed_forward(x))


class InformerExpert(nn.Module):
    """Informer adapted to relative positions and known historical inputs only."""

    def __init__(self, input_size=12, forecast_horizon=48, label_len=48,
                 d_model=64, n_heads=4, encoder_layers=2, decoder_layers=1,
                 d_ff=128, dropout=0.1, factor=5, max_length=1024):
        super().__init__()
        if forecast_horizon <= 0 or label_len <= 0:
            raise ValueError("forecast_horizon and label_len must be positive")
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.label_len = label_len
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_length)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(encoder_layers)
        ])
        self.distillation_layers = nn.ModuleList([
            DistillationLayer(d_model) for _ in range(max(0, encoder_layers - 1))
        ])
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(decoder_layers)
        ])
        self.output = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(length, device=None):
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, x, **_):
        validate_sequence(x, self.input_size)
        if x.size(1) < self.label_len:
            raise ValueError("input sequence must be at least label_len timesteps")
        memory = self.dropout(self.position(self.input_projection(x)))
        for index, layer in enumerate(self.encoder_layers):
            memory = layer(memory)
            if index < len(self.distillation_layers):
                memory = self.distillation_layers[index](memory)

        future = x.new_zeros(x.size(0), self.forecast_horizon, self.input_size)
        decoder_input = torch.cat((x[:, -self.label_len:], future), dim=1)
        decoder = self.dropout(self.position(self.input_projection(decoder_input)))
        mask = self.causal_mask(decoder.size(1), decoder.device)
        for layer in self.decoder_layers:
            decoder = layer(decoder, memory, mask)
        return self.output(decoder[:, -self.forecast_horizon:]).squeeze(-1)
