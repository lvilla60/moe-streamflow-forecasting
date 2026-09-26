"""LSTM, GRU, and attentive sequence-to-sequence forecasting experts."""

import random

import torch
from torch import nn

from .common import validate_sequence


class RecurrentDirectExpert(nn.Module):
    """Direct multi-step forecast from the final recurrent hidden state."""

    recurrent_class = None

    def __init__(self, input_size=12, hidden_size=32, num_layers=1, dropout=0.0,
                 forecast_horizon=48):
        super().__init__()
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.recurrent = self.recurrent_class(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_horizon),
        )

    def forward(self, x, **_):
        validate_sequence(x, self.input_size)
        _, state = self.recurrent(x)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.output(hidden[-1])


class LSTMExpert(RecurrentDirectExpert):
    recurrent_class = nn.LSTM


class GRUExpert(RecurrentDirectExpert):
    recurrent_class = nn.GRU


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, attention_size):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_size, attention_size, bias=False)
        self.decoder_projection = nn.Linear(decoder_size, attention_size, bias=False)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        energy = torch.tanh(
            self.encoder_projection(encoder_outputs)
            + self.decoder_projection(decoder_hidden).unsqueeze(1)
        )
        weights = torch.softmax(self.score(energy).squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Seq2SeqAttentionExpert(nn.Module):
    """Autoregressive LSTM decoder with Bahdanau-style attention."""

    requires_decoder_start = True

    def __init__(self, input_size=12, hidden_size=32, num_layers=1, dropout=0.0,
                 forecast_horizon=48, attention_size=None):
        super().__init__()
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=recurrent_dropout,
        )
        self.attention = AdditiveAttention(
            hidden_size, hidden_size, attention_size or hidden_size
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTMCell(hidden_size + 1, hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, target=None, teacher_forcing_ratio=0.0,
                decoder_start=None, return_attention=False):
        validate_sequence(x, self.input_size)
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")
        if decoder_start is None:
            raise ValueError("decoder_start in y-normalized units is required")
        if decoder_start.ndim == 2 and decoder_start.size(1) == 1:
            decoder_start = decoder_start.squeeze(1)
        if decoder_start.ndim != 1 or decoder_start.size(0) != x.size(0):
            raise ValueError("decoder_start must have shape [batch]")
        if target is not None and target.shape != (x.size(0), self.forecast_horizon):
            raise ValueError("target must have shape [batch, forecast_horizon]")

        encoder_outputs, (hidden, cell) = self.encoder(x)
        encoder_outputs = self.dropout(encoder_outputs)
        decoder_hidden = hidden[-1]
        decoder_cell = cell[-1]
        previous = decoder_start
        predictions = []
        attention_history = []

        for step in range(self.forecast_horizon):
            context, weights = self.attention(encoder_outputs, decoder_hidden)
            decoder_hidden, decoder_cell = self.decoder(
                torch.cat((previous.unsqueeze(-1), context), dim=-1),
                (decoder_hidden, decoder_cell),
            )
            prediction = self.output(
                self.dropout(torch.cat((decoder_hidden, context), dim=-1))
            ).squeeze(-1)
            predictions.append(prediction)
            attention_history.append(weights)
            use_teacher = (
                self.training
                and target is not None
                and teacher_forcing_ratio > 0
                and random.random() < teacher_forcing_ratio
            )
            previous = target[:, step] if use_teacher else prediction

        forecast = torch.stack(predictions, dim=1)
        if return_attention:
            return forecast, torch.stack(attention_history, dim=1)
        return forecast


def create_expert(name, **kwargs):
    name = name.lower()
    experts = {
        "lstm": LSTMExpert,
        "gru": GRUExpert,
        "seq2seq_attention": Seq2SeqAttentionExpert,
    }
    if name == "informer":
        from .informer import InformerExpert
        return InformerExpert(**kwargs)
    if name not in experts:
        raise ValueError(f"unknown expert {name!r}; expected {sorted((*experts, 'informer'))}")
    return experts[name](**kwargs)
