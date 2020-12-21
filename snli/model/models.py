from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_utils import PreTrainedModel

from typing import Callable


class NLIBiEncoder(nn.Module):
    def __init__(self,
                 encoder: PreTrainedModel,
                 n_classes: int,
                 get_cls_encoding_fn: Callable,
                 num_linear_layers: int = 2,
                 hidden_size: int = 512,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.premise_encoder = deepcopy(encoder)
        self.hypothesis_encoder = deepcopy(encoder)
        if num_linear_layers > 0:
            self.fc_hidden = nn.ModuleList([nn.Linear(encoder.config.hidden_size * 2, hidden_size)])
            if dropout_rate > 0:
                self.fc_hidden.append(nn.Dropout(dropout_rate))
            for _ in range(1, num_linear_layers):
                self.fc_hidden.append(nn.Linear(hidden_size, hidden_size))
                if dropout_rate > 0:
                    self.fc_hidden.append(nn.Dropout(dropout_rate))
        hidden_size = hidden_size if num_linear_layers > 0 else encoder.config.hidden_size * 2
        self.fc_output = nn.Linear(hidden_size, n_classes)
        self.get_cls_encoding = get_cls_encoding_fn

    def forward(self,
                premise: torch.Tensor,
                hypothesis: torch.Tensor):
        premise = self.get_cls_encoding(self.premise_encoder(premise))
        hypothesis = self.get_cls_encoding(self.hypothesis_encoder(hypothesis))
        sequence = torch.cat((premise, hypothesis), dim=1)
        if hasattr(self, 'fc_hidden'):
            for layer in self.fc_hidden:
                sequence = layer(sequence)
        logits = self.fc_output(sequence)

        return logits

    def predict(self,
                premise: torch.Tensor,
                hypothesis: torch.Tensor):
        self.eval()
        logits = self.forward(premise, hypothesis)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return preds


class NLICrossEncoder(nn.Module):
    def __init__(self,
                 encoder: PreTrainedModel,
                 n_classes: int,
                 num_linear_layers: int = 2,
                 hidden_size: int = 512,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder
        if num_linear_layers > 0:
            self.fc_hidden = nn.ModuleList([nn.Linear(encoder.config.hidden_size, hidden_size)])
            if dropout_rate > 0:
                self.fc_hidden.append(nn.Dropout(dropout_rate))
            for _ in range(1, num_linear_layers):
                self.fc_hidden.append(nn.Linear(hidden_size, hidden_size))
                if dropout_rate > 0:
                    self.fc_hidden.append(nn.Dropout(dropout_rate))
        hidden_size = hidden_size if num_linear_layers > 0 else encoder.config.hidden_size
        self.fc_output = nn.Linear(hidden_size, n_classes)

    def forward(self,
                sequence: torch.Tensor):
        sequence = self.get_cls_encoding(self.encoder(sequence))
        if hasattr(self, 'fc_hidden'):
            for layer in self.fc_hidden:
                sequence = layer(sequence)
        logits = self.fc_output(sequence)

        return logits

    def predict(self,
                sequence: torch.Tensor):
        self.eval()
        logits = self.forward(sequence)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return preds
