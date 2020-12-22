import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_utils import PreTrainedModel

from typing import Callable


class NLICrossEncoder(nn.Module):
    def __init__(self,
                 encoder: PreTrainedModel,
                 n_classes: int,
                 get_cls_encoding_fn: Callable,
                 num_linear_layers: int = 2,
                 hidden_size: int = 512,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder
        if num_linear_layers > 0:
            self.fc_hidden = nn.ModuleList([nn.Linear(encoder.config.hidden_size, hidden_size), nn.ReLU()])
            if dropout_rate > 0:
                self.fc_hidden.append(nn.Dropout(dropout_rate))
            for _ in range(1, num_linear_layers):
                self.fc_hidden.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
                if dropout_rate > 0:
                    self.fc_hidden.append(nn.Dropout(dropout_rate))
        hidden_size = hidden_size if num_linear_layers > 0 else encoder.config.hidden_size
        self.fc_output = nn.Linear(hidden_size, n_classes)
        self.get_cls_encoding = get_cls_encoding_fn

    def forward(self,
                sequence: torch.Tensor,
                attention_mask: torch.Tensor):
        sequence = self.get_cls_encoding(self.encoder(sequence, attention_mask=attention_mask))
        if hasattr(self, 'fc_hidden'):
            for layer in self.fc_hidden:
                sequence = layer(sequence)
        logits = self.fc_output(sequence)

        return logits

    def predict(self,
                sequence: torch.Tensor,
                attention_mask: torch.Tensor):
        self.eval()
        logits = self.forward(sequence, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return preds


class NLIBiEncoder(nn.Module):
    def __init__(self,
                 encoder: PreTrainedModel,
                 n_classes: int,
                 get_cls_encoding_fn: Callable,
                 num_linear_layers: int = 2,
                 hidden_size: int = 512,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder
        if num_linear_layers > 0:
            self.fc_hidden = nn.ModuleList([nn.Linear(encoder.config.hidden_size * 2, hidden_size), nn.ReLU()])
            if dropout_rate > 0:
                self.fc_hidden.append(nn.Dropout(dropout_rate))
            for _ in range(1, num_linear_layers):
                self.fc_hidden.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
                if dropout_rate > 0:
                    self.fc_hidden.append(nn.Dropout(dropout_rate))
        hidden_size = hidden_size if num_linear_layers > 0 else encoder.config.hidden_size * 2
        self.fc_output = nn.Linear(hidden_size, n_classes)
        self.get_cls_encoding = get_cls_encoding_fn

    def forward(self,
                premises: torch.Tensor,
                hypotheses: torch.Tensor,
                attention_mask_p: torch.Tensor,
                attention_mask_h: torch.Tensor):
        premises = self.get_cls_encoding(self.encoder(premises, attention_mask=attention_mask_p))
        hypotheses = self.get_cls_encoding(self.encoder(hypotheses, attention_mask=attention_mask_h))
        sequence = torch.cat((premises, hypotheses), dim=1)
        if hasattr(self, 'fc_hidden'):
            for layer in self.fc_hidden:
                sequence = layer(sequence)
        logits = self.fc_output(sequence)

        return logits

    def predict(self,
                premises: torch.Tensor,
                hypotheses: torch.Tensor,
                attention_mask_p: torch.Tensor,
                attention_mask_h: torch.Tensor):
        self.eval()
        logits = self.forward(premises, hypotheses, attention_mask_p, attention_mask_h)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return preds
