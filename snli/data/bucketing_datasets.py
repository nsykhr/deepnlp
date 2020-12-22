import math
from tqdm import tqdm
from overrides import overrides

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data.utils import cached_tokenize

from typing import Sequence, Tuple


class SNLICrossEncoderBucketingDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 premises: Sequence[str],
                 hypotheses: Sequence[str],
                 targets: Sequence[int],
                 max_length: int = 512,
                 batch_size: int = 32):
        self.tokenizer = tokenizer
        self.cache = {}
        self.max_length = max_length
        self.data = self.prepare_batches(premises, hypotheses, targets, batch_size)

    def __len__(self):
        return len(self.data)

    def prepare_batches(self, premises, hypotheses, targets, batch_size):
        tokenized = [cached_tokenize((p, h), self.tokenizer, self.cache)
                     for p, h in tqdm(list(zip(premises, hypotheses)))]

        data = sorted(zip(tokenized, targets), key=lambda x: len(x[0]))
        batched_data = []

        for i_batch in range(math.ceil(len(data) / batch_size)):
            batched_data.append((
                [x[0] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]],
                [x[1] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]]
            ))

        return batched_data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.data[index]
        batch_x, batch_y = [], []

        for sequence, target in zip(*batch):
            batch_x.append(sequence)
            batch_y.append(target)

        encoded = self.tokenizer(batch_x, padding='max_length', return_tensors='pt',
                                 return_attention_mask=True, is_split_into_words=True)
        batch_x, attention_mask = encoded.input_ids, encoded.attention_mask
        batch_y = torch.tensor(batch_y).long()

        return batch_x, attention_mask, batch_y


class SNLIBiEncoderBucketingDataset(SNLICrossEncoderBucketingDataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 premises: Sequence[str],
                 hypotheses: Sequence[str],
                 targets: Sequence[int],
                 max_length: int = 512,
                 batch_size: int = 32):
        super().__init__(tokenizer, premises, hypotheses, targets, max_length=max_length, batch_size=batch_size)

    @overrides
    def prepare_batches(self, premises, hypotheses, targets, batch_size):
        tokenized = [cached_tokenize(s, self.tokenizer, self.cache)
                     for s in tqdm(list(premises) + list(hypotheses))]
        premises_tokenized, hypotheses_tokenized = tokenized[:len(premises)], tokenized[-len(hypotheses):]

        data = sorted(zip(premises_tokenized, hypotheses_tokenized, targets), key=lambda x: len(x[0]))
        batched_data = []

        for i_batch in range(math.ceil(len(data) / batch_size)):
            batched_data.append((
                [x[0] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]],
                [x[1] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]],
                [x[2] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]]
            ))

        return batched_data

    @overrides
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.data[index]

        batch_p, batch_h, batch_y = [], [], []

        for premise, hypothesis, target in zip(*batch):
            batch_p.append(premise)
            batch_h.append(hypothesis)
            batch_y.append(target)

        premises = self.tokenizer(batch_p, padding=True, truncation=True, return_tensors='pt',
                                  return_attention_mask=True, is_split_into_words=True)
        batch_p, attention_mask_p = premises.input_ids, premises.attention_mask

        hypotheses = self.tokenizer(batch_h, padding=True, truncation=True, return_tensors='pt',
                                    return_attention_mask=True, is_split_into_words=True)
        batch_h, attention_mask_h = hypotheses.input_ids, hypotheses.attention_mask

        batch_y = torch.tensor(batch_y).long()

        return batch_p, batch_h, attention_mask_p, attention_mask_h, batch_y
