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
                 target: Sequence[int],
                 max_length: int = 512,
                 batch_size: int = 32):
        self.tokenizer = tokenizer
        self.cache = {}
        self.max_length = max_length
        self.data = self.prepare_batches(premises, hypotheses, target, batch_size)

    def __len__(self):
        return len(self.data)

    def prepare_batches(self, premises, hypotheses, target, batch_size):
        tokenized = [cached_tokenize((p, h), self.tokenizer, self.cache)
                     for p, h in tqdm(list(zip(premises, hypotheses)))]

        data = sorted(zip(tokenized, target), key=lambda x: len(x[0]))
        batched_data = []

        for i_batch in range(math.ceil(len(data) / batch_size)):
            batched_data.append((
                [x[0] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]],
                [x[1] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]]
            ))

        return batched_data

    def prepare_sequence(self, sequence: Sequence[str], max_length: int) -> Sequence[int]:
        sequence = sequence[:max_length]
        sequence += [self.tokenizer.pad_token] * (max_length - len(sequence))
        indices = self.tokenizer.convert_tokens_to_ids(sequence)

        return indices

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.data[index]
        max_length = min([self.max_length, max([len(sequence) for sequence in batch[0]])])

        batch_x, batch_y = [], []

        for sequence, target in zip(*batch):
            batch_x.append(self.prepare_sequence(sequence, max_length))
            batch_y.append(target)

        batch_x = torch.tensor(batch_x).long()
        batch_y = torch.tensor(batch_y).long()

        return batch_x, batch_y


class SNLIBiEncoderBucketingDataset(SNLICrossEncoderBucketingDataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 premises: Sequence[str],
                 hypotheses: Sequence[str],
                 target: Sequence[int],
                 max_length: int = 512,
                 batch_size: int = 32):
        self.tokenizer = tokenizer
        self.cache = {}
        self.max_length = max_length
        self.data = self.prepare_batches(premises, hypotheses, target, batch_size)

    @overrides
    def prepare_batches(self, premises, hypotheses, target, batch_size):
        tokenized = [cached_tokenize(s, self.tokenizer, self.cache)
                     for s in tqdm(list(premises) + list(hypotheses))]
        premises_tokenized, hypotheses_tokenized = tokenized[:len(premises)], tokenized[-len(hypotheses):]

        data = sorted(zip(premises_tokenized, hypotheses_tokenized, target), key=lambda x: len(x[0]))
        batched_data = []

        for i_batch in range(math.ceil(len(data) / batch_size)):
            batched_data.append((
                [x[0] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]],
                [x[1] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]],
                [x[2] for x in data[i_batch * batch_size: (i_batch+1) * batch_size]]
            ))

        return batched_data

    @overrides
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.data[index]

        max_p_length = min([self.max_length, max([len(sequence) for sequence in batch[0]])])
        max_h_length = min([self.max_length, max([len(sequence) for sequence in batch[1]])])

        batch_p, batch_h, batch_y = [], [], []

        for premise, hypothesis, target in zip(*batch):
            batch_p.append(self.prepare_sequence(premise, max_p_length))
            batch_h.append(self.prepare_sequence(hypothesis, max_h_length))
            batch_y.append(target)

        batch_p = torch.tensor(batch_p).long()
        batch_h = torch.tensor(batch_h).long()
        batch_y = torch.tensor(batch_y).long()

        return batch_p, batch_h, batch_y
