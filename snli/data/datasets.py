from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data.utils import cached_encode

from overrides import overrides
from typing import Sequence, Tuple, Union


class SNLICrossEncoderDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 premises: Sequence[str],
                 hypotheses: Sequence[str],
                 targets: Sequence[int],
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.premises = premises
        self.hypotheses = hypotheses
        self.targets = targets
        self.cache = {}
        self.max_length = max_length

    def __len__(self):
        return len(self.targets)

    def prepare_sequence(self, sequence: Union[Sequence[str], Tuple[Sequence[str]]]) -> Sequence[int]:
        indices = cached_encode(sequence, self.tokenizer, self.cache)
        indices = indices[:self.max_length]

        return indices

    def __getitem__(self, index: int) -> Tuple[Sequence[int], int]:
        p, h, y = self.premises[index], self.hypotheses[index], self.targets[index]
        x = self.prepare_sequence((p, h))

        return x, y


class SNLIBiEncoderDataset(SNLICrossEncoderDataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 premises: Sequence[str],
                 hypotheses: Sequence[str],
                 targets: Sequence[int],
                 max_length: int = 512):
        super().__init__(tokenizer, premises, hypotheses, targets, max_length=max_length)

    @overrides
    def __getitem__(self, index: int) -> Tuple[Sequence[int], Sequence[int], int]:
        p, h, y = self.premises[index], self.hypotheses[index], self.targets[index]
        p, h = tuple(map(self.prepare_sequence, [p, h]))

        return p, h, y
