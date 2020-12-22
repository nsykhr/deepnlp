from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from typing import Sequence, Tuple


class SNLIDataset(Dataset):
    def __init__(self,
                 premises: Sequence[str],
                 hypotheses: Sequence[str],
                 targets: Sequence[int],
                 max_length: int = 512):
        self.premises = premises
        self.hypotheses = hypotheses
        self.targets = targets
        self.cache = {}
        self.max_length = max_length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Tuple[str, str], int]:
        p, h, y = self.premises[index], self.hypotheses[index], self.targets[index]

        return (p, h), y
