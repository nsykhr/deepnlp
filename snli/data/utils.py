import os
import json
import torch
import pandas as pd


def get_data(path):
    def list_to_df(lst):
        return pd.DataFrame({
            'sentence1': [x['sentence1'] for x in lst],
            'sentence2': [x['sentence2'] for x in lst],
            'target': [x['gold_label'] for x in lst]
        })

    train, val, test = [], [], []
    for file in os.listdir(path):
        if not file.endswith('.jsonl'):
            continue

        if 'train' in file:
            data = train
        elif 'dev' in file:
            data = val
        elif 'test' in file:
            data = test
        else:
            assert False

        with open(os.path.join(path, file)) as f:
            for line in f.readlines():
                data.append(json.loads(line))

    train = list_to_df(train)
    val = list_to_df(val)
    test = list_to_df(test)

    return train, val, test


def cached_tokenize(sentence, tokenizer, cache):
    if (sentence, tokenizer.name_or_path) in cache:
        return cache[(sentence, tokenizer.name_or_path)]

    tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
    cache[(sentence, tokenizer.name_or_path)] = tokens

    return tokens


def cached_encode(sentence, tokenizer, cache):
    if (sentence, tokenizer.name_or_path) in cache:
        return cache[(sentence, tokenizer.name_or_path)]

    indices = tokenizer.encode(sentence, add_special_tokens=True)
    cache[(sentence, tokenizer.name_or_path)] = indices

    return indices


def collate_fn(batch, padding_idx: int = 0):
    sequences = [x[0] for x in batch]
    targets = [x[1] for x in batch]

    max_len = max(len(x) for x in sequences)
    sequences = torch.tensor([seq + [padding_idx] * (max_len - len(seq)) for seq in sequences]).long()
    targets = torch.tensor(targets).long()

    return sequences, targets


def collate_fn_biencoder(batch, padding_idx: int = 0):
    premises = [x[0] for x in batch]
    hypotheses = [x[1] for x in batch]
    targets = [x[2] for x in batch]

    max_len_p = max(len(p) for p in premises)
    premises = torch.tensor([p + [padding_idx] * (max_len_p - len(p)) for p in premises]).long()
    max_len_h = max(len(h) for h in hypotheses)
    hypotheses = torch.tensor([h + [padding_idx] * (max_len_h - len(h)) for h in hypotheses]).long()
    targets = torch.tensor(targets).long()

    return premises, hypotheses, targets


def collate_fn_buckets(batch):
    return batch[0]
