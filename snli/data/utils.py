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


def collate_fn(batch, tokenizer):
    sequences = [x[0] for x in batch]
    targets = [x[1] for x in batch]

    encoded = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True)
    sequences, attention_mask = encoded.input_ids, encoded.attention_mask
    targets = torch.tensor(targets).long()

    return sequences, attention_mask, targets


def collate_fn_biencoder(batch, tokenizer):
    premises = [x[0][0] for x in batch]
    hypotheses = [x[0][1] for x in batch]
    targets = [x[1] for x in batch]

    premises = tokenizer(premises, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True)
    premises, attention_mask_p = premises.input_ids, premises.attention_mask

    hypotheses = tokenizer(hypotheses, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True)
    hypotheses, attention_mask_h = hypotheses.input_ids, hypotheses.attention_mask

    targets = torch.tensor(targets).long()

    return premises, hypotheses, attention_mask_p, attention_mask_h, targets


def collate_fn_buckets(batch):
    return batch[0]
