import os
import json
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


def collate_fn(x):
    return x[0]
