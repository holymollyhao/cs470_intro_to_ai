import os
import warnings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import time
import numpy as np
import sys
import conf

opt = conf.FineFoodOpt

# reference: https://github.com/alinlab/MASKER/blob/9ba319389184942c430e4b6d71209f4b1162220e/data/base_dataset.py
def tokenize(tokenizer, raw_text):
    max_len = min(tokenizer.model_max_length, 512) # 512 is the default for BERT model

    if len(raw_text) > max_len:
        raw_text = raw_text[:max_len]

    tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    tokens = torch.tensor(tokens).long()

    if tokens.size(0) < max_len:
        padding = torch.zeros(max_len - tokens.size(0)).long()
        padding.fill_(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        tokens = torch.cat([tokens, padding])

    return tokens


# reference: https://github.com/alinlab/MASKER/blob/9ba319389184942c430e4b6d71209f4b1162220e/data/base_dataset.py
def create_tensor_dataset(inputs, labels, domains):
    assert len(inputs) == len(labels) == len(domains)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels).unsqueeze(1)  # (N, 1)
    domains = torch.stack(domains).unsqueeze(1) # (N, 1)

    dataset = torch.utils.data.TensorDataset(inputs, labels, domains)

    return dataset


# reference: https://github.com/alinlab/MASKER/blob/9ba319389184942c430e4b6d71209f4b1162220e/data/base_dataset.py
class FineFoodDataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, tokenizer=None, transform='none'):

        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.domains = domains

        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']

        self.tokenizer = tokenizer

        assert (len(domains) > 0)
        if domains[0].startswith('train'):
            self.filename = 'foods_train.txt'
        elif domains[0].startswith('test'):
            self.filename = 'foods_test.txt'

        ppt = time.time()
        self.preprocessing(raw_text=False)

        print('Loading data done. \tPreprocessing:{:f}\tTotal Time:{:f}'.format(time.time() - ppt,
                                                                                time.time() - st))

    def preprocessing(self, raw_text=False):

        data_path = os.path.join(self.file_path, self.filename)
        with open(data_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split(':')

            if int(toks[1]) == 1:  # pre-defined class 0
                label = 0
            elif int(toks[1]) == 5:  # pre-defined class 1
                label = 1
            else:
                continue

            if raw_text:
                text = toks[0]
            else:
                text = tokenize(self.tokenizer, toks[0])

            label = torch.tensor(label).long()

            inputs.append(text)
            labels.append(label)

        domains = [torch.zeros(1) for i in range(len(labels))]

        if raw_text:
            self.dataset = zip(inputs, labels, domains)
        else:
            self.dataset = create_tensor_dataset(inputs, labels, domains)

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def class_to_number(self, label):
        dic = {
            'Negative': 0,
            'Positive': 1,
        }
        return dic[label]

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return self.dataset[idx]


if __name__ == '__main__':
    pass
