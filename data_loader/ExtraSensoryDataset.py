import torch.utils.data
import pandas as pd
import time
import numpy as np
import sys

sys.path.append('..')
import conf

opt = conf.ExtraSensoryOpt
WIN_LEN = opt['seq_len']


class ExtraSensoryDataset(torch.utils.data.Dataset):
    # load static files

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            domains: condition on user-phone combination
            activities: condition on action

        """
        st = time.time()
        self.domain = domains
        self.activity = activities
        self.max_source = max_source

        self.domains = domains
        self.df = pd.read_csv(file)

        if domains is not None:
            cond_list = []
            for d in domains:
                cond_list.append('user == "{:s}"'.format(d))
            cond_str = ' | '.join(cond_list)
            self.df = self.df.query(cond_str)

        if activities is not None:
            cond_list = []
            for d in activities:
                cond_list.append('label == "{:s}"'.format(d))
            cond_str = ' | '.join(cond_list)
            self.df = self.df.query(cond_str)

        ppt = time.time()

        self.preprocessing()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.datasets = []  # list of dataset per each domain

        for idx in range(max(len(self.df) // WIN_LEN, 0)):
            domain = self.df.iloc[idx * WIN_LEN, 32]
            class_label = self.df.iloc[idx * WIN_LEN, 33]
            domain_label = self.domains.index(domain)

            feature = self.df.iloc[idx * WIN_LEN:(idx + 1) * WIN_LEN, 1:32].values
            feature = feature.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(class_label))
            self.domain_labels.append(domain_label)

        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        print(f'doamin_labels are  : {self.domain_labels}')

        # append dataset for each domain
        for domain_idx in range(self.get_num_domains()):
            indices = np.where(self.domain_labels == domain_idx)[0]
            self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features[indices]).float(),
                                                                torch.from_numpy(self.class_labels[indices]),
                                                                torch.from_numpy(self.domain_labels[indices])))
        # concated dataset
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    def __len__(self):
        # return max(len(self.df) // OVERLAPPING_WIN_LEN - 1, 0)
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def class_to_number(self, label):
        dic = {
            'label:LYING_DOWN': 0,
            'label:SITTING': 1,
            'label:FIX_walking': 2,
            'label:FIX_running': 3,
            'label:OR_standing': 4}
        return dic[label]

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return self.dataset[idx]


if __name__ == '__main__':
    pass
