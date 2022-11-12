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

opt = conf.MNISTOpt


class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.domains = domains

        self.img_shape = opt['img_size']
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']

        assert (len(domains) > 0)
        if domains[0].startswith('original'):
            self.sub_path1 = 'identity'
            self.data_filename = 'train_images.npy'
            self.label_filename = 'train_labels.npy'
        elif domains[0].startswith('test'):
            self.sub_path1 = 'identity'
            self.data_filename = 'test_images.npy'
            self.label_filename = 'test_labels.npy'
        else: # other corrupted domains
            self.sub_path1 = domains[0]
            self.data_filename = 'test_images.npy'
            self.label_filename = 'test_labels.npy'

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])


        self.preprocessing()

    def preprocessing(self):

        path = f'{self.file_path}/{self.sub_path1}/'

        data = np.load(path + self.data_filename)
        # change NHWC to NCHW format
        data = np.transpose(data, (0, 3, 1, 2))
        # make it compatible with our models (normalize)
        data = data.astype(np.float32) / 255.0

        self.features = data
        self.class_labels = np.load(path + self.label_filename)
        # assume that single domain is passed as List
        self.domain_labels = np.array([0 for i in range(len(self.features))])

        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.features),
            torch.from_numpy(self.class_labels),
            torch.from_numpy(self.domain_labels))

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl


if __name__ == '__main__':
    pass
