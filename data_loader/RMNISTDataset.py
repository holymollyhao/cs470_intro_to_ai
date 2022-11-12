import os
import warnings
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate

import pandas as pd
import time
import numpy as np
import sys
import conf

from torchvision.datasets import MNIST, ImageFolder

opt = conf.RMNISTOpt


class RMNISTDataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform='none'):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.preprocessing()

    def preprocessing(self):

        tr = transforms.Compose([transforms.Resize([32, 32]),
                                 transforms.Grayscale(3),
                                 transforms.ToTensor()])

        # train = torchvision.datasets.MNIST(os.path.join(conf.args.cache_path, 'mnist'), train=True, download=True,
        #                         transform=tr)
        dataset = torchvision.datasets.MNIST(os.path.join(conf.args.cache_path, 'mnist'), train=False, download=True,
                                             transform=tr)

        self.original_images = [x for (x, y) in dataset]
        self.original_labels = [y for (x, y) in dataset]
        for i, environment in enumerate(self.domains):
            index = opt['domains'].index(environment)
            images = self.original_images[index::len(opt['domains'])]
            labels = self.original_labels[index::len(opt['domains'])]
            (feature_list, label_list) = self.rotate_dataset(images, labels, int(environment))
            self.features += feature_list
            self.class_labels += label_list
            self.domain_labels += [i for j in range(len(feature_list))]

        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)

        self.dataset = torch.utils.data.TensorDataset(
            torch.stack(self.features),
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
        return img, cl, dl

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(
                lambda x: rotate(x, angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 3, 32, 32)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels

        return (x, y)


if __name__ == '__main__':
    pass
