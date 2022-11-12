import os
import warnings
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import time
import numpy as np
import sys
import conf

from torchvision.datasets import MNIST, ImageFolder

opt = conf.CMNISTOpt


class CMNISTDataset(torch.utils.data.Dataset):

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
                                 # transforms.Grayscale(3),
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
            (feature_list, label_list) = self.color_dataset(images, labels, float(environment))
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

    def color_dataset(self, images, labels, environment):
        labels = torch.tensor(torch.from_numpy(np.array(labels) < 5), dtype=torch.float64)
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.concat([torch.stack(images), torch.stack(images), torch.stack(images)], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        images[torch.tensor(range(len(images))), 2, :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return (x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


if __name__ == '__main__':
    pass
