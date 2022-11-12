import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.transforms.functional import rotate
from torchvision.datasets import MNIST, ImageFolder
from torchvision.datasets.vision import VisionDataset

import time
import conf
import sys
import conf
import numpy as np
import json

opt = conf.OfficeHomeOpt
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OfficeHomeDataset(torch.utils.data.Dataset):
    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform='none'):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.img_shape = opt['img_size']
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.preprocessing()

    def preprocessing(self):
        # path = os.path.join(self.file_path, self.domains[0])
        #
        for i, environment in enumerate(self.domains):
            env_transform = self.transform
            path = os.path.join(self.file_path, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            for (feature, classidx) in env_dataset:
                self.features.append(feature)
                self.class_labels.append(int(classidx))
                self.domain_labels.append(i)

        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        self.class_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.class_labels))
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

        print("done preprocessing")

    def __len__(self):
        return len(self.features)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        return self.features[idx], self.class_labels[idx][0], self.domain_labels[idx][0]
