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
from PIL import Image
opt = conf.TinyImageNetOpt

# Referred to https://github.com/henrikmarklund/arm/blob/release_2021/data/tinyimagenet_dataset.py

class TinyImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform='none'):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.domains = domains
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']

        assert (len(domains) > 0)
        if domains[0].startswith('original'):
            self.path = 'origin/train/'
        elif domains[0].startswith('test'):
            self.path = 'origin/val/'
        else :
            self.path = 'corrupted/'
            corruption, severity =domains[0].split('-')
            self.path += corruption+'/'+severity+'/'

        if transform == 'src':
            self.transform = None
        elif transform == 'val':
            self.transform = None
        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):

        path = self.file_path+'/'+self.path
        self.images=[]
        self.class_labels=[]
        self.domains_labels = []

        if self.domains[0].startswith('test'): # special dir structure for val data

            label_list = []
            data_dict = {}
            with open(path+'val_annotations.txt', 'r') as f:
                for l in f:
                    img_name, label, _, _, _, _ = l.split('\t')
                    data_dict[img_name]=label
                    label_list.append(label)


            for k, v in data_dict.items():
                img = np.array(Image.open(os.path.join(path+'/images/', k)).convert('RGB'))
                self.images.append(img)
                self.class_labels.append(sorted(list(set(label_list))).index(v))
                self.domains_labels.append(0)

        else:
            for label_i, label in enumerate(sorted(os.listdir(path))):
                img_path = os.path.join(path, label)

                if os.path.isdir(img_path+'/images/'):
                    img_path = img_path+'/images/' # structure of the source data

                for image_i, img_name in enumerate(sorted(os.listdir(img_path))):

                    img = np.array(Image.open(os.path.join(img_path, img_name)).convert('RGB'))
                    self.images.append(img)
                    self.class_labels.append(label_i)
                    self.domains_labels.append(0)
        self.images = np.array(self.images)
        self.images = np.transpose(self.images, (0,3,1,2))
        self.images = self.images.astype(np.float32)/255.0

        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domains_labels)

        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.images),
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
