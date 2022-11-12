import os
import warnings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder

from PIL import Image

import pandas as pd
import time
import numpy as np
import sys
import conf
import json
import tqdm as tqdm

opt = conf.ImageNetOpt




class ImageNetDataset(torch.utils.data.Dataset):

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
        self.transform_type = transform

        assert (len(domains) > 0)
        if domains[0].startswith('original'):
            self.path = 'origin/Data/CLS-LOC/train/'
        elif domains[0].startswith('test'):
            self.path = 'origin/Data/CLS-LOC/val/'
        else :
            self.path = 'corrupted/'
            corruption, severity =domains[0].split('-')
            self.path += corruption+'/'+severity+'/'

        if transform == 'src':
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        elif transform == 'val':
            self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):

        path = self.file_path+'/'+self.path
        self.features = []
        self.class_labels = []
        self.domain_labels = []
        print('preprocessing images..')
        self.dataset = ImageFolder(path, transform=self.transform)

        '''  #make cache for valid date. 30% slower than without cache..
        if self.transform_type == 'val':
            print('caching val data for imagenet...')
            dataset = self.dataset

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False,
                num_workers=16, pin_memory=False, drop_last=False)

            transformed_dataset = []

            for b_i, data in enumerate(dataloader):  # must be loaded from dataloader, due to transform in the __getitem__()
                feat, cl= data

                # convert a batch of tensors to list, and then append to our list one by one
                feats= torch.unbind(feat, dim=0)
                cls = torch.unbind(cl, dim=0)
                for i in range(len(feats)):
                    transformed_dataset.append((feats[i], cls[i]))
            self.dataset =transformed_dataset
        '''

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl = self.dataset[idx]
        return img, cl, 0


if __name__ == '__main__':


    ### code for making imagenet validation data compatiable with ImageFolder!
    '''
    import os
    root = '/mnt/sting/tsgong/WWW/dataset/ImageNet-C/origin/Data/CLS-LOC/val/'
    f = open(root+'LOC_val_solution.csv', 'r')
    i=0
    for l in f:
        if i ==0: # ignore header
            i += 1
            continue
        filename=l.split(',')[0]
        label=l.split(',')[1].split(' ')[0]
        dir = root+label

        ### 1. make dir
        # if not os.path.exists(dir):
        #     os.makedirs(dir)

        # print(os.path.join(root,filename,'.JPEG'))

        ### 2. move files to dir
        print(label)
        if os.path.isfile(os.path.join(root, filename + '.JPEG')):
            os.rename(os.path.join(root, filename + '.JPEG'), os.path.join(dir, filename + '.JPEG'))

        i += 1
    print(i)
    '''


