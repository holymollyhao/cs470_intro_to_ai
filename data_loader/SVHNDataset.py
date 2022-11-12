import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset
import torchvision
import time
import conf
import numpy as np

opt = conf.SVHNOpt
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SVHNDataset(torch.utils.data.Dataset):
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
        # path = os.path.join(self.file_path, self.domains[0])
        #

        for domain in self.domains:
            if domain =='SVHN':

                tr = transforms.Compose([transforms.Resize([32, 32]),
                                               transforms.ToTensor()])

                dataset = torchvision.datasets.SVHN(os.path.join(conf.args.cache_path, 'svhn'), split='train', download=True,
                                        transform=tr)
                # t_test = torchvision.datasets.SVHN(os.path.join(conf.args.cache_path, 'svhn'), split='test', download=True, transform=tr)
            elif domain == 'test':

                tr = transforms.Compose([transforms.Resize([32, 32]),
                                               transforms.ToTensor()])

                dataset = torchvision.datasets.SVHN(os.path.join(conf.args.cache_path, 'svhn'), split='test', download=True,
                                        transform=tr)
            elif domain =='MNIST':
                tr = transforms.Compose([transforms.Resize([32, 32]),
                                       transforms.Grayscale(3),
                                               transforms.ToTensor()])

                # train = torchvision.datasets.MNIST(os.path.join(conf.args.cache_path, 'mnist'), train=True, download=True,
                #                         transform=tr)
                dataset = torchvision.datasets.MNIST(os.path.join(conf.args.cache_path, 'mnist'), train=False, download=True,
                                        transform=tr)
            elif domain == 'USPS':
                tr = transforms.Compose([transforms.Resize([32, 32]),
                                               transforms.ToTensor()])

                # t_train = torchvision.datasets.USPS(os.path.join(conf.args.cache_path, 'svhn'), train=True, download=True,
                #                         transform=tr)
                dataset = torchvision.datasets.USPS(os.path.join(conf.args.cache_path, 'svhn'), train=False, download=True, transform=tr)
            elif domain == 'MNIST-M':
                tr = transforms.Compose([transforms.Resize([32, 32]),
                                               transforms.ToTensor()])

                # t_train = torchvision.datasets.USPS(os.path.join(conf.args.cache_path, 'svhn'), train=True, download=True,
                #                         transform=tr)
                # t_test = torchvision.datasets.USPS(os.path.join(conf.args.cache_path, 'svhn'), train=False, download=True, transform=tr)

                dataset =torchvision.datasets.ImageFolder(root=opt['mnist_m_file_path'],
                                               transform=tr)


        for x, y in dataset:
            self.features.append(x)
            self.class_labels.append(y)
            self.domain_labels.append(0)

        self.features = torch.stack(self.features)
        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        # self.features = torch.utils.data.TensorDataset(torch.from_numpy(self.features))
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

