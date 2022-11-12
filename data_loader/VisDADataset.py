import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset
import torchvision
import time
import conf
import numpy as np

opt = conf.VisDAOpt
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VisDADataset(torch.utils.data.Dataset):
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
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.preprocessing()

    def preprocessing(self):
        # path = os.path.join(self.file_path, self.domains[0])
        #

        for domain in self.domains:
            if domain =='sim':

                tr = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()])

                dataset =torchvision.datasets.ImageFolder(root=os.path.join(opt['file_path'], 'train'),
                                               transform=tr)
            elif domain == 'real':

                tr = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.ToTensor()])

                dataset =torchvision.datasets.ImageFolder(root=os.path.join(opt['file_path'], 'validation'),
                                               transform=tr)
        self.dataset = dataset
        # import tqdm
        # for x, y in tqdm.tqdm(dataset, total=len(dataset)):
        #     self.features.append(x)
        #     self.class_labels.append(y)
        #     self.domain_labels.append(0)
        #
        # self.features = torch.stack(self.features)
        # self.class_labels = np.array(self.class_labels)
        # self.domain_labels = np.array(self.domain_labels)
        # # self.features = torch.utils.data.TensorDataset(torch.from_numpy(self.features))
        # self.class_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.class_labels))
        # self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

        print("done preprocessing")

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], 0

