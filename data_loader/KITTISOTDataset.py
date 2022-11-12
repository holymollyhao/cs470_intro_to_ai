import os
import warnings
import torch.utils.data
import pandas as pd
import time
import numpy as np
import sys
import conf
from PIL import Image
# from skimage.transform import resize
import torchvision.transforms as transforms

opt = conf.KITTI_SOT_Opt


class KITTISOTDataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform=None):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source
        self.features = []
        self.class_labels = []
        self.domain_labels = []
        self.file_path = opt['file_path']
        self.max_objects = 50
        self.class_index = [opt['classes'].index(i) for i in opt['sub_classes']]

        assert (len(domains) > 0)
        if domains[0].startswith('original'):
            self.sub_path1 = 'origin'
            self.sub_path2 = ''

        elif domains[0] == '2d_detection':
            self.sub_path1 = '2d_detection'
            self.sub_path2 = ''
        elif domains[0].startswith('rain'):
            self.sub_path1 = domains[0].split('-')[0]
            self.sub_path2 = domains[0].split('-')[1] + 'mm'
            # self.sub_path1 = 'rain'
            # self.sub_path2 = domains[0].lstrip('rain-')
        elif domains[0] == 'half1':
            self.sub_path1 = 'origin'
            self.sub_path2 = 'train'
        elif domains[0] == 'half2':
            self.sub_path1 = 'origin'
            self.sub_path2 = 'train'


        if transform == 'src':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        elif transform == 'val':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):

        path = f'{self.file_path}/{self.sub_path1}/{self.sub_path2}/'
        # ['0000', '0001']

        for sequence_name in sorted(os.listdir(path)):
            if conf.args.dataset == 'kitti_sot_test':
                if sequence_name not in ['0000', '0001']:
                    continue
            if self.domains[0] == 'half1':
                if sequence_name not in ['0000', '0001']:
                    # if sequence_name not in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']:
                    continue
            elif self.domains[0] == 'half2':
                # if sequence_name not in ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']:
                if sequence_name not in ['0011', '0012']:
                    continue
            elif self.domains[0].endswith('val'):
                if sequence_name not in ['0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019',
                                         '0020']:
                    continue
            elif self.domains[0].endswith('tgt'):
                if sequence_name not in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008',
                                         '0009']:
                    continue

            png_list = [f for f in sorted(os.listdir(os.path.join(path, sequence_name))) if f.endswith('png')]
            print(f'total_len is : {len(png_list)}')
            for png_file in png_list:
                # ---------
                #  Image
                # ---------

                img_path = os.path.join(path, sequence_name, png_file)
                img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

                # ---------
                #  Label
                # ---------

                label_path = os.path.join(path, sequence_name, png_file.replace('png','txt'))
                labels = None
                if os.path.exists(label_path):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        labels = np.loadtxt(label_path)

                # filled_labels = torch.from_numpy(filled_labels)
                self.features.append(img)
                self.class_labels.append(int(labels))

                # assuming that single domain is passed as list
                self.domain_labels.append(0)  # TODO: change this if domain label is required

        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        self.class_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.class_labels))
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

    def __len__(self):
        return len(self.features)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.get_datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl, dl = self.features[idx], self.class_labels[idx], self.domain_labels[idx]
        img = self.transform(img)
        return img, cl[0], dl[0]  # cl, dl are tuple of size 1

if __name__ == '__main__':
    pass
