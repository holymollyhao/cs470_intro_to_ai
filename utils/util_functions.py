import conf
# import tensorboard
import os
import sys
from torchsummary import summary
import numpy as np


def load_opt(dataset: str): # load according opt to dataset name
    if 'harth' in conf.args.dataset:
        opt = conf.HARTHOpt
    elif 'reallifehar' in conf.args.dataset:
        opt = conf.RealLifeHAROpt
    elif 'extrasensory' in conf.args.dataset:
        opt = conf.ExtraSensoryOpt
    elif 'kitti_sot' in conf.args.dataset:
        opt = conf.KITTI_SOT_Opt
    elif 'cifar100' in conf.args.dataset:
        opt = conf.CIFAR100Opt
    elif 'cifar10' in conf.args.dataset:
        opt = conf.CIFAR10Opt
    elif 'tinyimagenet' in conf.args.dataset:
        opt = conf.TinyImageNetOpt
    elif 'imagenet' in conf.args.dataset:
        opt = conf.ImageNetOpt
    elif 'vlcs' in conf.args.dataset:
        opt = conf.VLCSOpt
    elif 'officehome' in conf.args.dataset:
        opt = conf.OfficeHomeOpt
    elif 'pacs' in conf.args.dataset:
        opt = conf.PACSOpt
    elif 'svhn' in conf.args.dataset:
        opt = conf.SVHNOpt
    elif 'visda' in conf.args.dataset:
        opt = conf.VisDAOpt
    elif 'cmnist' in conf.args.dataset:
        opt = conf.CMNISTOpt
    elif 'rmnist' in conf.args.dataset:
        opt = conf.RMNISTOpt
    elif 'domainnet' in conf.args.dataset:
        opt = conf.DomainNetOpt
    elif 'terra_incognita' in conf.args.dataset:
        opt = conf.TerraIncognitaOpt
    elif 'mnist' in conf.args.dataset:
        opt = conf.MNISTOpt
    elif 'imdb' in conf.args.dataset:
        opt = conf.IMDBOpt
    elif 'sst-2' in conf.args.dataset:
        opt = conf.SST2Opt
    elif 'finefood' in conf.args.dataset:
        opt = conf.FineFoodOpt
    elif 'tomatoes' in conf.args.dataset:
        opt = conf.TomatoesOpt
    else:
        raise ValueError(f'No matching opt for dataset {dataset}')
    return opt

def ensure_result_directory(result_path, checkpoint_path, conf):
    if not os.path.exists(result_path):
        oldumask = os.umask(0)
        os.makedirs(result_path, 0o777)
        os.umask(oldumask)
    if not os.path.exists(checkpoint_path):
        oldumask = os.umask(0)
        os.makedirs(checkpoint_path, 0o777)
        os.umask(oldumask)
    # for arg in vars(conf.args):
        # tensorboard.log_text('args/' + arg, getattr(conf.args, arg), 0)
    script = ' '.join(sys.argv[1:])
    # tensorboard.log_text('args/script', script, 0)

def print_summary(model):
    print("###### SUMMARY of MODEL ######")
    cnt = 0
    for params in model.parameters():
        if params.requires_grad:
            print(f'{cnt}th trainable parameter: {np.prod(params.size())}')
            cnt += 1
    if cnt == 0:
        print(f'No trainable parameters')
    print("##############################")