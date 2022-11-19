import conf
import os
import sys
import numpy as np
import torch


def load_opt(dataset: str): # load according opt to dataset name
    if 'imdb' in dataset:
        opt = conf.IMDBOpt
    elif 'sst-2' in dataset:
        opt = conf.SST2Opt
    elif 'finefood' in dataset:
        opt = conf.FineFoodOpt
    elif 'tomatoes' in dataset:
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


def get_path():
    path = 'log/'

    # information about used data type
    path += conf.args.dataset + '/'

    # information about used model type
    path += conf.args.method + '/'

    # information about domain(condition) of training data
    if conf.args.src == ['rest']:
        path += 'src_rest' + '/'
    elif conf.args.src == ['all']:
        path += 'src_all' + '/'
    elif conf.args.src is not None and len(conf.args.src) >= 1:
        path += 'src_' + '_'.join(conf.args.src) + '/'

    if conf.args.tgt:
        path += 'tgt_' + conf.args.tgt + '/'

    # get_default_log_prefix(path)
    path += conf.args.log_prefix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path


def is_instance_of_any(module, instance_list):
    for instance in instance_list:
        if isinstance(module, instance):
            return True
    return False


def set_gradients(parameter, boolean):
    for params in parameter:
        params.requires_grad = boolean


def get_max_position_embeddings():
    if 'bert' in conf.args.model:
        return 512, 512
    elif 'bart' in conf.args.model:
        return 1024, 1024


def get_device():
    return torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")