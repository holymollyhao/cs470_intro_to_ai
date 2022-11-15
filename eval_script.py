import argparse
import os
import sys
import re
import torch.utils.tensorboard as tf #use pytorch tensorboard; conda install -c conda-forge tensorboard
import multiprocessing as mp
from tqdm import tqdm
import json
import conf

args = None

def load_epoch(file_path):
    found = False
    for file in os.listdir(file_path):
        if 'events' in file:
            file_path = file_path + '/' + file
            found = True
            break
    if found:
        for e in tf.compat.v1.train.summary_iterator(file_path):
            for v in e.summary.value:
                if v.tag == 'args/epoch':
                    return str(v.tensor.string_val[0].decode('utf-8'))

    return str(0)


def get_avg_acc(file_path):
    result = 0

    f = open(file_path)
    lines = f.readlines()

    for line in lines:
        matchObj = re.match('.+\s([\d\.]+)', line)
        result += float(matchObj.group(1))

    f.close()

    return result/len(lines)

def get_avg_online_acc(file_path):

    f = open(file_path)
    json_data = json.load(f)


    f.close()

    return json_data['accuracy'][-1]

def get_avg_online_f1(file_path):

    f = open(file_path)
    json_data = json.load(f)


    f.close()

    return json_data['f1_macro'][-1]


def mp_work(path):
    # print(f'Current file:\t{path}')

    if args.eval_type == 'avg_acc': # with separate test data
        tmp_dict = {}
        tmp_dict[path] = get_avg_acc(path + '/accuracy.txt')
        return tmp_dict

    elif args.eval_type == 'avg_acc_online':  # test data is also training data
        tmp_dict = {}
        tmp_dict[path] = get_avg_online_acc(path + '/online_eval.json')
        return tmp_dict
    elif args.eval_type == 'avg_f1_online':  # test data is also training data
        tmp_dict = {}
        tmp_dict[path] = get_avg_online_f1(path + '/online_eval.json')
        return tmp_dict

    elif args.eval_type == 'estimation':
        tmp_dict = {}
        tmp_dict[path] = {}
        result = ''

        for method in args.method:
            tmp_dict[path][method] = {}
            try:

                re_found = re.search('\./log/(.+)/(.+)/(tgt_.+)/(.+)', path)
                alg = re_found.groups()[1]
                log_prefix = re_found.groups()[-1]
                dataset = re_found.groups()[0]
                epoch = load_epoch(path)
                tgt = re_found.groups()[2]
                result += alg + '\t' + log_prefix + '\t' + \
                          dataset + '\t' + epoch + '\t' + tgt + '\t'

                method, l1, l2, idx_diff, dpa, DTW, best_acc, best_epoch, acc_error = calc_estimation_metric(args, path,
                                                                                                             method)
            except Exception as e:
                print(e)
                print(f'Error Path:{path}')
                exit(1)

            result += f'{method}\t{l1:.4f}\t{l2:.4f}\t{idx_diff}\t{dpa:.4f}\t{DTW:.4f}\t{best_acc:.4f}\t{best_epoch:d}\t{acc_error:.4f}\n'

            tmp_dict[path][method]['l1'] = l1
            tmp_dict[path][method]['l2'] = l2
            tmp_dict[path][method]['idx_diff'] = idx_diff
            tmp_dict[path][method]['dpa'] = dpa
            tmp_dict[path][method]['DTW'] = DTW
            tmp_dict[path][method]['best_acc'] = best_acc
            tmp_dict[path][method]['best_epoch'] = best_epoch
            tmp_dict[path][method]['acc_error'] = acc_error
            tmp_dict[path][method]['string'] = result

        return tmp_dict


def print_per_domain(all_dict, opt, is_iabn):
    list_of_domains = opt['tgt_domains']
    list_of_acc_per_domain = [[] for i in range(len(list_of_domains))]
    for domain in list_of_domains:
        if is_iabn:
            iter_list = [(key, value) for (key, value) in sorted(all_dict.items()) if 'iabn' in key]
        else:
            iter_list = [(key, value) for (key, value) in sorted(all_dict.items()) if 'iabn' not in key]
        for (k, v) in iter_list:
            if domain in k:
                list_of_acc_per_domain[list_of_domains.index(domain)].append((k, v))
    return list_of_acc_per_domain


def format_print_per_domain(input_dict, opt):
    for domain in opt['tgt_domains']:
        print(domain, end=' ')
    print('')
    for index in range(3):
        for domain in opt['tgt_domains']:
            print(input_dict[opt['tgt_domains'].index(domain)][index][1], end=' ')
        print('')


def main(args):
    pattern_of_path = args.regex
    # print(pattern_of_path)
    root = '/mnt/sting/tsgong/WWW/log/' + args.directory

    path_list = []

    pattern_of_path = re.compile(pattern_of_path)
    # print(pattern_of_path)

    for (path, dir, files) in os.walk(root):
        if pattern_of_path.match(path):
            if not path.endswith('/cp'):  # ignore cp/ dir
                path_list.append(path)
    ########### Main Experiments #############
    # LOG_SUFFIX="220511_src" ## main
    # LOG_SUFFIX="220511_baselines" ## main
    # LOG_SUFFIX="220511_ablation" ## main

    ################### Hyper parameters #################
    is_iabn = False
    if 'iabn' in args.regex:
        is_iabn = True

    if 'harth' in args.directory:
        opt = conf.HARTHOpt
    elif 'reallifehar' in args.directory:
        opt = conf.RealLifeHAROpt
    elif 'extrasensory' in args.directory:
        opt = conf.ExtraSensoryOpt
    elif 'kitti_sot' in args.directory:
        opt = conf.KITTI_SOT_Opt
    elif 'cifar100' in args.directory:
        opt = conf.CIFAR100Opt
    elif 'cifar10' in args.directory:
        opt = conf.CIFAR10Opt
    elif 'pacs' in args.directory:
        opt = conf.PACSOpt
    elif 'svhn' in args.directory:
        opt = conf.SVHNOpt
    elif 'imagenet' in args.directory:
        opt = conf.ImageNetOpt
    elif 'finefood' in args.directory:
        opt = conf.FineFoodOpt
    elif 'sst-2' in args.directory:
        opt = conf.SST2Opt
    elif 'imdb' in args.directory:
        opt = conf.IMDBOpt
    elif 'tomatoes' in args.directory:
        opt = conf.TomatoesOpt
    else:
        raise NotImplementedError

    pool = mp.Pool()
    all_dict = {}
    with pool as p:
        ret = list(tqdm(p.imap(mp_work, path_list, chunksize=1), total=len(path_list)))
        for d in ret:
            all_dict.update(d)

    # print(f'Result from {len(path_list)} paths:')
    # print(sorted(all_dict.items()))
    avg = 0

    if args.per_domain:
        dict_list_per_domain = print_per_domain(all_dict, opt, is_iabn)
        format_print_per_domain(dict_list_per_domain, opt)
        # for dict_list in dict_list_per_domain:
        #     avg = 0
        #     for (k, v) in dict_list:
        #         avg += v
        #         print(k, v)
        #     # print(f'avg_acc:\t{avg / len(dict_list)}')
    else:
        for k, v in sorted(all_dict.items()):
            avg += v
            print(k, v)

        # for path in path_list:
        #     print(path)

        print(f'avg_acc:\t{avg / len(all_dict.keys())}')


def parse_arguments(argv):
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--regex', type=str, default='', help='train condition regex')
    parser.add_argument('--directory', type=str, default='',
                        help='which directory to search through? ex: ichar/FT_FC')
    parser.add_argument('--eval_type', type=str, default='avg_acc',
                        help='what type of evaluation? in [result, log, estimation, dtw, avg_acc]')

    ### Methods ###
    parser.add_argument('--method', default=[], nargs='*',
                        help='method to evaluate, e.g., dev, iwcv, entropy, etc.')
    parser.add_argument('--per_domain', action='store_true', default=False, help='evaluation done per domain')

    return parser.parse_args()

if __name__ == '__main__':
    import time

    # print('Command:', end='\t')
    # print(" ".join(sys.argv))

    st = time.time()
    args = parse_arguments(sys.argv[1:])
    main(args)
    print('')
    # print(f'time:{time.time() - st}')
