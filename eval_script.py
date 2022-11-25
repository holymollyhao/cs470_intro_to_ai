import argparse
import os
import sys
import re
import torch.utils.tensorboard as tf #use pytorch tensorboard; conda install -c conda-forge tensorboard
import multiprocessing as mp
from tqdm import tqdm
import json
import conf
from texttable import Texttable


args = None

method_format = {
    'Src': 'Source',
    'ln_tent' : 'TENT-LN',
    'dattaprompttune' : 'TeTra'
}
best_config ={
    'finefood':{
        'distilbert':{
            'uex': '16',
            'memsize': '16',
            'lr': '0.01'
        },
        'bert':{
            'uex': '32',
            'memsize': '32',
            'lr': '0.01'
        },
    },
    'imdb':{
        'distilbert':{
            'uex': '16',
            'memsize': '16',
            'lr': '0.00001'
        },
        'bert':{
            'uex': '32',
            'memsize': '32',
            'lr': '0.00001'
        },
    },
    'sst-2':{
        'distilbert':{
            'uex': '128',
            'memsize': '128',
            'lr': '0.00001'
        },
        'bert':{
            'uex': '128',
            'memsize': '128',
            'lr': '0.00001'
        },
    }
}

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
    print(pattern_of_path)
    root = '/home/twkim/git/tetra/' + args.directory

    path_list = []

    pattern_of_path = re.compile(pattern_of_path)
    # print(pattern_of_path)

    for (path, dir, files) in os.walk(root):
        if pattern_of_path.match(path):
            if not path.endswith('/cp'):
                path_list.append(path)

    pool = mp.Pool()
    all_dict = {}
    with pool as p:
        ret = list(tqdm(p.imap(mp_work, path_list, chunksize=1), total=len(path_list)))
        for d in ret:
            all_dict.update(d)

    for model in ['bert', 'distilbert']:
        draw_single_instance(model, all_dict)
        print('\n')

def get_accuracy_from_dict(
        method: str,
        model: str,
        memsize: str,
        dataset1: str,
        dataset2: str,
        lr: str,
        all_dict: dict
):
    if method == 'Src':
        target_str = f'model_{model}_from_{dataset2}_to_{dataset1}'
    else:
        target_str = f'model_{model}_lr{lr}_memsize{memsize}_uex{memsize}_from_{dataset2}_to_{dataset1}'
        
    target = None
    for key in all_dict.keys():
        if target_str in key and method in key:
            target = (key, all_dict[key])
            break
    assert target != None
    
    return target

def draw_single_instance(
        model: str,
        all_dict: dict,
    ):

    datasets = sorted(list(best_config.keys()))
    table = Texttable()
    table.set_cols_align(['l', 'c', 'c', 'c', 'c', 'c', 'c'])

    first_col = [f'Method\nwith\n{model.upper()}']
    for dataset1 in datasets:
        for dataset2 in datasets:
            if dataset1 != dataset2:
                first_col.append(f'{dataset2}\nto\n{dataset1}')
    table.add_row(first_col)

    for method in ['Src','ln_tent', 'dattaprompttune']:
        single_row = [method_format[method]]
        for dataset1 in datasets:
            for dataset2 in datasets:
                if dataset1 != dataset2:
                    single_acc = get_accuracy_from_dict(
                        method=method,
                        model=model,
                        memsize=best_config[dataset1][model]['uex'],
                        dataset1=dataset1,
                        dataset2=dataset2,
                        lr=best_config[dataset1][model]['lr'],
                        all_dict= all_dict
                    )[1]
                    single_row.append(
                        single_acc
                    )
        table.add_row(single_row)
    print(table.draw())


def parse_arguments(argv):
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--regex', type=str, default='', help='train condition regex')
    parser.add_argument('--directory', type=str, default='',
                        help='which directory to search through? ex: ichar/FT_FC')
    parser.add_argument('--eval_type', type=str, default='avg_acc_online',
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
