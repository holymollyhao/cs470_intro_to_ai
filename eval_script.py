import argparse
import os
import sys
import re
import multiprocessing as mp
from tqdm import tqdm
import json
import conf
from texttable import Texttable


args = None

method_format = {
    'Src': 'Source',
    'ln_tent' : 'TENT-LN',
    'ttaprompttune': 'Ours'
}

def get_avg_online_acc(file_path):

    if os.path.exists(file_path):
        f = open(file_path)
        json_data = json.load(f)


        f.close()
        return json_data['accuracy'][-1]
    else:
        print("one of the results isn't saved properly, probably due to OOM error")
        return -1



def mp_work(path):

    tmp_dict = {}
    tmp_dict[path] = get_avg_online_acc(path + '/online_eval.json')

    return tmp_dict


def main(args):
    pattern_of_path = args.regex
    print(pattern_of_path)
    root = './' + args.directory

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

    for model in ['distilbert']:
        draw_single_instance(model, all_dict)
        print('\n')

def get_accuracy_from_dict(
        method: str,
        model: str,
        memsize: str,
        dataset1: str,
        dataset2: str,
        lr: str,
        all_dict: dict,
        single: bool
):
    if method == 'Src':
        target_str = f'model_{model}_from_{dataset2}_to_{dataset1}'
    else:
        target_str = f'model_{model}_lr{lr}_memsize{memsize}_uex{memsize}_from_{dataset2}_to_{dataset1}'
    target = []
    # print(target_str)
    # print(method)
    for key in sorted(all_dict.keys()):
        if target_str in key and method in key:
            # target = (key, all_dict[key])
            target.append((key, all_dict[key]))
    assert target != None
    # print(target)
    if single:
        return target[0]
    else:
        return target

def draw_single_instance(
        model: str,
        all_dict: dict,
    ):

    datasets = sorted(['finefood', 'sst-2', 'imdb'])
    table = Texttable()
    table.set_cols_align(['l', 'c', 'c', 'c', 'c'])
    table.set_cols_width(['10', '15', '15', '15', '15'])

    first_col = [f'Method\nwith\n{model.upper()}']
    for dataset1 in datasets:
        for dataset2 in datasets:
            if dataset1 != dataset2:
                first_col.append(f'{dataset2}\nto\n{dataset1}')
    table.add_row(first_col)

    for method in ['Src', 'ln_tent', 'ttaprompttune']:
        single_row = [method_format[method]]
        for dataset1 in ['finefood', 'sst-2']:
            for dataset2 in datasets:
                if dataset1 != dataset2:
                    import numpy as np
                    multi_acc = get_accuracy_from_dict(
                        method=method,
                        model=model,
                        memsize=32,
                        dataset1=dataset1,
                        dataset2=dataset2,
                        lr=0.001,
                        all_dict= all_dict,
                        single=False
                    )
                    avg = np.round(np.average([i[1] for i in multi_acc]), 2)
                    std = np.round(np.std([i[1] for i in multi_acc]), 1)
                    single_row.append(f'{avg} +- {std}')

        table.add_row(single_row)
    print(table.draw())


def parse_arguments(argv):
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--regex', type=str, default='.*submission_results.*', help='train condition regex')
    parser.add_argument('--directory', type=str, default='log',
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

    st = time.time()
    args = parse_arguments(sys.argv[1:])
    main(args)
    print('')
