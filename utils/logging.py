import conf
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tqdm
# https://stackoverflow.com/questions/10097477/python-json-array-newlines
def to_json(o, level=0, indent=3, space=" ", newline="\n"):
    INDENT = indent
    SPACE = space
    NEWLINE = newline

    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ",".join(to_json(e, level + 1) for e in o) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.float32):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


def log_sequence_stats(dirichlet_numchunks, sequence_stats):
    if conf.args.dataset == 'cifar10':
        color_map = {
            0: 'tomato',
            1: 'sandybrown',
            2: 'gold',
            3: 'forestgreen',
            4: 'aquamarine',
            5: 'royalblue',
            6: 'blueviolet',
            7: 'lightpink',
            8: 'darkgrey',
            9: 'saddlebrown'
        }

    # print(sequence_stats)

    plt.figure(figsize=(10,2), dpi=200)
    plt.ylim(-0.5, 0.5)
    plt.xlim(0, 10000)
    for i, seq in tqdm.tqdm(enumerate(sequence_stats), total=len(sequence_stats)):
        plt.barh(range(1), 1, height=0.1, left=i, color=color_map[seq])

    title = conf.args.dataset + "_shuffle_classes_"
    title += str(dirichlet_numchunks) + "chunks_beta" + str(conf.args.dirichlet_beta)
    plt.title(title)
    fname = title + ".png"
    plt.savefig(fname, dpi=200)

def log_dirichlet_data_stats(dirichlet_numchunks, cl_labels, idx_batch, idx_batch_cls):
    chunk_dataidx_map = {}
    for j in range(dirichlet_numchunks):
        chunk_dataidx_map[j] = idx_batch[j]
    chunk_cls_counts = {}
    for chunk_i, dataidx in chunk_dataidx_map.items():
        unq, unq_cnt = np.unique(cl_labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        chunk_cls_counts[chunk_i] = tmp
    print("DIRICHLET STATS")
    print('Data statistics: %s' % str(chunk_cls_counts))

class LogBNStats:
    def __init__(self, write_path, num_bn_layers):
        self.write_path = write_path
        self.num_bn_layers = num_bn_layers

        self.total_list_raw = []  # list of temp_bn_list_raws
        self.total_list_mean = []  # List of temp_bn_list_means
        self.total_list_running_mean = []
        self.total_list_running_var = []

        self.temp_bn_list_raw = []  # list of (module_input, module_output)
        self.temp_bn_list_mean = []  # list of (module_input_mean, module_output_mean)
        self.temp_bn_list_running_mean = []
        self.temp_bn_list_running_var = []

        # json is as follows :
        # {
        #     sample  # : [(module_input, module_output), ...]
        #     -> the list keeps piling on and on
        # }
        self.json = {}

        pass

    def __call__(self, module, module_in, module_out):
        if conf.args.method != 'Src':
            # using the total number of bn layers, we can save the outputs accordingly
            if len(self.temp_bn_list_raw) < self.num_bn_layers:
                self.temp_bn_list_raw.append((module_in[0].tolist(), module_out.tolist()))
                self.temp_bn_list_mean.append((float(torch.mean(module_in[0])), float(torch.mean(module_out))))
                self.temp_bn_list_running_mean.append(float(torch.mean(module.running_mean)))
                self.temp_bn_list_running_var.append(float(torch.mean(module.running_var)))

            if len(self.temp_bn_list_raw) == self.num_bn_layers:
                self.total_list_raw.append(self.temp_bn_list_raw)
                self.total_list_mean.append(self.temp_bn_list_mean)

                self.total_list_running_mean.append(self.temp_bn_list_running_mean)
                self.total_list_running_var.append(self.temp_bn_list_running_var)

                self.temp_bn_list_raw = []
                self.temp_bn_list_mean = []
                self.temp_bn_list_running_mean = []
                self.temp_bn_list_running_var = []

                self.json = {
                    'raw': self.total_list_raw,
                    'mean': self.total_list_mean,
                    'running_mean': self.total_list_running_mean,
                    'running_var': self.total_list_running_var,
                    'track_running_stats_bool': module.track_running_stats,
                }
            pass
        # print(f'\nmodule_in & out has shape : {module_in[0].size()}')

    def get_json(self):
        return self.json

    def dump_logbnstats_result(self):
        pickle_file = open(self.write_path + 'logbnstats.pickle', 'wb')
        pickle.dump(self.json, pickle_file)
        pickle_file.close()
        print("saving pickle complete")

        # saving the file as string json takes excruciating amount of time
        # json_file = open(self.write_path + 'logbnstats.json', 'w')
        # json.dump(self.total_list, json_file)
        # json_file.close()
        # print("saving json complete")


