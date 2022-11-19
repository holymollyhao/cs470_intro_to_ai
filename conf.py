args = None

IMDBOpt = {
    'name': 'imdb',
    'batch_size': 8,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './imdb_dataset', #'./dataset/imdb',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}

SST2Opt = {
    'name': 'sst-2',
    'batch_size': 8,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'prompt_learning_rate': 0.3,
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './dataset/sst-2',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}

FineFoodOpt = {
    'name': 'finefood',
    'batch_size': 8,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'prompt_learning_rate': 0.3,
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './dataset/finefood',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}

TomatoesOpt = {
    'name': 'tomatoes',
    'batch_size': 8,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'prompt_learning_rate': 0.3,
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './dataset/tomatoes',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
    'src_domains': ["train"],
    'tgt_domains': ["test"],
}
