import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import conf
from .dnn import DNN
from models.emebdding_layer import SoftEmbedding
from utils.loss_functions import *
from utils.util_functions import print_summary, get_device

device = get_device()

class Prompt_tuning(DNN):
    def __init__(self, *args, **kwargs):
        super(Prompt_tuning, self).__init__(*args, **kwargs)

        n_tokens = conf.args.n_tokens
        initialize_from_vocab = not conf.args.no_init_from_vocab,
        self.n_tokens = n_tokens
        self.initialize_from_vocab = initialize_from_vocab

        if conf.args.parallel:
            input_embeddings = self.net.module.get_input_embeddings()
        else:
            input_embeddings = self.net.get_input_embeddings()


        ## only can be used with huggingface transformer models
        self.soft_embedding = SoftEmbedding.SoftEmbedding(
            input_embeddings,
            n_tokens=self.n_tokens,
            initialize_from_vocab = self.initialize_from_vocab
        )

        # setting the previous input_embeddings to current embedding
        if conf.args.parallel:
            self.net.module.set_input_embeddings(self.soft_embedding)
        else:
            self.net.set_input_embeddings(self.soft_embedding)

        # load checkpoint from specified path, if specified
        checkpoint_path = conf.args.load_checkpoint_path
        if checkpoint_path and 'Src' in checkpoint_path:
            self.load_checkpoint_naive(checkpoint_path)
        elif checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.set_gradients('embed')


    # prefix tuning can only be done in an offline manner
    def train(self, current_num_sample):
        """
        Train the model
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        self.net.train()
        class_loss_sum = 0.0
        total_iter = 0

        num_iter = len(self.source_dataloader['train'])
        total_iter += num_iter


        for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):

            feats, cls, _ = labeled_data
            feats, cls = feats.to(device), cls.to(device)
            if conf.args.dataset in ['imdb', 'sst-2', 'finefood', 'tomatoes']:
                cls = cls.squeeze(1)  # (B)

            # compute the feature
            preds = self.net(feats)  # (B, C)
            class_loss = self.class_criterion(preds, cls)
            class_loss_sum += float(class_loss * feats.size(0))

            # reset optimizer gradient
            self.optimizer.zero_grad()

            # backpropagation of loss
            class_loss.backward()

            # take gradient step
            self.optimizer.step()

            # take scheduler step
            self.scheduler.step()


    def train_online(self, current_num_sample):
        """
        Train the model
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, 'previous_train_loss'):
            self.previous_train_loss = 0

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        # Add a sample
        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        self.mem.add_instance(current_sample)

        if conf.args.use_learned_stats:  # batch-free inference
            self.evaluation_online(current_num_sample, '',
                                   [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                return SKIPPED

        # Evaluate with a batch
        if not conf.args.use_learned_stats:  # batch-based inference
            self.evaluation_online(current_num_sample, '', self.mem.get_memory())

        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)

        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        entropy_loss = HLoss()

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)
                preds_of_data = self.net(feats)

                loss = entropy_loss(preds_of_data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()





