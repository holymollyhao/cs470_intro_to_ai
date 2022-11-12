import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import math
import conf
from copy import deepcopy
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import memory

from utils import iabn
from utils.logging import *
from utils.bn_remover import *
from utils.normalize_layer import *
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator


class DNN():
    def __init__(self, model, source_dataloader, target_dataloader, write_path):
        self.device = device
        # self.tensorboard = tensorboard

        # init dataloader
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader

        if conf.args.tgt_train_dist == 0 and \
                conf.args.dataset in ['cifar10', 'cifar100', 'vlcs', 'officehome', 'pacs', 'tinyimagenet', 'svhn',
                                      'visda', 'cmnist', 'rmnist', 'terra_incognita','domainnet', 'mnist', 'finefood',
                                      'sst-2', 'imdb', 'tomatoes']:
            self.tgt_train_dist = 4  # Dirichlet is default for non-real-distribution data
        else:
            self.tgt_train_dist = conf.args.tgt_train_dist

        self.target_data_processing()
        self.write_path = write_path

        ################## Init & prepare model###################
        self.conf_list = []

        # Load model
        if conf.args.model in ['bert', 'distilbert', 'bert-tiny', 'bert-mini',
                               'bert-small', 'bert-medium', 'mobilebert', 'rvt']:
            self.net = model
        elif 'resnet' in conf.args.model:
            if conf.args.dataset == 'imagenet':
                self.net = model
            else:
                num_feats = model.fc.in_features
                model.fc = nn.Linear(num_feats, conf.args.opt['num_class'])  # match class number
                self.net = model
        else:
            self.net = model.Net()


        # IABN
        # if conf.args.iabn and not conf.args.pretrain_wo_iabn:
        #     iabn.convert_iabn(self.net)


        if conf.args.load_checkpoint_path and conf.args.model not in ['wideresnet28-10',
                                                                      'resnext29']:  # false if conf.args.load_checkpoint_path==''
            if conf.args.dataset in ['imagenet'] and conf.args.method != 'Ours': # Baselines other than NOTE should use a pre-trained model.
                pass
            elif conf.args.dataset in ['imagenet'] and conf.args.method == 'Ours' and not conf.args.iabn: #if ablation on without iabn, load pretrained imagenet.
                pass
            else:
                self.load_checkpoint(conf.args.load_checkpoint_path)


        # if conf.args.iabn and conf.args.pretrain_wo_iabn:
        #     iabn.convert_iabn(self.net)


        # Add normalization layers (for vision dataset), additional normalization layer in front of network
        norm_layer = get_normalize_layer(conf.args.dataset)
        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)

        # Parallelization
        if conf.args.parallel and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net.to(device)


        ################## Init Criterions, Optimizers, Schedulers ###################
        if conf.args.method == 'Src':
            if conf.args.dataset in ['cifar10', 'cifar100', 'harth', 'reallifehar', 'extrasensory', 'svhn', 'cmnist',
                                     'rmnist', 'mnist']:
                self.optimizer = torch.optim.SGD(
                    self.net.parameters(),
                    conf.args.opt['learning_rate'],
                    momentum=conf.args.opt['momentum'],
                    weight_decay=conf.args.opt['weight_decay'],
                    nesterov=True)

                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=conf.args.epoch * len(
                    self.source_dataloader['train']))

            elif conf.args.dataset in ['tinyimagenet', 'imagenet']:
                self.optimizer = torch.optim.SGD(
                    self.net.parameters(),
                    conf.args.opt['learning_rate'],
                    momentum=conf.args.opt['momentum'],
                    weight_decay=conf.args.opt['weight_decay'],
                    nesterov=True)

            # The NLP datasets that we are interested in
            elif conf.args.dataset in ['imdb', 'sst-2', 'finefood', 'tomatoes']:
                self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                           weight_decay=conf.args.opt['weight_decay'], eps=conf.args.opt['eps'])

            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                            weight_decay=conf.args.opt['weight_decay'])

        else:
            if conf.args.method == 'TENT' and conf.args.dataset == 'imagenet': # TENT use SGD for imagenet
                self.optimizer = optim.SGD(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                            weight_decay=conf.args.opt['weight_decay'])
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                    weight_decay=conf.args.opt['weight_decay'])

        self.class_criterion = nn.CrossEntropyLoss() # already contains softmax, do not include softmax in the layer
        self.freeze_layers()  # this will call overriden method


        ################## Set Memory Management Techniques ###################
        if conf.args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'CBRS':
            self.mem = memory.CBRS_debug(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'Reservoir':
            self.mem = memory.Reservoir(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'Diversity':
            self.mem = memory.Diversity(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'CBFIFO':
            self.mem = memory.CBFIFO_debug(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'CBReservoir':
            self.mem = memory.CBReservoir_debug(capacity=conf.args.memory_size)

        self.json = {}
        self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]


        # add hooks to BN layers
        # if conf.args.log_bn_stats:
        #     num_bn_layers = 0
        #     for layer in self.net.modules():
        #         if isinstance(layer, torch.nn.BatchNorm1d):
        #             num_bn_layers += 1
        #     self.hook_logger = LogBNStats(self.write_path, num_bn_layers)
        #     for layer in self.net.modules():
        #         if isinstance(layer, torch.nn.BatchNorm1d):
        #             print("This may batchnorms")
        #             layer.register_forward_hook(self.hook_logger)


    def freeze_bn(self): #TODO: what is this?
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_layers(self):
        if 'FT_FC' in conf.args.method:  # transfer learning
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def target_data_processing(self):

        features = []
        cl_labels = []
        do_labels = []

        for b_i, data in enumerate(self.target_dataloader['train']): #must be loaded from dataloader, due to transform in the __getitem__()
            feat, cl, dl = data

            # convert a batch of tensors to list, and then append to our list one by one
            features.extend(torch.unbind(feat, dim=0))
            cl_labels.extend(torch.unbind(cl, dim=0))
            do_labels.extend(torch.unbind(dl, dim=0))

        tmp = list(zip(features, cl_labels, do_labels))

        features, cl_labels, do_labels = zip(*tmp)
        features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

        num_class = conf.args.opt['num_class']

        result_feats = []
        result_cl_labels = []
        result_do_labels = []

        # real distribution
        if self.tgt_train_dist == 0:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # random distribution
        if self.tgt_train_dist == 1:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = np.random.randint(len(features))
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # dirichlet distribution
        elif self.tgt_train_dist == 4:

            dirichlet_numchunks = conf.args.opt['num_class']

            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
            min_size = -1
            N = len(features)
            min_size_thresh = 10 #if conf.args.dataset in ['tinyimagenet'] else 10
            while min_size < min_size_thresh:  # prevent any chunk having too less data
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [[] for _ in range(dirichlet_numchunks)] # contains data per each class
                for k in range(num_class):
                    cl_labels_np = torch.Tensor(cl_labels).numpy()
                    idx_k = np.where(cl_labels_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(conf.args.dirichlet_beta, dirichlet_numchunks))

                    # balance
                    proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                            zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    # store class-wise data
                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []

            # create temporally correlated toy dataset by shuffling classes
            for chunk in idx_batch_cls:
                cls_seq = list(range(num_class))
                np.random.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    result_feats.extend([features[i] for i in idx])
                    result_cl_labels.extend([cl_labels[i] for i in idx])
                    result_do_labels.extend([do_labels[i] for i in idx])
                    sequence_stats.extend(list(np.repeat(cls, len(idx))))

            # log statistics
            # log_dirichlet_data_stats(dirichlet_numchunks, cl_labels_np, idx_batch, idx_batch_cls)

            # trim data if num_sample is smaller than the original data size
            num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
            result_feats = result_feats[:num_samples]
            result_cl_labels = result_cl_labels[:num_samples]
            result_do_labels = result_do_labels[:num_samples]

        # TODO: manage num_target_train_set..
        # if conf.args.online:
        remainder = len(result_feats) % conf.args.update_every_x  # drop leftover samples
        if remainder == 0:
            pass
        else:
            result_feats = result_feats[:-remainder]
            result_cl_labels = result_cl_labels[:-remainder]
            result_do_labels = result_do_labels[:-remainder]

        try:
            self.target_train_set = (torch.stack(result_feats),
                                     torch.stack(result_cl_labels),
                                     torch.stack(result_do_labels))
        except:
            self.target_train_set = (torch.stack(result_feats),
                                     result_cl_labels,
                                     torch.stack(result_do_labels))

        # self.target_support_remaining_set = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))


    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):

        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net

        if isinstance(net, nn.Sequential):
            if isinstance(net[0], NormalizeLayer):
                cp = net[1]
        else:
            cp = net


    def load_checkpoint(self, checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
        try:
            checkpoint = checkpoint_dict['state_dict']
        except KeyError:
            checkpoint = checkpoint_dict


        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0], NormalizeLayer):
                self.net[1].load_state_dict(checkpoint, strict=True)
        else:
            self.net.load_state_dict(checkpoint, strict=True)

        # # https://discuss.pytorch.org/t/runtimeerror-error-s-in-loading-state-dict-for-dataparallel-missing-key-s-in-state-dict/31725/6
        # self.net.load_state_dict(checkpoint, strict=True)
        self.net.to(device)

    def get_loss_and_confusion_matrix(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)

        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm, preds_of_data

    def get_loss_cm_error(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)
        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        assert (len(label) == len(pred_label))
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        errors = [0 if label[i] == pred_label[i] else 1 for i in range(len(label))]
        return loss_of_data, cm, errors

    def log_loss_results(self, condition, epoch, loss_avg):

        # print loss
        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, loss_avg
        ))

        # self.tensorboard.log_scalar(condition + '/loss_sum', loss_avg, epoch)


        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        assert (condition in ['valid', 'test'])
        # assert (suffix in ['labeled', 'unlabeled', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)
        # self.tensorboard.log_scalar(condition + '/' + 'accuracy_class_' + suffix, class_accuracy, epoch)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))
        # self.tensorboard.log_confusion_matrix(condition + '_accuracy_class_' + suffix, cm_class,
        #                                       conf.args.opt['classes'], epoch)

        return class_accuracy

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.net.train()
        class_loss_sum = 0.0
        total_iter = 0

        from utils.util_functions import print_summary
        print_summary(self.net)

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter

            if conf.args.log_percentile:
                assert conf.args.epoch == 1
                self.net.eval()

            for batch_idx, labeled_data in tqdm.tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                feats, cls, _ = labeled_data
                feats, cls = feats.to(device), cls.to(device)
                if conf.args.dataset in ['imdb', 'sst-2', 'finefood', 'tomatoes']:
                    cls = cls.squeeze(1) # (B)

                # compute the feature
                preds = self.net(feats) # (B, C)
                class_loss = self.class_criterion(preds, cls)
                class_loss_sum += float(class_loss * feats.size(0))

                # reset optimizer gradient
                self.optimizer.zero_grad()

                # backpropagation of loss
                class_loss.backward()

                # take gradient step
                self.optimizer.step()

                # take scheduler step
                if conf.args.dataset in ['cifar10', 'cifar100', 'harth', 'reallifehar', 'extrasensory']:
                    self.scheduler.step()

        ######################## LOGGING #######################

        avg_loss = class_loss_sum / total_iter
        self.log_loss_results('train', epoch=epoch, loss_avg=avg_loss)
        return avg_loss

    def train_online(self, current_num_sample):

        """
        Train the model
        """
        raise NotImplementedError  # training Src with online is currently not enabled.

        # TRAINED = 0
        # SKIPPED = 1
        # FINISHED = 2
        #
        # if not hasattr(self, 'previous_train_loss'):
        #     self.previous_train_loss = 0
        #
        # if current_num_sample > len(self.target_train_set[0]):
        #     return FINISHED
        #
        # # Add a sample
        # feats, cls, dls = self.target_train_set
        # current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        # self.mem.add_instance(current_sample)
        # self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])
        #
        # if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
        #     if not (current_num_sample == len(self.target_train_set[
        #                                           0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data
        #
        #         self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
        #         return SKIPPED
        #
        # # setup models
        #
        # self.net.train()
        #
        # class_loss_sum = 0.0
        # total_iter = 0
        #
        # feats, cls, dls = self.mem.get_memory()
        # feats, cls, dls = torch.stack(feats), torch.stack(cls), torch.stack(dls)
        # print(len(feats))
        #
        # if len(feats) == 1:  # avoid BN error
        #     self.feature_extractor.eval()
        #     self.class_classifier.eval()
        #
        # dataset = torch.utils.data.TensorDataset(feats, cls, dls)
        # data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
        #                          shuffle=True,
        #                          drop_last=False, pin_memory=False)
        # num_iter = len(data_loader)
        #
        # for e in range(conf.args.epoch):
        #
        #     total_iter += num_iter
        #
        #     for batch_idx, labeled_data in enumerate(data_loader):
        #         feats, cls, dls = labeled_data
        #         feats, cls = feats.to(device), cls.to(device)
        #
        #         feature_of_labeled_data = self.feature_extractor(feats)
        #         # compute the class loss of feature_of_labeled_data
        #         class_loss, _, _ = self.get_loss_and_confusion_matrix(self.class_classifier,
        #                                                               self.class_criterion,
        #                                                               feature_of_labeled_data,
        #                                                               cls)
        #
        #         class_loss_sum += float(class_loss * feats.size(0))
        #         self.optimizer.zero_grad()
        #         class_loss.backward()
        #         self.optimizer.step()
        #
        # self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=class_loss_sum / total_iter)
        # avg_loss = class_loss_sum / total_iter
        # self.previous_train_loss = avg_loss
        #
        # return TRAINED

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')
        # self.tensorboard.log_scalar(condition + '/' + name, value, epoch)

    def draw_tsne(self, epoch):
        feats, cls, _ = self.target_train_set
        feats, cls = feats.to(device), cls.to(device)
        # compute the feature
        feature_of_labeled_data = self.feature_extractor(feats)

        tsne = TSNE(n_components=2, verbose=1, random_state=conf.args.seed)
        _, cls, _ = self.target_train_set
        z = tsne.fit_transform(feature_of_labeled_data.cpu().detach().numpy())
        df = pd.DataFrame()
        df["y"] = cls
        df["d1"] = z[:, 0]
        df["d2"] = z[:, 1]

        sns.scatterplot(x="d1", y="d2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", conf.args.opt['num_class']),
                        data=df).set(title=f'Epoch:{epoch}')

        # self.tensorboard.log_tsne("Test" + '_tsne', global_step=epoch)

    def evaluation(self, epoch, condition):

        #########################################################################################################
        ##############################----- evaluation with target data -----####################################
        #########################################################################################################

        self.net.eval()

        with torch.no_grad():
            inputs, cls, dls = self.target_train_set
            tgt_inputs = inputs.to(device)
            tgt_cls = cls.to(device)
            if conf.args.dataset in ['imdb', 'sst-2', 'finefood', 'tomatoes']:
                tgt_cls = tgt_cls.squeeze(1)  # (B)

            preds = self.net(tgt_inputs)

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(preds, tgt_cls)
            y_pred = preds.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(tgt_cls.cpu(), y_pred.cpu(), labels=labels)


        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, class_loss_of_test_data
        ))
        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))
        # self.tensorboard.log_confusion_matrix(condition + '_accuracy_class_' + 'test', class_cm_test_data,
        #                                       conf.args.opt['classes'], epoch)

        self.logger('accuracy', class_accuracy, epoch, condition)
        self.logger('loss', class_loss_of_test_data, epoch, condition)

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def evaluation_online(self, epoch, condition, current_samples):
        #########################################################################################################
        ##############################----- evaluation with target data -----####################################
        #########################################################################################################
        # evaluation is done as list

        self.net.eval()

        with torch.no_grad():

            # extract each from list of current_sample
            features, cl_labels, do_labels = current_samples

            feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
            feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

            if conf.args.method == 'T3A':
                z = self.featurizer(feats)

                if conf.args.model == 'wideresnet28-10':  # rest operations not in the model.modules()
                    z = torch.nn.functional.avg_pool2d(z, 8)
                    z = z.view(-1, 640)

                y_pred = self.batch_evaluation(z)

            elif conf.args.method == 'LAME':
                y_pred = self.batch_evaluation(feats).argmax(-1)

            elif conf.args.method == 'COTTA':
                x = feats
                anchor_prob = torch.nn.functional.softmax(self.net_anchor(x), dim=1).max(1)[0]
                standard_ema = self.net_ema(x)

                N = 32
                outputs_emas = []

                # Threshold choice discussed in supplementary
                # enable data augmentation for vision datasets
                if anchor_prob.mean(0) < self.ap:
                    for i in range(N):
                        outputs_ = self.net_ema(self.transform(x)).detach()
                        outputs_emas.append(outputs_)
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema
                y_pred=outputs_ema
                y_pred = y_pred.max(1, keepdim=False)[1]

            else:

                y_pred = self.net(feats)
                y_pred = y_pred.max(1, keepdim=False)[1]

            ###################### SAVE RESULT
            # get lists from json

            try:
                true_cls_list = self.json['gt']
                pred_cls_list = self.json['pred']
                accuracy_list = self.json['accuracy']
                f1_macro_list = self.json['f1_macro']
                distance_l2_list = self.json['distance_l2']
            except KeyError:
                true_cls_list = []
                pred_cls_list = []
                accuracy_list = []
                f1_macro_list = []
                distance_l2_list = []

            # append values to lists
            true_cls_list += [int(c) for c in cl_labels]
            pred_cls_list += [int(c) for c in y_pred.tolist()]
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) / float(
                len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)
            f1_macro_list.append(f1_score(true_cls_list, pred_cls_list,
                                          average='macro'))

            self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

            # epoch: 1~len(self.target_train_set[0])
            progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):  # consider a batch input
                if i in progress_checkpoint:
                    print(
                        f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

            # update self.json file
            self.json = {
                'gt': true_cls_list,
                'pred': pred_cls_list,
                'accuracy': accuracy_list,
                'f1_macro': f1_macro_list,
                'distance_l2': distance_l2_list,
            }


    def dump_eval_online_result(self, is_train_offline=False):

        if is_train_offline:

            feats, cls, dls = self.target_train_set

            for num_sample in range(0, len(feats), conf.args.opt['batch_size']):
                current_sample = feats[num_sample:num_sample+conf.args.opt['batch_size']], cls[num_sample:num_sample+conf.args.opt['batch_size']], dls[num_sample:num_sample+conf.args.opt['batch_size']]
                self.evaluation_online(num_sample + conf.args.opt['batch_size'], '',
                                       [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])

        # logging json files
        json_file = open(self.write_path + 'online_eval.json', 'w')
        json_subsample = {key: self.json[key] for key in self.json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

        if is_train_offline: #avoid accumulating previous results
            self.json = {
                'gt': [],
                'pred': [],
                'accuracy': [],
                'f1_macro': [],
                'distance_l2': [],
            }


    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """

        #### for test data
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss
