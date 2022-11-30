import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from utils import memory
from utils.logging import *
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

        # if conf.args.tgt_train_dist == 0 and \
        #         conf.args.dataset in ['cifar10', 'cifar100', 'vlcs', 'officehome', 'pacs', 'tinyimagenet', 'svhn',
        #                               'visda', 'cmnist', 'rmnist', 'terra_incognita','domainnet', 'mnist', 'finefood',
        #                               'sst-2', 'imdb', 'tomatoes']:
        #     self.tgt_train_dist = 4  # Dirichlet is default for non-real-distribution data
        # else:
        #     self.tgt_train_dist = conf.args.tgt_train_dist

        self.target_data_processing()
        self.write_path = write_path

        ################## Init & prepare model###################
        self.conf_list = []
        self.net = model


        # Add normalization layers (for vision dataset), additional normalization layer in front of network
        norm_layer = get_normalize_layer(conf.args.dataset)
        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)

        # Parallelization
        if conf.args.parallel and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net.to(device)


        ################## Init Criterions, Optimizers, Schedulers ###################
        from transformers import get_scheduler
        self.optimizer = optim.AdamW(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                     weight_decay=conf.args.opt['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)


        self.class_criterion = nn.CrossEntropyLoss() # already contains softmax, do not include softmax in the layer


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

        result_feats = []
        result_cl_labels = []
        result_do_labels = []


        num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
        for _ in range(num_samples):
            tgt_idx = np.random.randint(len(features))
            result_feats.append(features.pop(tgt_idx))
            result_cl_labels.append(cl_labels.pop(tgt_idx))
            result_do_labels.append(do_labels.pop(tgt_idx))

            # trim data if num_sample is smaller than the original data size
            num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
            result_feats = result_feats[:num_samples]
            result_cl_labels = result_cl_labels[:num_samples]
            result_do_labels = result_do_labels[:num_samples]


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

        torch.save({
            'state_dict': cp.state_dict(),
            'epoch': epoch,
        }, checkpoint_path)


    def load_checkpoint(self, checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
        try:
            checkpoint = checkpoint_dict['state_dict']
        except KeyError:
            checkpoint = checkpoint_dict

        if conf.args.parallel:
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in checkpoint.items():
                if 'module' not in k:
                    k = 'module.' + k
                new_state_dict[k] = v
            checkpoint = new_state_dict


        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0], NormalizeLayer):
                self.net[1].load_state_dict(checkpoint, strict=True)
        else:
            self.net.load_state_dict(checkpoint, strict=True)

        # # https://discuss.pytorch.org/t/runtimeerror-error-s-in-loading-state-dict-for-dataparallel-missing-key-s-in-state-dict/31725/6
        # self.net.load_state_dict(checkpoint, strict=True)
        self.net.to(device)

    # loading from source, which does not have any softempbedding layer
    def load_checkpoint_naive(self, checkpoint_path=''):
        from models.BaseTransformer import BaseNet
        if checkpoint_path:
            checkpoint_dict = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
            try:
                checkpoint = checkpoint_dict['state_dict']
            except KeyError:
                checkpoint = checkpoint_dict

            temp_net = BaseNet(self.net.model_name)
            temp_net.load_state_dict(checkpoint, strict=True)
        else:
            temp_net = BaseNet(self.net.model_name)

        if conf.args.method == 'dattaprompttune':
            from models.emebdding_layer.DATTAEmbedding import DATTAEmbedding
            embedding = DATTAEmbedding(
                temp_net.get_input_embeddings(),
                n_tokens=self.n_tokens,
                initialize_from_vocab=self.initialize_from_vocab,
            )
        elif conf.args.method == 'ttaprompttune':
            from models.emebdding_layer.SoftEmbedding import SoftEmbedding
            embedding = SoftEmbedding(
                temp_net.get_input_embeddings(),
                n_tokens=self.n_tokens,
                initialize_from_vocab=self.initialize_from_vocab,
            )
        else:
            raise NotImplementedError
        temp_net.set_input_embeddings(embedding)
        self.net.load_state_dict(temp_net.state_dict(), strict=True)
        self.net.to(device)

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.net.train()
        # if conf.args.parallel:
        #     self.net.module.set_backbone_gradient(conf.args.set_backbone_true)
        # else:
        #     self.net.set_backbone_gradient(conf.args.set_backbone_true)
        class_loss_sum = 0.0
        total_iter = 0

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
                self.optimizer.step()

                # take scheduler step
                self.scheduler.step()
        # self.log_current_test_acc()

        ######################## LOGGING #######################

        print(f'class_loss_sum is : {class_loss_sum}')

        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def train_online(self, current_num_sample):

        """
        Train the model
        """
        raise NotImplementedError  # training Src with online is currently not enabled.

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


        # print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
        #     condition, epoch, class_loss_of_test_data
        # ))
        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def evaluation_online(self, epoch, condition, current_samples):
        #########################################################################################################
        ##############################----- evaluation with target data -----####################################
        #########################################################################################################
        self.net.eval()

        with torch.no_grad():

            # extract each from list of current_sample
            features, cl_labels, do_labels = current_samples

            feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
            feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)


            dataset = torch.utils.data.TensorDataset(feats, cls)
            data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                     shuffle=True,
                                     drop_last=False, pin_memory=False)
            pred_list = []
            for batch_idx, (feats, cls,) in enumerate(data_loader):
                feats = feats.to(device)
                cls = cls.to(device)

                # prediction from network
                preds_of_data = self.net(feats)
                pred_list.append(preds_of_data)

            y_pred = torch.cat(pred_list)
            y_pred = y_pred.max(1, keepdim=False)[1]

            # SAVE RESULTS
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

    def log_current_test_acc(self):
        dataset = torch.utils.data.TensorDataset(self.target_train_set[0], self.target_train_set[1])
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        pred_list = []
        for batch_idx, (feats, cls,) in enumerate(data_loader):
            feats = feats.to(device)
            cls = cls.to(device)

            # prediction from network
            preds_of_data = self.net(feats)
            pred_list.append(preds_of_data)

        y_pred = torch.cat(pred_list)
        y_pred = y_pred.max(1, keepdim=False)[1]
        acc = sum(1 for gt, pred in zip(y_pred, self.target_train_set[1]) if gt == pred) / float(
                len(y_pred)) * 100
        print(f'Current epoch test acc is: {acc}')


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
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss

    def set_gradients(self, type='all'):
        assert "prompttune" in conf.args.method
        from utils.util_functions import is_instance_of_any, set_gradients

        if type == 'all':
            set_gradients(parameter=self.net.parameters(), boolean=True)
            if conf.args.parallel:
                set_gradients(parameter=self.net.module.backbone.parameters(), boolean=False)
                set_gradients(parameter=self.net.module.backbone.get_input_embeddings().parameters(), boolean=True)
            else:
                set_gradients(parameter=self.net.backbone.parameters(), boolean=False)
                set_gradients(parameter=self.net.backbone.get_input_embeddings().parameters(), boolean=True)

            for m in self.net.modules():
                if is_instance_of_any(m, [nn.BatchNorm1d, nn.BatchNorm2d]):
                    if conf.args.use_learned_stats:
                        m.track_running_stats = True
                        m.momentum = conf.args.bn_momentum
                    else:
                        m.track_running_stats = False
                        m.running_mean = None
                        m.running_var = None

                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)

        elif type in ['bn', 'ln', 'bnln']:
            if type == 'bn':
                instance_list = [nn.BatchNorm1d, nn.BatchNorm2d]
            elif type == 'ln':
                instance_list = [nn.LayerNorm]
            else:
                instance_list = [nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm]

            for m in self.net.modules():
                if is_instance_of_any(m, instance_list):
                    if conf.args.use_learned_stats:
                        m.track_running_stats = True
                        m.momentum = conf.args.bn_momentum
                    else:
                        m.track_running_stats = False
                        m.running_mean = None
                        m.running_var = None

                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                else:
                    set_gradients(parameter=m.parameters(), boolean=False)

        elif type == 'embed':

            set_gradients(parameter=self.net.parameters(), boolean=False)
            if conf.args.parallel:
                set_gradients(parameter=self.net.module.backbone.parameters(), boolean=False)
                set_gradients(parameter=self.net.module.backbone.get_input_embeddings().parameters(), boolean=True)
            else:
                set_gradients(parameter=self.net.backbone.parameters(), boolean=False)
                set_gradients(parameter=self.net.backbone.get_input_embeddings().parameters(), boolean=True)


        elif type == 'all_ln_bn':
            set_gradients(parameter=self.net.parameters(), boolean=True)
            if conf.args.parallel:
                set_gradients(parameter=self.net.module.backbone.parameters(), boolean=False)
                set_gradients(parameter=self.net.module.backbone.get_input_embeddings().parameters(), boolean=True)
            else:
                set_gradients(parameter=self.net.backbone.parameters(), boolean=False)
                set_gradients(parameter=self.net.backbone.get_input_embeddings().parameters(), boolean=True)

            for m in self.net.modules():
                if is_instance_of_any(m, [nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm]):
                    if conf.args.use_learned_stats:
                        m.track_running_stats = True
                        m.momentum = conf.args.bn_momentum
                    else:
                        m.track_running_stats = False
                        m.running_mean = None
                        m.running_var = None

                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
        else:
            raise NotImplementedError