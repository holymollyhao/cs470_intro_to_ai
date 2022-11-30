# -*- coding: utf-8 -*-
import sys
import re
import argparse
import random
import numpy as np
import torch
import models
import models.BaseTransformer
import models.emebdding_layer.SoftEmbedding
import torchvision
import time
import os
import conf

from data_loader import data_loader
from utils.util_functions import *
# from utils.tensorboard_logger import Tensorboard
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():
    ######################################################################
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################### Hyperparameters #################
    # load according opt and hyperparms
    opt = load_opt(conf.args.dataset)
    if conf.args.lr:
        opt['learning_rate'] = conf.args.lr
    if conf.args.weight_decay:
        opt['weight_decay'] = conf.args.weight_decay
    conf.args.opt = opt
    ################### Load pretrained model and tokenizer #################

    model = None
    tokenizer = None

    if conf.args.model == "resnet50_scratch":
        model = torchvision.models.resnet50(pretrained=False)
    elif conf.args.model == "resnet50_pretrained":
        model = torchvision.models.resnet50(pretrained=True)
    elif conf.args.model == "resnet18":
        from models import ResNet
        model = ResNet.ResNet18()
    elif conf.args.model == "resnet18_scratch":
        model = torchvision.models.resnet18(pretrained=False)
    elif conf.args.model == "resnet18_pretrained":
        model = torchvision.models.resnet18(pretrained=True)
    elif conf.args.model == "resnet34_scratch":
        model = torchvision.models.resnet34(pretrained=False)
    elif conf.args.model == "resnet34_pretrained":
        model = torchvision.models.resnet34(pretrained=True)
    elif conf.args.model == "bert":
        model = models.BaseTransformer.BaseNet(model_name='bert')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "distilbert":
        model = models.BaseTransformer.BaseNet(model_name='distilbert')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "bert-tiny":
        model = models.BaseTransformer.BaseNet(model_name='bert-tiny')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "bert-mini":
        model = models.BaseTransformer.BaseNet(model_name='bert-mini')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "bert-small":
        model = models.BaseTransformer.BaseNet(model_name='bert-small')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "bert-medium":
        model = models.BaseTransformer.BaseNet(model_name='bert-medium')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "mobilebert":
        model = models.BaseTransformer.BaseNet(model_name='mobilebert')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "bart":
        model = models.BaseTransformer.BaseNet(model_name='bart')
        tokenizer = model.get_tokenizer()
    elif conf.args.model == "rvt":
        print('constructing rvt+ small')
        from timm.models import create_model
        model = create_model('rvt_small_plus', opt=opt).to(device)
        print(model)

    # config = AutoConfig.from_pretrained(config_name, cache_dir=conf.args.cache_dir)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=conf.args.cache_dir)

    result_path, checkpoint_path, log_path = get_path()
    # tensorboard = Tensorboard(log_path)

    ################### Load method #################

    learner_method = None

    if conf.args.method == 'Src':
        from learner.dnn import DNN
        learner_method = DNN
    elif conf.args.method == 'ln_tent':
        from learner.ln_tent import LN_TENT
        learner_method = LN_TENT
    elif conf.args.method == 'ttaprompttune':
        from learner.tta_prompt_tuning import TTA_Prompt_tuning
        learner_method = TTA_Prompt_tuning
    elif conf.args.method == "dattaprompttune":
        from learner.tta_domainaware_prompt_tuning import TTA_DomainAware_Prompt_tuning
        learner_method = TTA_DomainAware_Prompt_tuning
    else:
        raise NotImplementedError(
            "Please specify the method using --method"
        )


    ################### Load Dataset #################
    print('##############Source Data Loading...##############')
    source_data_loader = data_loader.domain_data_loader(conf.args.dataset, conf.args.src,
                                                        conf.args.opt['file_path'] + '_' + conf.args.model,
                                                        batch_size=conf.args.opt['batch_size'],
                                                        valid_split=0,  # to be used for the validation
                                                        test_split=0,
                                                        separate_domains=conf.args.src_sep, is_src=True,
                                                        num_source=conf.args.num_source,
                                                        tokenizer=tokenizer)

    print('##############Target Data Loading...##############')
    target_data_loader = data_loader.domain_data_loader(conf.args.dataset, conf.args.tgt,
                                                        conf.args.opt['file_path'] + '_' + conf.args.model,
                                                        batch_size=conf.args.opt['batch_size'],
                                                        valid_split=0,
                                                        test_split=0,
                                                        separate_domains=False, is_src=False,
                                                        num_source=conf.args.num_source,
                                                        tokenizer=tokenizer)

    ################### Set Learner #################
    # learner = learner_method(model, tensorboard=tensorboard, source_dataloader=source_data_loader,
    #                          target_dataloader=target_data_loader, write_path=log_path)
    learner = learner_method(model, source_dataloader=source_data_loader,
                             target_dataloader=target_data_loader, write_path=log_path)

    ################### Ensure Result Directory is Created #################
    ensure_result_directory(result_path, checkpoint_path, conf)

    ################### Training #################
    # start time
    since = time.time()
    if not conf.args.online:

        start_epoch = 1
        best_acc = -9999
        best_epoch = -1

        ## for resuming training
        if conf.args.resume:
            checkpoint_list = [i for i in os.listdir(checkpoint_path) if re.match("cp_[0-9]+.pth.tar", i)]
            checkpoint_list.sort(key=lambda x: int(re.match("cp_([0-9]+).pth.tar", x).group(1)))

            model_data_dict = torch.load(checkpoint_path + checkpoint_list[-1],
                                         map_location=f'cuda:{conf.args.gpu_idx}')
            print(f"loading parameters from {checkpoint_path + checkpoint_list[-1]}")
            learner.load_checkpoint(checkpoint_path + checkpoint_list[-1])

            start_epoch = model_data_dict['epoch'] + 1

        # actual training

        if conf.args.iter_step != 1: # when specifies iter_steps, use iteration steps
            num_steps_per_epoch = len(source_data_loader['train'])
            end_epoch = int(conf.args.iter_step / num_steps_per_epoch)
        else: # else, use epochs
            end_epoch = conf.args.epoch

        for epoch in range(start_epoch, end_epoch + 1):
            learner.train(epoch)

            if conf.args.eval_every_n != -1 and epoch % conf.args.eval_every_n == 0:
                learner.dump_eval_online_result(is_train_offline=True)  # eval with final model

            # saving checkpoint
            if conf.args.save_every_n != -1 and epoch % conf.args.save_every_n == 0:
                print(f"saving parameters at epoch {epoch}, at {checkpoint_path}")
                learner.save_checkpoint(epoch=epoch, epoch_acc=-1, best_acc=best_acc,
                                        checkpoint_path=checkpoint_path + f'cp_{epoch}.pth.tar')

        learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
        learner.dump_eval_online_result(is_train_offline=True)  # eval with final model

        # if conf.args.log_bn_stats:
        #     learner.hook_logger.dump_logbnstats_result()

        time_elapsed = time.time() - since
        print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    elif conf.args.online:

        current_num_sample = 1
        num_sample_end = conf.args.nsample
        best_acc = -9999
        best_epoch = -1

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        finished = False
        while not finished and current_num_sample < num_sample_end:

            ret_val = learner.train_online(current_num_sample)

            if ret_val == FINISHED:
                break
            elif ret_val == SKIPPED:
                pass
            elif ret_val == TRAINED:
                pass

            current_num_sample += 1

        learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
        learner.dump_eval_online_result()

        # if conf.args.log_bn_stats:
        #     learner.hook_logger.dump_logbnstats_result()

        time_elapsed = time.time() - since
        print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    # tensorboard.close()




def parse_arguments(argv):
    """Parse command line."""
    # Note that 'type=bool' args should be False in default.
    # Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ### MANDATORY ###
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to use, in []')
    parser.add_argument('--model', type=str, default='',
                        help='Model to use, in []')
    parser.add_argument('--method', type=str, default='',
                        help='specify the method name')
    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')
    parser.add_argument('--parallel', action='store_true',
                        help='allow parallel training with multi gpus')
    parser.add_argument('--tgt_train_dist', type=int, default=1,
                        help='0: real selection'
                             '1: random selection'
                             '2: sorted selection'
                             '3: uniform selection'
                             '4: dirichlet distribution'
                        )
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--use_learned_stats', action='store_true', help='Use learned stats')


    ### Optional ###
    parser.add_argument('--src', nargs='*', default=None,
                        help='Specify source domains; not passing an arg load default src domains specified in conf.py')
    parser.add_argument('--tgt', type=str, default=None,
                        help='specific target domain; give "src" if you test under src domain')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight decay to overwrite conf.py')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--epoch', type=int, default=1,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--iter_step', type=int, default=1,
                        help="How many iteration steps you would like to take")
    parser.add_argument('--log_prefix', type=str, default='',
                        help='Suffix of log file path')
    parser.add_argument('--cache_dir', type=str, default='/mnt/sting/twkim/tetra/cache/',
                        help='cache directory that stores the pretrained models')
    parser.add_argument('--src_sep', action='store_true', help='Separate domains for source')
    parser.add_argument('--src_sep_noshuffle', action='store_true', help='Separate domains for source')
    parser.add_argument('--num_source', type=int, default=100,
                        help='number of available sources')
    parser.add_argument('--cache_path', type=str, default='/mnt/sting/twkim/tetra/cache/', help='cache path')
    parser.add_argument('--nsample', type=int, default=99999,
                        help='How many samples do you want use for train')
    parser.add_argument('--log_percentile', action='store_true', help='percentile logging process')
    parser.add_argument('--validation', action='store_true', help='Use validation data instead of test data for hyperparameter tuning')



    ### Used for Test-time Adaptation ###
    parser.add_argument('--update_every_x', type=int, default=64, help='number of target samples used for every update')
    parser.add_argument('--online', action='store_true', help='training via online learning?')
    parser.add_argument('--adapt_with_ln', action='store_true', help='adapt layer normalization layer as well')

    ### Memory Type ###
    parser.add_argument('--memory_size', type=int, default=64,
                        help='number of previously trained data to be used for training')
    parser.add_argument('--memory_type', type=str, default='FIFO',
                        help='FIFO'
                             'Reservoir'
                             'Diversity'
                             'CBRS'
                        )

    ### Used for Efficient Processing ###
    parser.add_argument('--save_every_n', type=int, default=-1, help='save checkpoint every n')
    parser.add_argument('--eval_every_n', type=int, default=-1, help='eval every n during training')
    parser.add_argument('--resume', action='store_true', help='resume from saved lastest model')


    ### Used for Ours ###
    parser.add_argument('--adapt_type', type=str, default='all', help='adaptation type of online learning')
    parser.add_argument('--use_gt', action='store_true', help='if specified, use ground truth during online learning')
    parser.add_argument('--n_tokens', type=int, default=20, help='number of tokens to use')
    parser.add_argument('--no_init_from_vocab', action='store_true', help='if specified, do not initialize from vocab')
    parser.add_argument('--set_backbone_true', action='store_true', help='if specified, the backbone gradients are set to true')

    # parser.add_argument('--iabn', action='store_true', help='replace bn with iabn layer')
    # parser.add_argument('--iabn_k', type=float, default=4.0,
    #                     help='k for iabn')
    # parser.add_argument('--pretrain_wo_iabn', action='store_true', help='replace bn with iabn layer wo pretraining the model with iabn')

    return parser.parse_args()


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('Command:', end='\t')
    print(" ".join(sys.argv))
    conf.args = parse_arguments(sys.argv[1:])
    set_seed()
    main()
