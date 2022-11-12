import conf
from .dnn import DNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
from models.emebdding_layer import SoftEmbedding
from utils.loss_functions import *
from utils.util_functions import print_summary

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

def set_requries_grad(module, bool):
    for param in module.parameters():
        param.requires_grad = bool

# def initialize_gradient(module, bool):

class Prompt_tuning(DNN):
    def __init__(self, *args, **kwargs):
        super(Prompt_tuning, self).__init__(*args, **kwargs)

        n_tokens = 20
        initialize_from_vocab = True,
        self.n_tokens = n_tokens
        self.initialize_from_vocab = initialize_from_vocab
        # print_summary(self.net) # for debugging purpose

        # initially turn off gradients for all
        set_requries_grad(self.net, False)
        # print_summary(self.net)

        ## only can be used with huggingface transformer models
        self.soft_embedding = SoftEmbedding.SoftEmbedding(
            self.net.get_input_embeddings(),
            n_tokens=self.n_tokens,
            initialize_from_vocab = self.initialize_from_vocab
        )
        set_requries_grad(self.soft_embedding, True)

        for param in self.soft_embedding.parameters():
            param.requires_grad = True
        
        # setting the previous input_embeddings to current embedding
        self.net.set_input_embeddings(self.soft_embedding)
        # print_summary(self.net)


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
            if conf.args.dataset in ['cifar10', 'cifar100', 'harth', 'reallifehar', 'extrasensory']:
                self.scheduler.step()



    def train_online(self, current_num_sample):

        raise NotImplementedError # training Prompt tuning with online is currently not enabled.