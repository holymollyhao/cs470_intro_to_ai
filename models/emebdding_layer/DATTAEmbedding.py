"""
imported from https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
"""

import torch
import torch.nn as nn
import conf
from transformers import BertTokenizer
from utils.util_functions import get_max_position_embeddings, get_device

device = get_device()

domain_dict = {
    "sst-2": "This is a sentence from standford sentiment analysis datastet:  ",
    "imdb": "The is a movie review from imdb",
    "tomatoes": "This is a movie review from rottentomatoes",
    "finefood": "The follwing sentence is a food review:",
}

class DATTAEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 20,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True,
                 model_config: dict={'max_position_embeddings' : 512}):
        """ appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(DATTAEmbedding, self).__init__()

        self.wte = wte.to(device)

        self.tokens_domain_info = torch.tensor(BertTokenizer.from_pretrained('bert-base-uncased')\
            .encode(domain_dict[conf.args.dataset], add_special_tokens=True)).long().to(device)

        self.n_tokens =n_tokens
        self.n_tokens_domain_info = len(self.tokens_domain_info)
        self.n_tokens_embedding = self.n_tokens - self.n_tokens_domain_info


        self.max_position_emebeddings, self.embed_tensor_size = get_max_position_embeddings()
        self.output_size = self.wte.weight.size()[1]

        self.new_linear = torch.nn.Parameter(
            self.initialize_tensor(torch.randn(self.n_tokens_embedding, self.max_position_emebeddings))).to(device)
        self.new_bias = torch.nn.Parameter(
            self.initialize_tensor(torch.randn(self.n_tokens_embedding))).to(device)


    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def initialize_tensor(self,
                           size,
                           random_range: float = 0.5):
        return torch.FloatTensor(size).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, :self.max_position_emebeddings - self.n_tokens_embedding - self.n_tokens_domain_info].to(device))
        domain_embedding = self.wte(self.tokens_domain_info.to(device)).repeat(input_embedding.shape[0],1,1)

        learned_embedding_input = self.wte(tokens).transpose(2,1).to(device)
        learned_embedding = torch.nn.functional.linear(learned_embedding_input,self.new_linear, self.new_bias).view(-1, self.n_tokens_embedding,self.output_size)
        learned_embedding = torch.nn.functional.instance_norm(learned_embedding)

        return torch.cat([learned_embedding, domain_embedding, input_embedding], 1)