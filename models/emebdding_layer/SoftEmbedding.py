"""
Built upon https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
"""

import torch
import torch.nn as nn
import conf


domain_dict = {
    "sst-2": "sentiment analysis:",
    "imdb": "movie review:",
    "tomatoes": "movie review:",
    "finefood": "food review:",
}
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens

        if conf.args.model == 'bert':
            from transformers import BertTokenizer
            self.tokens_domain_info = torch.tensor(BertTokenizer.from_pretrained('bert-base-uncased')\
                .encode(domain_dict[conf.args.dataset], add_special_tokens=True)).long().to(device)
        elif conf.args.model == 'distilbert':
            from transformers import DistilBertTokenizer
            self.tokens_domain_info = torch.tensor(DistilBertTokenizer.from_pretrained('bert-base-uncased') \
                                                   .encode(domain_dict[conf.args.dataset],
                                                           add_special_tokens=True)).long().to(device)
        else:
            raise NotImplementedError

        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(n_tokens))

    def initialize_embedding(self,
                             n_tokens: int = 10):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """

        token_list = []
        for token_idx in self.tokens_domain_info:
            token_list.append(self.wte.weight[token_idx].clone().detach())

        init_len = len(token_list)
        for token_idx in range(n_tokens - init_len):
            token_list.append(self.wte.weight[token_idx].clone().detach())

        return torch.stack(token_list)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        ## text concatenation within the forward
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)