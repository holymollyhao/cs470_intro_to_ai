"""
imported from https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
"""

import torch
import torch.nn as nn
import conf
from transformers import BertTokenizer
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

domain_dict = {
    "sst-2": "sentimental analysis",
    "imdb": "the review of the movie is :",
    "tomatoes": "movie review analysis",
    "finefood": "food review analysis",
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

        self.wte = wte

        self.tokens_domain_info = torch.tensor(BertTokenizer.from_pretrained('bert-base-uncased')\
            .encode(domain_dict[conf.args.dataset], add_special_tokens=True)).long()

        self.n_tokens_domain_info = len(self.tokens_domain_info)
        self.n_tokens_embedding = n_tokens - len(self.n_tokens_domaininfo)

        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  self.n_tokens_embedding,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

        self.max_position_emebeddings = model_config.max_position_embeddings
        self.interm_size = 200
        self.output_size = self.wte.weight.size()[1]
        self.learned_embedding_net = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.max_position_emebeddings, self.interm_size),
            nn.BatchNorm1d(self.interm_size),
            nn.Linear(self.interm_size, self.n_tokens_embedding * self.output_size),
        )
        self.learned_embedding_net.to(device)

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

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, :len(tokens) - self.n_tokens_embedding - self.n_tokens_domain_info])
        domain_embedding = self.wte(self.tokens_domain_info)

        input_to_learned_embedding = tokens.to(torch.float32).to(device)
        input_to_learned_embedding -= input_to_learned_embedding.min(1, keepdim=True)[0]
        input_to_learned_embedding /= input_to_learned_embedding.max(1, keepdim=True)[0]
        learned_embedding = self.learned_embedding_net(input_to_learned_embedding).view(-1, self.n_tokens_embedding, self.output_size)


        return torch.cat([domain_embedding, learned_embedding, input_embedding], 1)