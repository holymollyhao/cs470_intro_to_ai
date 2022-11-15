"""
imported from https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
"""

import torch
import torch.nn as nn
import conf
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class TTAEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
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
        super(TTAEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))
        print(f'wte size is : {self.wte.weight.size()}')
        # torch.Size([16, 512 max_positional_embedding]) (input token size)
        # torch.Size([16 batch size, 20 num tokens, 768 output size])
        print('max positional embedding is : #################################')
        print(model_config)
        self.max_position_emebeddings = model_config.max_position_embeddings
        self.interm_size = 200
        self.output_size = self.wte.weight.size()[1]
        self.learned_embedding_net = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.max_position_emebeddings, self.interm_size),
            nn.BatchNorm1d(self.interm_size),
            nn.Linear(self.interm_size, self.n_tokens * self.output_size),
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
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding_net(tokens.to(torch.float32).to(device)).view(-1, self.n_tokens, self.output_size)
        return torch.cat([learned_embedding, input_embedding], 1)