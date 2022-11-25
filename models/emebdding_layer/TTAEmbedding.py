"""
imported from https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
"""

import torch
import torch.nn as nn
import conf
from utils.util_functions import get_max_position_embeddings

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

class TTAEmbedding(nn.Module):
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
        super(TTAEmbedding, self).__init__()

        self.wte = wte
        self.n_tokens =n_tokens

        self.max_position_emebeddings, self.embed_tensor_size = get_max_position_embeddings()
        self.output_size = self.wte.weight.size()[1]

        self.new_linear = torch.nn.Parameter(
            self.initialize_tensor(torch.randn(self.n_tokens, self.max_position_emebeddings)))
        self.new_bias = torch.nn.Parameter(
            self.initialize_tensor(torch.randn(self.n_tokens)))


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
        input_embedding = self.wte(tokens[:, :self.max_position_emebeddings - self.n_tokens])

        learned_embedding_input = self.wte(tokens).transpose(2,1).to(device)
        learned_embedding = torch.nn.functional.linear(learned_embedding_input,self.new_linear, self.new_bias).view(-1, self.n_tokens,self.output_size)
        learned_embedding = torch.nn.functional.instance_norm(learned_embedding)

        return torch.cat([learned_embedding, input_embedding], 1)