import conf
import torch
import torch.nn as nn
import BaseTransformer
import models.emebdding_layer.SoftEmbedding
from transformers import (
    BartForConditionalGeneration,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)

dimension_dictionary = {
    'bert': 768,
    'distilbert': 768,
    'bert-tiny': 128,
    'bert-mini': 256,
    'bert-small': 512,
    'bert-medium': 512,
    'mobilebert': 512,
}

class EmbeddedNet(BaseTransformer):

    def __init__(self, model_name, mode='base', num_tokens=20, initialize_from_vocab=True):
        super(EmbeddedNet, self).__init__()

        self.softembedding = models.emebdding_layer.SoftEmbedding(
            self.backbone.get_input_embeddings(),
            n_tokens=num_tokens,
            initialize_from_vocab=initialize_from_vocab
        )

        self.backbone.set_input_embeddings(self.softembedding)


    def forward(self, x):
        if 'bert' in self.model_name:
            attention_mask = (x>0).float() # 0 is the pad_token for BERT family

            if self.model_name in ['bert', 'bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'mobilebert']:
                out_h, out_p = self.backbone(x, attention_mask, return_dict=False)  # hidden, pooled
                out_p = self.dropout(out_p)
                out_cls = self.net_cls(out_p)
            # https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/distilbert/modeling_distilbert.py#L689
            elif self.model_name in ['distilbert']:
                out_h = self.backbone(x, attention_mask)[0]  # hidden state. (bs, seq_len, dim)
                out_p = out_h[:, 0]  # (bs, dim)
                out_p = self.dense(out_p)  # (bs, dim)
                out_p = torch.nn.ReLU()(out_p)  # (bs, dim)
                out_p = self.dropout(out_p)  # (bs, dim)
                out_cls = self.net_cls(out_p)  # (bs, num_labels) # TODO: include self.dense?

            return out_cls

    def get_feature(self, x): # used for LAME, which replaces the final classification layer
        if 'bert' in self.model_name:
            attention_mask = (x > 0).float()  # 0 is the pad_token for BERT family

            if self.model_name in ['bert', 'bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'mobilebert']:
                out_h, out_p = self.backbone(x, attention_mask, return_dict=False)  # hidden, pooled
                out_p = self.dropout(out_p)
            # https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/distilbert/modeling_distilbert.py#L689
            elif self.model_name in ['distilbert']:
                out_h = self.backbone(x, attention_mask)[0]  # hidden state. (bs, seq_len, dim)
                out_p = out_h[:, 0]  # (bs, dim)
                out_p = self.dense(out_p)  # (bs, dim)
                out_p = torch.nn.ReLU()(out_p)  # (bs, dim)
                out_p = self.dropout(out_p)  # (bs, dim)

            return out_p

    def get_tokenizer(self):
        return self.tokenizer


    def get_feat_dim(self, model_name):
        return dimension_dictionary[model_name]
