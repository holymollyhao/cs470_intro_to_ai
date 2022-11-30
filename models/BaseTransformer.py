import conf
import torch
import torch.nn as nn
from models.emebdding_layer import SoftEmbedding
dimension_dictionary = {
    'bert': 768,
    'distilbert': 768,
    'bert-tiny': 128,
    'bert-mini': 256,
    'bert-small': 512,
    'bert-medium': 512,
    'mobilebert': 512,
    'bart': 1024,
}

def load_backbone(name, output_attentions=False):
    if name == 'bert':
        from transformers import BertModel, BertTokenizer
        backbone = BertModel.from_pretrained('bert-base-uncased', output_attentions=output_attentions)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
    elif name == 'distilbert':
        from transformers import DistilBertModel, DistilBertTokenizer
        backbone = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=output_attentions)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer.name = 'distilbert-base-uncased'
    elif name in ['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium']:
        from transformers import AutoModel, AutoTokenizer
        backbone = AutoModel.from_pretrained(f'prajjwal1/{name}')
        tokenizer = AutoTokenizer.from_pretrained(f'prajjwal1/{name}')
        tokenizer.name = name
    elif name == 'mobilebert':
        from transformers import MobileBertModel
        from transformers import MobileBertTokenizer
        backbone = MobileBertModel.from_pretrained('google/mobilebert-uncased')
        tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
        tokenizer.name = 'mobilebert-uncased'
    elif name == 'bart':
        from transformers import  BartModel
        from transformers import BartTokenizer
        backbone = BartModel.from_pretrained("facebook/bart-large")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        tokenizer.name = 'bart'
    else:
        raise ValueError('No matching backbone network')

    return backbone, tokenizer

class BaseNet(nn.Module):

    def __init__(self, model_name='', mode='base'):
        super(BaseNet, self).__init__()
        self.model_name = model_name

        if mode == "base":
            backbone, tokenizer = load_backbone(self.model_name)
            self.backbone = backbone
            self.tokenizer = tokenizer
            self.n_classes = conf.args.opt['num_class']
            self.dropout = nn.Dropout(0.1)

            dim = self.get_feat_dim(model_name)
            self.dense_layer = nn.Linear(dim, dim)
            self.class_layer = nn.Linear(dim, self.n_classes)

    def forward(self, x):
        if 'bert' in self.model_name:
            attention_mask = (x>0).float() # 0 is the pad_token for BERT family

            if self.model_name in ['bert', 'bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'mobilebert']:
                out_h, out_p = self.backbone(x, attention_mask, return_dict=False)  # hidden, pooled
                out_p = self.dropout(out_p)
                out_cls = self.class_layer(out_p)
            elif self.model_name in ['distilbert']:
                out_h = self.backbone(x, attention_mask)[0]  # hidden state. (bs, seq_len, dim)
                out_p = out_h[:, 0]  # (bs, dim)
                out_p = self.dense_layer(out_p)  # (bs, dim)
                out_p = torch.nn.ReLU()(out_p)  # (bs, dim)
                out_p = self.dropout(out_p)  # (bs, dim)
                out_cls = self.class_layer(out_p)  # (bs, num_labels) # TODO: include self.dense?
        elif 'bart' in self.model_name:
            attention_mask = (x > 0).float()
            out_h = self.backbone(x, attention_mask)[0]  # hidden state. (bs, seq_len, dim)

            out_p = out_h[:, 0]  # (bs, dim)
            out_p = self.dense_layer(out_p)  # (bs, dim)
            out_p = torch.nn.ReLU()(out_p)  # (bs, dim)
            out_p = self.dropout(out_p)  # (bs, dim)
            out_cls = self.class_layer(out_p)  # (bs, num_labels)
        return out_cls

    def get_feature(self, x): # used for LAME, which replaces the final classification layer
        if 'bert' in self.model_name:
            attention_mask = (x > 0).float()  # 0 is the pad_token for BERT family

            if self.model_name in ['bert', 'bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'mobilebert']:
                out_h, out_p = self.backbone(x, attention_mask, return_dict=False)  # hidden, pooled
                out_p = self.dropout(out_p)
            elif self.model_name in ['distilbert']:
                out_h = self.backbone(x, attention_mask)[0]  # hidden state. (bs, seq_len, dim)
                out_p = out_h[:, 0]  # (bs, dim)
                out_p = self.dense_layer(out_p)  # (bs, dim)
                out_p = torch.nn.ReLU()(out_p)  # (bs, dim)
                out_p = self.dropout(out_p)  # (bs, dim)

            return out_p

    def get_tokenizer(self):
        return self.tokenizer


    def get_feat_dim(self, model_name):
        return dimension_dictionary[model_name]

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        return self.backbone.set_input_embeddings(embeddings)


    def get_config(self):
        return self.backbone.config

    def set_backbone_gradient(self, bool):
        for params in self.backbone.parameters():
            params.requires_grad = bool