import torch, copy
import torch.nn as nn
from transformers import BertModel
from collections import namedtuple



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        return 


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        return 


class Encoder(nn.Module):
    def __init__(self, config, bert):
        super(Encoder, self).__init__()
        self.bert = bert
        self.layers = clones()


    def forward(self, input_ids, token_type_ids, attention_mask):
        return 


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.n_layers = config.n_layers
        self.embeddings = nn.Embedding()
        self.layers = nn.TransformerDecoderLayer()


    def forward(self, x, m):
        for layer in self.layers:
            x = layer(x, m)
        
        return x



class FuseModel(nn.Module):
    def __init__(self, config, bert, bert_embeddings):
        super(FineModel, self).__init__()
        
        self.device = config.device
        self.encoder = Encoder(config, bert)
        self.decoder = Decoder(config, bert, bert_embeddings)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.init_weights()
        self.weight_sharing()

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, 
                                             label_smoothing=0.1).to(self.device)
        self.outputs = namedtuple('outputs', ('logits', 'loss'))


    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        
        memory = self.bert(src, segs, mask_src)
        dec_out = self.decoder(trg, memory, )
        logits = self.generator(dec_out)
        loss = self.criterion(logits.view(-1, self.vocab_size), 
                              labels[:, 1:].contiguous().view(-1))

        return self.outputs(logits, loss)

