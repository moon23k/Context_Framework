import torch, copy
import torch.nn as nn
from transformers import BertModel
from collections import namedtuple




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_name)
        pos_embeddings = nn.Embedding(config.max_pos, config.hidden_size)
        pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
        pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(config.max_len-512,1)
        self.bert.model.embeddings.position_embeddings = pos_embeddings


    def forward(self, x, m):
        return self.bert(x, m).pooler_output


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
    def __init__(self, config):
        super(FineModel, self).__init__()
        
        self.device = config.device
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
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

    def init_weights(self):
        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


        #Init Generator Weights
        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()  


    def weight_sharing(self):
        self.decoder.embeddings.weight = copy.deepcopy(self.encoder.embeddings.word_embeddings.weight)  
        self.generator[0].weight = self.dec_embeddings.weight      
        