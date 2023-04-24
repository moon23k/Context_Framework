import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class AttnBlock(nn.Module):
    def __init__(self, config):
        super(AttnBlock, self).__init__()

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=config.hidden_dim, 
                                          num_heads=config.n_heads, 
                                          batch_first=True)
        self.dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, query, key, value, key_padding_mask, attn_mask=None):
        query = self.norm(query)
        
        attn_out = self.attn(query, key, value,
                             key_padding_mask=key_padding_mask,
                             attn_mask=attn_mask,
                             need_weights=False)[0]

        return self.dropout(attn_out)


class PffBlock(nn.Module):
    def __init__(self, config):
        super(PffBlock, self).__init__()

        self.activation = F.gelu
        self.norm = nn.LayerNorm(config.hidden_dim)

        self.linear1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.linear2 = nn.Linear(config.pff_dim, config.hidden_dim)

        self.dropout1 = nn.Dropout(config.dropout_ratio)
        self.dropout2 = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        x = self.linear1(self.norm(x))
        x = self.dropout1(self.activation(x))
        return self.dropout2(self.linear2(x))



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.bert_attn = AttnBlock(config)
        self.self_attn = AttnBlock(config)
        self.pff = PffBlock(config)


    def forward(self, x, bert_out, x_pad_mask):
        bert_attn_out = self.bert_attn(x, bert_out, bert_out, x_pad_mask)
        self_attn_out = self.self_attn(x, x, x, x_pad_mask)
        
        x = x + (bert_attn_out * 0.5) + (self_attn_out * 0.5)

        return x + self.pff(x)




class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        
        self.bert_attn = AttnBlock(config)
        self.self_attn = AttnBlock(config)
        self.enc_dec_attn = AttnBlock(config)
        self.pff = PffBlock(config)


    def forward(self, x, bert_out, memory, x_sub_mask, x_pad_mask, m_pad_mask):
        self_attn_out = self.self_attn(x, x, x, x_pad_mask, x_sub_mask)
        x += self_attn_out

        bert_attn_out = self.bert_attn(x, bert_out, bert_out, m_pad_mask)
        self_attn_out = self.self_attn(x, memory, memory, m_pad_mask)
        x = x + (bert_attn_out * 0.5) + (self_attn_out * 0.5)

        return x + self.pff(x)



class Encoder(nn.Module):
    def __init__(self, config, bert_embeddings):
        super(Encoder, self).__init__()

        self.embeddings = bert_embeddings
        self.layers = clones(EncoderLayer(config), config.n_layers)
        

    def forward(self, x , bert_out, x_pad_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, bert_out, x_pad_mask)
        return x



class Decoder(nn.Module):
    def __init__(self, config, embeddings):
        super(Decoder, self).__init__()

        self.embeddings = embeddings
        self.layers = clones(DecoderLayer(config), config.n_layers)

        
    def forward(self, x, bert_out, memory, x_sub_mask, x_pad_mask, m_pad_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, bert_out, memory, x_sub_mask, x_pad_mask, m_pad_mask)
        return x


class FuseModel(nn.Module):
    def __init__(self, config, bert, bert_embeddings):
        super(FuseModel, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.bert = bert
        self.encoder = Encoder(config, bert_embeddings)
        self.decoder = Decoder(config, bert_embeddings)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, 
                                             label_smoothing=0.1).to(self.device)
        
        self.outputs = namedtuple('outputs', ('logits', 'loss'))



    def forward(self, x, x_seg_mask, y):

        #shift y
        y_input = y[:, :-1]
        y_label = y[:, 1:]


        #create masks
        x_pad_mask = (x == self.pad_id).to(self.device)
        y_pad_mask = (y_input == self.pad_id).to(self.device)
        y_size = y_input.size(1)
        y_sub_mask = torch.triu(torch.full((y_size, y_size), float('-inf')), diagonal=1).to(self.device)

        bert_out = self.bert(input_ids=x, token_type_mask=x_seg_mask, 
                             attention_mask=x_pad_mask).last_hidden_state        

        memory = self.encoder(x, bert_out, x_pad_mask)
        
        d_out = self.decoder(y_input, bert_out, memory, y_sub_mask, y_pad_mask, x_pad_mask)
        
        logits = self.generator(d_out)
        
        loss = self.criterion(logits.view(-1, self.vocab_size), 
                              y_label.contiguous().view(-1))

        return self.outputs(logits, loss)
