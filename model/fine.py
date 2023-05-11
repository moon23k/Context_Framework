import copy, torch
import torch.nn as nn
from collections import namedtuple



class Decoder(nn.Module):
    def __init__(self, config, embeddings):
        super(Decoder, self).__init__()
        
        self.embeddings = embeddings
        
        layer = nn.TransformerDecoderLayer(d_model=config.hidden_dim,
                                           nhead=config.n_heads,
                                           dim_feedforward=config.pff_dim,
                                           dropout=config.dropout_ratio,
                                           batch_first=True, norm_first=True)
        
        self.decoder = nn.TransformerDecoder(decoder_layer=layer, 
                                             num_layers=config.n_layers,
                                             norm=nn.LayerNorm(config.hidden_dim))
        

    def forward(self, x, memory, x_sub_mask, x_pad_mask, m_pad_mask):
        return self.decoder(self.embeddings(x), memory, 
                            memory_key_padding_mask=m_pad_mask, 
                            tgt_key_padding_mask=x_pad_mask, 
                            tgt_mask=x_sub_mask)



class FineModel(nn.Module):
    def __init__(self, config, bert, bert_embeddings):
        super(FineModel, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.encoder = bert
        self.decoder = Decoder(config, copy.deepcopy(bert_embeddings))
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, 
                                             label_smoothing=0.1).to(self.device)
        self.outputs = namedtuple('outputs', ('logits', 'loss'))



    def forward(self, x, x_seg_mask, y):

        #shift y
        y_input = y[:, :-1]
        y_label = y[:, 1:]


        #create masks
        x_pad_mask = (x != self.pad_id).to(self.device)
        y_pad_mask = (y_input != self.pad_id).to(self.device)
        y_size = y_input.size(1)
        y_sub_mask = torch.triu(torch.full((y_size, y_size), float('-inf')), diagonal=1).to(self.device)
        

        memory = self.encoder(input_ids=x, token_type_ids=x_seg_mask, 
                              attention_mask=x_pad_mask).last_hidden_state
        
        d_out = self.decoder(x=y_input, memory=memory, 
                             x_sub_mask=y_sub_mask, 
                             x_pad_mask=y_pad_mask, 
                             m_pad_mask=x_pad_mask)
        
        logits = self.generator(d_out)
        loss = self.criterion(logits.view(-1, self.vocab_size), 
                              y_label.contiguous().view(-1))

        return self.outputs(logits, loss)
