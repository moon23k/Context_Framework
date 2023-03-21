import math, torch
import torch.nn as nn
from collections import namedtuple
from transformers import BertModel



class PositionalEncoding(nn.Module):
    def __init__(self, config, encoder=False, decoder=False):
        
        assert not (encoder or decoder)
        self.max_len = config.max_sents if encoder else config.max_tokens

        self.embedding = nn.Embedding(self.max_len, config.hidden_dim)
        self.register_buffer('pos_ids', torch.arange(self.max_len).expand((1, -1)))

        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        curr_len = x.size(1)

        pos_ids = self.pos_ids[:, :curr_len]
        pos_emb = self.pos_emb(pos_ids)
        
        x += pos_emb

        return self.dropout(self.layer_norm(x))



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(config, encoder=True)
        self.layers = nn.TransformerEncoderLayer(d_model=config.hidden_dim, 
        										 nhead=config.n_heads,
        										 dim_feedforward=config.pff_dim,
        										 dropout=config.dropout_ratio,
        										 batch_first=config.batch_first,
        										 norm_first=config.norm_first,
        										 activation=config.act,
        										 device=config.device)

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return x



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_id)
        self.pos_emb = PositionalEncoding(config, decoder=True)
        
        self.bert_emb = BertModel.from_pretrained(config.bert_name).embeddings.to(config.device)
        self.emb_linear = nn.Linear(config.bert_dim, config.hidden_dim)
        self.emb_dropout = nn.Dropout(config.dropout_ratio)

        self.layers = nn.TransformerEncoderLayer(d_model=config.hidden_dim, 
        	  									 nhead=config.n_heads,
        										 dim_feedforward=config.pff_dim,
        										 dropout=config.dropout_ratio,
        										 batch_first=config.batch_first,
        										 norm_first=config.norm_first,
        										 activation=config.act,
        										 device=config.device)
        

    def forward(self, x, memory, e_mask, d_mask):
        x = self.pos_emb(self.token_emb(x))

        for layer in self.layers:
            x = layer(x, memory, e_mask, d_mask)
        return x



class FeatModel(nn.Module):
    def __init__(self, config):
        super(FeatModel, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size

		self.sent_pos_encoding = PositionalEncoding(config)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, 
                                             label_smoothing=0.1).to(self.device)
        self.outputs = namedtuple('outputs', ('logits', 'loss'))


    def init_weights(self):
        
        return


    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)


    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)


    def shift_right(self, labels):
        shifted = labels.new_zeros(labels.size(0), labels.size(1)-1)
        shifted = labels[:, :-1].clone()
        #shifted[:, 0] = self.pad_id #or self.decoder_start_token_id
        return shifted



    def forward(self, sent_embs, sent_masks, labels):
    	sent_embs += self.sent_pos_encoding(sent_embs)

        shifted_labels = self.shift_right(labels)
        label_masks = self.dec_mask(shifted_labels)
        
        memory = self.encoder(sent_embs, sent_masks)
        dec_out = self.decoder(shifted_labels, memory, sent_masks, label_masks)
        
        logits = self.generator(dec_out)
        loss = self.criterion(logits.view(-1, self.vocab_size), 
                              labels[:, 1:].contiguous().view(-1))

        return self.outputs(logits, loss)



class Search:
    def __init__(self, config, model):
        self.model = model

    def greedy_search(self, input_tensor):
        return

    def beam_search(self, input_tensor):
        return