import random, torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim, 
                           config.hidden_dim, 
                           config.n_layers, 
                           batch_first=True, 
                           dropout=config.dropout_ratio)
        self.dropout = nn.Dropout(config.dropout_ratio)
    
    def forward(self, x):
        x = self.dropout(self.embedding(x)) 
        _, hiddens = self.rnn(x)
        return hiddens


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(config.output_dim, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim,
                           config.hidden_dim, 
                           config.n_layers,
                           batch_first=True,
                           dropout=config.dropout_ratio)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
    
    def forward(self, x, hiddens):
        x = x.unsqueeze(1)
        x = self.dropout(self.embedding(x))

        out, hiddens = self.rnn(x, hiddens)
        out = self.fc_out(out.squeeze(1))
        return out, hiddens


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.device = config.device
        self.output_dim = config.output_dim
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if hasattr(module, 'weight') and module.weight.dim() > 1:
            nn.init.uniform_(module.weight.data, -0.08, 0.08)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        outputs = torch.ones(max_len, batch_size, self.output_dim).to(self.device)

        dec_input = trg[:, 0]
        hiddens = self.encoder(src)

        for idx in range(1, max_len):
            out, hiddens = self.decoder(dec_input, hiddens)
            outputs[idx] = out
            pred = out.argmax(-1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, idx] if teacher_force else pred

        outputs = outputs.permute(1, 0, 2)
        return outputs[:, 1:].contiguous()