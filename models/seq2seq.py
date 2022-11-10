import random, torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hidden_dim = config.hidden_dim

        self.dropout = nn.Dropout(config.dropout_ratio)
        self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
        
        self.sequence_rnn = nn.LSTM(config.emb_dim,
                                    config.hidden_dim,
                                    config.n_layers,
                                    batch_first=True,
                                    dropout=config.dropout_ratio)
        
        self.context_rnn = nn.LSTM(config.hidden_dim,
                                   config.hidden_dim,
                                   config.n_layers,
                                   batch_first=True,
                                   dropout=config.dropout_ratio)        
    
    def forward(self, x):
        batch_size, seq_num, seq_len, _ = x.shape
        seq_hiddens = torch.empty(seq_num, batch_size, self.hidden_dim) 
        seq_cells = torch.empty(seq_num, batch_size, self.hidden_dim)

        x = self.dropout(self.embedding(x))
        
        for i in range(seq_num):
            _, hidden, cell = self.rnn(:, i, :, :)
            seq_hiddens[i] = hidden.squeeze(0)
            seq_cells[i] = cell.squeeze(0)

        _, con_hiddens = self.context_rnn(hiddens.permute(1, 0, 2), 
                                          cells.permute(1, 0, 2))
        
        return con_hiddens



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