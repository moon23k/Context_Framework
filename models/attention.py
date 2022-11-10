import random
import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hidden_dim = config.hidden_dim

        self.dropout = nn.Dropout(config.dropout_ratio)
        self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
        
        self.sequence_fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.context_fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.sequence_rnn = nn.GRU(config.emb_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        self.context_rnn = nn.GRU(config.hidden_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        

    def forward(self, x):
        batch_size, seq_num, seq_len = x.shape
        seq_hiddens = torch.empty(seq_num, batch_size, self.hidden_dim)        
        
        x = self.dropout(self.embedding(x))
        for i in range(seq_num):
            _, seq_hidden = self.sequence_rnn(x[:, i, :, :])
            seq_hidden = torch.cat((seq_hidden[0], seq_hidden[1]), dim=1)
            seq_hidden = torch.tanh(self.sequence_fc(seq_hidden))
            seq_hiddens[i] = seq_hidden

        out, con_hidden = self.context_rnn(seq_hiddens)
        con_hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        con_hidden = torch.tanh(self.context_fc(hidden))

        return out, con_hidden




class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attn = nn.Linear((config.hidden_dim * 3), config.hidden_dim)
        self.v = nn.Linear(config.hidden_dim, 1, bias=False)


    def forward(self, hidden, enc_out):
        batch_size, src_len, _ = enc_out.shape
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))
        attn_value = self.v(energy).squeeze(2)
        attn_value = F.softmax(attn_value, dim=1)

        return attn_value #attn_value: [batch_size, seq_len]



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.output_dim = config.output_dim
        self.attention = Attention(config)
        self.emb = nn.Embedding(config.output_dim, config.emb_dim)
        self.rnn = nn.GRU((config.hidden_dim * 2) + config.emb_dim, config.hidden_dim, batch_first=True)
        self.fc_out = nn.Linear((config.hidden_dim * 3) + config.emb_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    
    def transform_tensor(self, tensor):
        tensor = tensor.permute(1, 0, 2)
        tensor = tensor.squeeze(0)
        return tensor


    def forward(self, x, hidden, enc_out):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.emb(x))

        attn_value = self.attention(hidden, enc_out)
        attn_value = attn_value.unsqueeze(1)
        weighted = torch.bmm(attn_value, enc_out)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        out, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        hidden = hidden.squeeze(0)
        out = self.transform_tensor(out)

        assert(out == hidden).all()
        embedded = self.transform_tensor(embedded)
        weighted = self.transform_tensor(weighted)

        pred = self.fc_out(torch.cat((out, weighted, embedded), dim=1))        
        return pred, hidden.squeeze(0)


        
class Seq2SeqAttn(nn.Module):
    def __init__(self, config):
        super(Seq2SeqAttn, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.device = config.device
        self.output_dim = config.output_dim


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        outputs = torch.ones(max_len, batch_size, self.output_dim).to(self.device)

        enc_out, hidden = self.encoder(src)

        dec_input = trg[:, 0]
        for t in range(1, max_len):
            out, hidden = self.decoder(dec_input, hidden, enc_out)
            outputs[t] = out

            pred = out.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else pred
        
        outputs = outputs.permute(1, 0, 2)
        return outputs[:, 1:].contiguous()