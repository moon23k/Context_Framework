import math, torch
import torch.nn as nn
from collections import namedtuple



class PositionalEncoding(nn.Module):
	def __init__(self, config):
		super(PositionalEncoding, self).__init__()

	def forward(self, x):
		return



class Encoder(nn.Module):
	def __init__(self, config):
		super(Encoder, self).__init__()

		self.layers = nn.Module()
		self.n_layers = config.n_layers

	def forward(self, x):
		for layer in self.layers:
			x = layer(x + self.pos_emb)

		return out



class Decoder(nn.Module):
	def __init__(self, config):
		super(Decoder, self).__init__()

		self.n_layers

	def forward(self, x):
		return out



class SimpleModel(nn.Module):
	def __init__(self, config):
		super(SimpleModel, self).__init__()

		self.sent_pos_encoding = PositionalEncoding()

		self.encoder = Encoder(config)
		self.decoder = Decoder(config)
		self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
		self.dropout = nn.Dropout(config.dropout_ratio)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


	def forward(self, sent_embs, sent_masks, labels):
		local_memory += self.sent_pos_encoding(some_length)

		global_memory = self.encoder(sent_repr, some_mask)
		dec_out = self.decoder(global_memory, labels, lables_mask)

		return self.generator(dec_out)