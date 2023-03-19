import torch
import torch.nn as nn



class FineModel(nn.Module):
    def __init__(self, config):
        super(FineModel, self).__init__()

        self.device = config.device
        self.bert = BertModel.from_pretrained(config.bert_name).to(self.device)

        #for seq length more than 512
        my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
        my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
        my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
        self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)

        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        #디코더 초기화
        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        #generator 초기화
        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()

        #기존 BERT의 EMB를 활용하기 위한 코드
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
        self.decoder.embeddings = tgt_embeddings
        self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None


    def init_weights(self):
        return