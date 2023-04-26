import torch, os
import torch.nn as nn
from model.fine import FineModel
from model.fuse import FuseModel
from transformers import BertModel



def init_weights(model):
    if isinstance(model, FineModel):
        except_list = ['encoder', 'embeddings', 'norm', 'bias']
    elif isinstance(model, FuseModel):
        except_list = ['bert', 'embeddings', 'norm', 'bias']
    
    for name, param in model.named_parameters():
        if any([x in name for x in except_list]):
            continue
        nn.init.xavier_uniform_(param)    


def print_model_desc(model):
    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params

    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")



def load_bert(config):
    bert = BertModel.from_pretrained(config.bert_name)
    bert_embeddings = bert.embeddings

    max_len = config.model_max_length
    temp_emb = nn.Embedding(max_len, 512)
    temp_emb.weight.data[:512] = bert.embeddings.position_embeddings.weight.data
    temp_emb.weight.data[512:] = bert.embeddings.position_embeddings.weight.data[-1][None,:].repeat(max_len-512, 1)

    bert.embeddings.position_embeddings = temp_emb

    bert.config.max_position_embeddings = max_len
    bert.embeddings.position_ids = torch.arange(max_len).expand((1, -1))
    bert.embeddings.token_type_ids = torch.zeros(max_len, dtype=torch.long).expand((1, -1))
    
    return bert, bert_embeddings
    


def load_model(config):
    #Load bert and embeddings
    bert, bert_embeddings = load_bert(config)

    #Load Initial Model
    if config.strategy == 'fine':
        model = FineModel(config, bert, bert_embeddings)
    elif config.strategy == 'fuse':
        model = FuseModel(config, bert, bert_embeddings)

    init_weights(model)
    print(f'{config.strategy.upper()} Model has Loaded')

    if config.mode != 'train':
        ckpt = config.ckpt
        assert os.path.exists(ckpt)
        model_state = torch.torch.load(ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Model States has loaded from {ckpt}")

    print_model_desc(model)
    return model.to(config.device)