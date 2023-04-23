import torch, os
import torch.nn as nn
from model.fine import FineModel
from model.fuse import FuseModel



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
    bert.embeddings.position_ids = torch.arange(config.model_max_length).expand((1, -1))
    bert.embeddings.token_type_ids = torch.zeros(config.model_max_length).expand((1, -1))
    
    orig_pos_emb = bert.embeddings.position_embeddings.weight

    bert.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))
    bert.config.max_position_embeddings = config.model_max_length
    
    return bert
    


def load_model(config):
    #Load Initial Model
    if config.stratefy == 'feat':
        model = FeatModel(config)
    elif config.strategy == 'fuse':
        model = FuseModel(config)

    print(f'{config.strategy.upper()} Model has Loaded')

    if config.mode != 'train':
        ckpt = config.ckpt
        assert os.path.exists(ckpt)
        model_state = torch.torch.load(ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Model States has loaded from {ckpt}")

    print_model_desc(model)
    return model.to(config.device)