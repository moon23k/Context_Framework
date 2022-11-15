import torch, os
import torch.nn as nn
from models.seq2seq import Seq2Seq
from models.attention import Seq2SeqAttn
from models.transformer import Transformer




def init_uniform(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)



def init_normal(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)



def init_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)



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



def load_model(config):
    if config.model_name == 'seq2seq':
        model = Seq2Seq(config)
        model.apply(init_uniform)

    elif config.model_name == 'attention':
        model = Seq2SeqAttn(config)
        model.apply(init_normal)

    elif config.model_name == 'transformer':
        model = Transformer(config)
        model.apply(init_xavier)
        
    if config.task != 'train':
        assert os.path.exists(config.ckpt_path)
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    print(f"The {config.model_name} model has loaded")
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    return model.to(config.device)