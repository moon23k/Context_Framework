import torch, os
import torch.nn as nn
from model.simple import SimpleModel
from model.fused import FusedModel




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
    if config.model_name == 'simple':
        model = SimpleModel(config)
        model.apply(init_uniform)

    elif config.model_name == 'fused':
        model = FusedModel(config)
        model.apply(init_normal)
        
    if config.task != 'train':
        assert os.path.exists(config.ckpt_path)
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    print(f"The {config.model_name} model has loaded")
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    return model.to(config.device)