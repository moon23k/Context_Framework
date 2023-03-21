import torch, os
import torch.nn as nn
from model.fine import FineModel
from model.feat import FeatModel



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
    if config.strategy == 'fine':
        model = FineModel(config)

    elif config.strategy == 'feat':
        model = FeatModel(config)
        
    if config.task != 'train':
        assert os.path.exists(config.ckpt_path)
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    print(f"Trained {config.strategy.upper()} model has loaded")
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    return model.to(config.device)