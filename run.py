import numpy as np
import sentencepiece as spm
import yaml, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.seq2seq import Seq2Seq
from models.attention import Seq2SeqAttn
from models.transformer import Transformer

from modules.test import Tester
from modules.train import Trainer
from modules.inference import Translator
from modules.data import load_dataloader




class Config(object):
    def __init__(self, args):    
        with open('configs/model.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[args.model]
            for p in params.items():
                setattr(self, p[0], p[1])

        self.task = args.task
        self.model_name = args.model
        
        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        self.clip = 1
        self.n_epochs = 1
        self.batch_size = 128

        if self.task != 'train':
            self.ckpt = f'ckpt/{self.model_name}.pt'

        if self.task == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            if self.scheduler == 'constant':
                self.learning_rate = 1e-3
                self.scheduler = None
            elif self.scheduler == 'noam':
                self.learning_rate = 1e-3
                self.scheduler = optim.lr_scheduler()

        if self.task == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=0.1).to(self.device)

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



def load_tokenizer(lang):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{lang}_tokenizer.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')
    return tokenizer



def load_model(config):
    if config.model_name == 'seq2seq':
        model = Seq2Seq(config)
    elif config.model_name == 'attention':
        model = Seq2SeqAttn(config)
    elif config.model_name == 'transformer':
        model = Transformer(config)

    if config.task != 'train':
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    return model.to(config.device)



def main(config):
    model = load_model(config)

    if config.task == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')        
        trainer = Trainer(model, config, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.task == 'test':
        test_dataloader = load_dataloader(config, 'test')
        trg_tokenizer = load_tokenizer('de')
        tester = Tester(config, model, test_dataloader, trg_tokenizer)
        tester.test()
    
    elif config.task == 'inference':
        src_tokenizer = load_tokenizer('en')
        trg_tokenizer = load_tokenizer('de')
        translator = Translator(model, config, src_tokenizer, trg_tokenizer)
        translator.translate()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-scheduler', required=False)
    
    args = parser.parse_args()
    assert args.task in ['train', 'test', 'inference']
    assert args.model in ['seq2seq', 'attention', 'transformer']
 
    set_seed()
    config = Config(args)
    main(config)