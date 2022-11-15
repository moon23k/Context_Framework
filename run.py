import numpy as np
import sentencepiece as spm
import os, yaml, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from modules.model import load_model
from modules.data import load_dataloader

from modules.test import Tester
from modules.train import Trainer
from modules.inference import Summarizer



class Config(object):
    def __init__(self, args):    
        with open('configs/model.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[args.model]
            for p in params.items():
                setattr(self, p[0], p[1])

        self.task = args.task
        self.model_name = args.model
        
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.ckpt_path = f"ckpt/{self.model_name}.pt"

        if self.task == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')
        else:
            self.search = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

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



def load_tokenizer():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')
    return tokenizer



def tranaslate(config, model, tokenizer):
    print('Type "quit" to terminate Translation')
    
    while True:
        user_input = input('Please Type Text >> ')
        if user_input.lower() == 'quit':
            print('--- Terminate the Translation ---')
            print('-' * 30)
            break

        src = config.src_tokenizer.Encode(user_input)
        src = torch.LongTensor(src).unsqueeze(0).to(config.device)

        if config.search == 'beam':
            pred_seq = config.search.beam_search(src)
        elif config.search == 'greedy':
            pred_seq = config.search.greedy_search(src)

        print(f" Original  Sequence: {user_input}")
        print(f'Translated Sequence: {tokenizer.Decode(pred_seq)}\n')



def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)

    if config.task == 'train': 
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.task == 'test':
        tokenizer = load_tokenizer()
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, test_dataloader, tokenizer)
        tester.test()
        tester.inference_test()
    
    elif config.task == 'inference':
        tokenizer = load_tokenizer()
        summarizer = Summarizer(config, model, tokenizer)
        summarizer.summarize()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task in ['train', 'test', 'inference']
    assert args.model in ['seq2seq', 'attention', 'transformer']

    if args.task == 'inference':
        assert args.search in ['greedy', 'beam']

    main(args)