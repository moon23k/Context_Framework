import os, argparse, torch
from module.train import Trainer
from transformers import set_seed, AutoTokenizer, AutoModel




class Config(object):
    def __init__(self, args):    

        self.mode = args.mode

        self.clip = 1
        self.lr = 5e-4
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def main(args):
    set_seed(42)
    config = Config(args)


	return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']

    main(args)	