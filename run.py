import copy, argparse, torch
from module.model import load_model
from module.data import load_dataloader
from module.test import Tester
from module.train import Trainer
from transformers import set_seed, BertModel, BertTokenizerFast




class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.strategy = args.strategy
        self.bert_name = 'prajjwal1/bert-small'

        #Training args
        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 32
        self.lr = 5e-4
        self.iters_to_accumulate = 4
        self.ckpt_path = f"ckpt/{self.strategy}.pt"

        self.early_stop = True
        self.patience = 3

        #Model args
        self.n_heads = 8
        self.n_layers = 6
        self.pff_dim = 2048
        self.bert_dim = 768
        self.hidden_dim = 512
        self.dropout_ratio = 0.1
        self.model_max_length = 1024
        self.act = 'gelu'
        self.norm_first = True
        self.batch_first = True

        if self.mode == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')
        else:
            self.search_method = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def inference(config, model, tokenizer):
    print('Type "quit" to terminate Summarization')
    
    while True:
        user_input = input('Please Type Text >> ')
        if user_input.lower() == 'quit':
            print('--- Terminate the Summarization ---')
            print('-' * 30)
            break

        src = config.src_tokenizer.Encode(user_input)
        src = torch.LongTensor(src).unsqueeze(0).to(config.device)

        if config.search == 'beam':
            pred_seq = config.search.beam_search(src)
        elif config.search == 'greedy':
            pred_seq = config.search.greedy_search(src)

        print(f" Original  Sequence: {user_input}")
        print(f'Summarized Sequence: {tokenizer.Decode(pred_seq)}\n')



def main(args):
    set_seed(42)
    config = Config(args)
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_name)
    config.pad_id = tokenizer.pad_token_id
    config.vocab_size = tokenizer.vocab_size
    model = load_model(config)


    if config.mode == 'train': 
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
        return

    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
        return
    
    elif config.mode == 'inference':
        summarizer = inference(config, model, tokenizer)
        return
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-strategy', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.strategy in ['fine', 'feat', 'fuse']

    if args.task == 'inference':
        import nltk
        nltk.download('punkt')
        assert args.search in ['greedy', 'beam']

    main(args)