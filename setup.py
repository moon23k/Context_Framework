import os, re, json, nltk, argparse, torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizerFast



class SetupConfig(object):
    def __init__(self, strategy):    

        self.strategy = strategy
        
        self.volumn = 32000
        self.max_num = 50
        self.min_len = 500
        self.max_len = 3000
        
        self.max_sents = 0
        self.max_tokens = 0
        self.pad_token_id = None
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_type = device_type
        self.device = torch.device(device_type)
        self.bert_name= 'bert-base-uncased'
        self.batch_size = 32
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def list2arr(lst):
    return np.array(lst).astype(int)


def select_data(config, orig_data):
    volumn_cnt, selected = 0, []

    for elem in orig_data:
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (config.min_len < len(src) < config.max_len):
            continue
        if len(trg) > config.min_len:
            continue
        
        #Filter too long Sentences 
        src = nltk.tokenize.sent_tokenize(src)
        src_sents = len(src)
        if src_sents > config.max_num:
            continue
        for seq in src:
            if len(seq) > config.min_len:
                break

        #remove unnecessary characters in trg sequence
        trg = re.sub(r'\n', ' ', trg)                 #remove \n
        trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

        selected.append({'src': src, 'trg': trg})

        volumn_cnt += 1
        if volumn_cnt == config.volumn:
            break
    
    return selected



def tokenize_data(config, tokenizer, data_obj):
    tokenized = []
    max_sents, max_tokens = 0, 0

    for elem in data_obj:
        src_encodings = tokenizer(elem['src'], padding=True, truncation=True, return_tensors='np')
        trg_encodings = tokenizer(elem['trg'], padding=True, truncation=True, return_tensors='np')

        curr_sents, curr_tokens = src_encodings.input_ids.shape

        if max_sents < curr_sents:
            max_sents = curr_sents

        if max_tokens < curr_tokens:
            max_tokens = curr_tokens            

        tokenized.append({'input_ids': src_encodings.input_ids.astype(int),
                          'attention_mask':src_encodings.attention_mask.astype(int),
                          'labels': trg_encodings.input_ids.squeeze().astype(int)})
        
    config.max_sents = max_sents
    config.max_tokens = max_tokens
        
    return tokenized



def fine_processing(config, data_obj):
    processed = []
    for elem in data_obj:
        temp = []
        for idx, seq in enumerate(elem['input_ids']):
            seq_ids = [tok for tok in seq if tok != config.pad_token_id]
            temp.extend(seq_ids)
        
        processed.append({'input_ids': list2arr(temp),
                          'labels': elem['labels']})

    return processed



class BertDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]['input_ids'],
                'attention_mask': self.data[idx]['attention_mask']}


class BertCollator(object):
    def __init__(self, config):
        self.max_sents = config.max_sents
        self.max_tokens = config.max_tokens
    
    def __call__(self, batch):
        ids_batch, mask_batch, sent_batch = [], [], []
        for elem in batch:
            seq_num, seq_len = elem['input_ids'].shape
            
            sents_pads_len = self.max_sents-seq_num
            token_pads_len = self.max_tokens-seq_len

            token_pads = np.zeros((seq_num, token_pads_len)).astype(int)
            sents_pads = np.zeros((sents_pads_len, self.max_tokens)).astype(int)

            padded_ids = np.concatenate((elem['input_ids'], token_pads), axis=1)
            padded_ids = np.concatenate((padded_ids, sents_pads), axis=0)

            padded_mask = np.concatenate((elem['attention_mask'], token_pads), axis=1)
            padded_mask = np.concatenate((padded_mask, sents_pads), axis=0)

            sentence_mask = [1 for _ in range(seq_num)] + [0 for _ in range(sents_pads_len)]
            sentence_mask = np.array(sentence_mask).astype(int)

            ids_batch.append(padded_ids)
            mask_batch.append(padded_mask)
            sent_batch.append(sentence_mask)

        ids_batch = torch.LongTensor(ids_batch)
        mask_batch = torch.LongTensor(mask_batch)

        return {'input_ids': ids_batch, 
                'attention_mask': mask_batch,
                'sentence_mask': sent_mask}




def feat_processing(config, bert_model, dataloader):
    processed = []    
    bert_model.eval()

    for batch in dataloader:
    
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        sentence_mask = batch['sentence_mask']

        input_ids = input_ids.view(-1, config.max_tokens)
        attention_mask = attention_mask.view(-1, config.max_tokens)


        with torch.no_grad():
            with torch.autocast(device_type=config.device_type, dtype=torch.float16):
                bert_out = bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
                bert_out = bert_out.view(config.batch_size, config.max_sents, -1)

        processed.append({'sent_embs': bert_out.cpu().detach().numpy(),
                          'sent_masks': sentence_mask,
                          'labels': elem['labels']})

    return processed



def save_data(config, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    os.makedirs(f"data/{config.strategy}", exist_ok=True)
    for key, val in data_dict.items():
        with open(f'data/{config.strategy}/{key}.npy', 'wb') as f:
            np.save(f, val)        
        assert os.path.exists(f'data/{config.strategy}/{key}.npy')



def main(config):
    #pre-requisites
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_name)
    config.pad_token_id = tokenizer.pad_token_id

    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    selected = select_data(config, orig)
    tokenized = tokenize_data(config, tokenizer, selected)

    if config.strategy == 'feat':
        
        bert_model = BertModel.from_pretrained(config.bert_name).to(config.device)
        
        bert_dataloader = DataLoader(BertDataset(tokenized),
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     collate_fn=BertCollator(config),
                                     num_workers=2)

        processed = feat_processing(config, bert_model, tokenized)
    
    elif config.strategy == 'fine':
        processed = fine_processing(config, tokenized)

    save_data(config, processed)



if __name__ == '__main__':
    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', required=True)
    args = parser.parse_args()
    assert args.strategy in ['fine', 'feat']

    config = SetupConfig(args.strategy)
    main(config)