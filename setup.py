import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from collections import namedtuple
import os, re, json, nltk, argparse
from transformers import BertModel, BertTokenizerFast



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

        if config.strategy == 'fine':
            src_segs = []
            for idx, seq in enumerate(src):
                if idx % 2 == 0:
                    seg = [0 for _ in range(len(seq))]
                else:
                    seg = [1 for _ in range(len(seq))]
                src_segs.extend(seg)

            src = ' '.join(src)

        #remove unnecessary characters in trg sequence
        trg = re.sub(r'\n', ' ', trg)                 #remove \n
        trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

        selected.append({'src': src,
                         'trg': trg,
                         'src_segs': src_segs})

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
        
        #update config field for feat_processing
        if config.strategy == 'feat':
            config = config._replace(max_sents = max_sents)
            config = config._replace(max_tokens = max_tokens)
        
    return tokenized



def fine_processing(config, data_obj):
    processed = []
    for elem in data_obj:
        ids_temp, seg_temp = [], []
        for idx, seq in enumerate(elem['input_ids']):
            seq_ids = [tok for tok in seq if tok != config.pad_token_id]
            seq_len = len(seq)

            if idx % 2 == 0:
                seq_seg = [0 for _ in range(seq_len)]
            else:
                seq_seg = [1 for _ in range(seq_len)]
            
            assert len(seq_ids) == len(seq_seg)
            ids_temp.extend(seq_ids)
            seg_temp.extend(seq_seg)
        
        processed.append({'input_ids': list2arr(ids_temp),
                          'token_type_ids': list2arr(seg_temp),
                          'attention_mask': list2arr([1 for _ in range(seq_len)]),
                          'labels': elem['labels']})
    return processed



def feat_processing(config, bert_model, data_obj):
    max_sents, max_tokens = config.max_sents, config.max_tokens

    processed = []
    bert_model.eval()

    for elem in tqdm(data_obj):
        curr_sents, curr_tokens = elem['input_ids'].shape

        pad_tokens = np.zeros((curr_sents, max_tokens-curr_tokens), dtype='int64')
        pad_sents = np.zeros((max_sents-curr_sents, max_tokens), dtype='int64')

        padded_input_ids = np.concatenate((elem['input_ids'], pad_tokens), axis=1)
        padded_input_ids = np.concatenate((padded_input_ids, pad_sents), axis=0)

        padded_attention_mask = np.concatenate((elem['attention_mask'], pad_tokens), axis=1)
        padded_attention_mask = np.concatenate((padded_attention_mask, pad_sents), axis=0)

        input_ids = torch.LongTensor(padded_input_ids).to(bert_model.device)
        attention_mask = torch.LongTensor(padded_attention_mask).to(bert_model.device)

        #get bert_out
        with torch.no_grad():
            with torch.autocast(device_type=config.device_type, dtype=torch.float16):
                bert_out = bert_model(input_ids = input_ids, attention_mask = attention_mask).pooler_output
                
        #get sent_masks
        sent_masks = np.concatenate((np.ones(curr_sents), np.zeros(max_sents-curr_sents)), axis=0).astype(int)

        processed.append({'sent_embs': bert_out.cpu().detach().numpy(),
                          'sent_masks': sent_masks,
                          'labels': elem['labels']})

    return processed



def save_data(config, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{config.strategy}/{key}.npy', 'wb') as f:
            np.save(f, val)        
        assert os.path.exists(f'data/{config.strategy}/{key}.npy')
    


def main(config):
    #pre-requisites
    bert_name = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(bert_name)
    config = config._replace(pad_token_id = tokenizer.pad_token_id)

    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    selected = select_data(orig)
    tokenized = tokenize_data(tokenizer, selected)

    if config.strategy == 'feat':
        bert_model = BertModel.from_pretrained(bert_name)
        processed = feat_processing(config, bert_model, tokenized)
    elif config.strategy == 'fine':
        processed = fine_processing(config, tokenized)

    save_data(config, tokenized)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['fine', 'feat']

    config_dict = {'strategy': args.strategy,
                   'device_type': 'cuda' if torch.cuda.is_available() else 'cpu',
                   'volumn': 32000,
                   'max_num': 50,
                   'min_len': 500,
                   'max_len': 3000,
                   'max_sents': 0,
                   'max_tokens': 0,
                   'pad_token_id': None}

    config = namedtuple('Config', config_dict.keys())(**config_dict)

    nltk.download('punkt')
    main(config)