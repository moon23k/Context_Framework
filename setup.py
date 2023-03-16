import numpy as np
from tqdm import tqdm
import os, re, json, nltk
from datasets import load_dataset
from transformers import BertModel, BertTokenizerFast




def preprocess_data(orig_data, volumn=32000, max_num=50, min_len=500, max_len=3000):
    volumn_cnt, processed = 0, []

    for elem in orig_data:
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(src) < max_len):
            continue
        if len(trg) > min_len:
            continue
        
        #Filter too long Sentences 
        src_split = nltk.tokenize.sent_tokenize(src)
        if len(src_split) > max_num:
            continue
        for seq in src_split:
            if len(seq) > min_len:
                break

        #remove unnecessary characters in trg sequence
        trg = re.sub(r'\n', ' ', trg)                 #remove \n
        trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

        temp_dict = dict()
        temp_dict['src'] = src_split
        temp_dict['trg'] = trg

        processed.append(temp_dict)

        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
    
    return processed



def tokenize_data(tokenizer, data_obj):
    tokenized_data = []
    max_sents, max_tokens = 0, 0

    #tokenization
    for elem in data_obj:
        
        src_encodings = tokenizer(elem['src'], padding=True, truncation=True, return_tensors='np')
        trg_encodings = tokenizer(elem['trg'], padding=True, truncation=True, return_tensors='np')

        curr_sents, curr_tokens = src_encodings.input_ids.shape

        if max_sents < curr_sents:
            max_sents = curr_sents

        if max_tokens < curr_tokens:
            max_tokens = curr_tokens            

        tokenized_data.append({'input_ids': src_encodings.input_ids,
                               'attention_mask':src_encodings.attention_mask,
                               'labels': trg_encodings.input_ids.squeeze().long()})

    return tokenized_data, max_sents, max_tokens


def bert_embedding(bert_model, data_obj, max_sents, max_tokens):
    embedded = []
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                bert_out = bert_model(input_ids = input_ids, attention_mask = attention_mask).pooler_output
                
        #get sent_masks
        sent_masks = np.concatenate((np.ones(curr_sents), np.zeros(max_sents-curr_sents)), axis=0).astype(int)

        embedded.append({'sent_embs': bert_out.cpu().detach().numpy(),
                         'sent_masks': sent_masks,
                         'labels': elem['labels']})

    return embedded



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.npy', 'wb') as f:
            np.save(f, val)        
        assert os.path.exists(f'data/{key}.npy')
    


def main(task):
    
    bert_name = 'bert-base-uncased'
    bert_model = BertModel.from_pretrained(bert_name)
    tokenizer = BertTokenizerFast.from_pretrained(bert_name)

    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    processed = preprocess_data(orig)

    tokenized, max_sents, max_tokens = tokenize_data(tokenizer, processed)
    embedded = bert_embedding(bert_model, tokenized, max_sents, max_tokens)

    save_data(task, tokenized)



if __name__ == '__main__':
    nltk.download('punkt')
    main()        