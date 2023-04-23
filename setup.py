import os, re, json, nltk, torch
from datasets import load_dataset
from transformers import BertTokenizerFast



#Select and Tokenize Data
def process_data(orig_data, tokenizer):

    processed = []
    cnt, volumn = 0, 34000
    min_len, max_len = 1000, 3000

    
    for elem in orig_data:
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(src) < max_len):
            continue

        src = nltk.tokenize.sent_tokenize(src)
        src = tokenizer(src).input_ids
        
        temp_ids, temp_segs = [], []
        for idx, ids in enumerate(src):
            _len = len(ids)
            
            #Add ids
            temp_ids.extend(ids)
            
            #Add segs
            if not idx % 2:
                temp_segs.extend([0 for _ in range(_len)])
            else:
                temp_segs.extend([1 for _ in range(_len)])

        processed.append({"input_ids": temp_ids,
                         "token_type_ids": temp_segs,
                         'labels': tokenizer(trg).input_ids})
        
        cnt += 1
        if cnt == volumn:
            break

    return processed



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-4000], data_obj[-4000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)                    



def main():
    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-small')
    tokenizer.model_max_length = 1024

    processed = process_data(orig, tokenizer)    
    save_data(processed)



if __name__ == '__main__':
    nltk.download('punkt')    
    main()