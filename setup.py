import os, re, json, nltk, torch, argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast



class SetupConfig:
    def __init__(self):            

        self.volumn = 12000
        self.min_len = 1000
        self.max_len = 3000
        self.model_max_length = 1024        
        
        self.bert_name= 'prajjwal1/bert-small'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



class BertDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = torch.LongTensor(self.data[idx]['input_ids'])
        seg = torch.LongTensor(self.data[idx]['token_type_ids'])
        indice = torch.LongTensor(self.data[idx]['cls_indice'])
        mask = torch.ones_like(ids, dtype=torch.long)

        return {'input_ids': ids,
                'token_type_ids': seg,
                'attention_mask': mask,
                'cls_indice': indice}



def load_model(config):
    model = BertModel.from_pretrained(config.bert_name)

    model.embeddings.position_ids = torch.arange(config.model_max_length).expand((1, -1))
    model.embeddings.token_type_ids = torch.zeros(config.model_max_length).expand((1, -1))
    orig_pos_emb = model.embeddings.position_embeddings.weight
    model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))
    model.config.max_position_embeddings = config.model_max_length

    return model.to(config.device)



#Select and Tokenize Data
def process_data(config, orig_data, tokenizer):
    selected = []
    volumn_cnt = 0
    
    for elem in orig_data:
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (config.min_len < len(src) < config.max_len):
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

        selected.append({"input_ids": temp_ids,
                         "token_type_ids": temp_segs,
                         "cls_indice": [i for i, x in enumerate(temp_ids) if x == 101],
                         'labels': tokenizer(trg).input_ids})
        
        volumn_cnt += 1
        if volumn_cnt == config.volumn:
            break

    return selected



def extract_features(model, dataloader):
    model.eval()
    features = []
    device = model.device
    cpu = torch.device('cpu')
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = model(input_ids=batch['input_ids'].to(device), 
                            token_type_ids=batch['token_type_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device))
        
        out = out.last_hidden_state.squeeze().to(cpu)
        out = torch.index_select(out, 0, batch['cls_indice'].squeeze())
        features.append({'last_hidden_state': out})    

    return features



def save_data(data_obj, strategy):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        if strategy != 'feat':
            with open(f'data/{key}.json', 'w') as f:
                json.dump(val, f)                    
        else:
            torch.save(val, f"data/{key}.pt")


def main(strategy):
    config = SetupConfig()
    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_name)
    tokenizer.model_max_length = config.model_max_length
    model = load_model(config)

    processed = process_data(config, orig, tokenizer)
    
    if strategy != 'feat':
        save_data(processed, 'fine')

    if strategy != 'fine':
        bert_dataloader = DataLoader(BertDataset(processed), shuffle=False, num_workers=2)
        feature_data = extract_features(model, bert_dataloader)
        save_data(feature_data, 'feat')



if __name__ == '__main__':
    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', required=True)
    
    args = parser.parse_args()
    assert args.strategy in ['all', 'fine', 'feat']    
    main(args.strategy)