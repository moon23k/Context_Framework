import torch, json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, strategy, split):
        super().__init__()
        self.strategy = strategy
        self.data = self.load_data(split)

        if self.strategy == 'feat':
            self.feat = torch.load(f"data/{split}.pt")

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.strategy != 'feat':
            ids = self.data[idx]['input_ids']
            seg = self.data[idx]['token_type_ids']
            indice = self.data[idx]['cls_indice']
            mask = [1 for _ in range(len(ids))]
            label = self.data[idx]['labels']
            
            return {'input_ids': ids,
                    'token_type_ids': seg,
                    'attention_mask': mask,
                    'cls_indice': indice,
                    'labels': label}
        else:
            hidden = self.feat[idx]['last_hidden_state']
            indice = self.data[idx]['cls_indice']
            mask = [1 for _ in range(hidden.size(0))]
            label = self.data[idx]['labels']

            return {'last_hidden_state': hidden,
                    'attention_mask': mask,
                    'cls_indice': indice,
                    'labels': label}



class Collator(object):
    def __init__(self, strategy, pad_id):
        self.strategy = strategy
        self.pad_id = pad_id

    def __call__(self, batch):
        if self.strategy != 'feat':
            return self.fine_collate(batch)
        elif self.strategy == 'feat':
            return self.feat_collate(batch)


    def feat_collate(self, batch):
        repr_batch, mask_batch, indice_batch, label_batch = [], [], [], []
        
        for elem in batch:
            repr_batch.append(elem['last_hidden_state'])
            mask_batch.append(torch.LongTensor(elem['attention_mask']))
            indice_batch.append(elem['cls_indice'])
            label_batch.append(torch.LongTensor(elem['labels']))

        return {'last_hidden_state': self.pad_batch(repr_batch),
                'attention_mask': self.pad_batch(mask_batch),
                'cls_indice': indice_batch,
                'labels': self.pad_batch(label_batch)}


    def fine_collate(self, batch):
        ids_batch, seg_batch, mask_batch, indice_batch, label_batch = [], [], [], [], []
        
        for elem in batch:
            ids_batch.append(torch.LongTensor(elem['input_ids'])) 
            seg_batch.append(torch.LongTensor(elem['token_type_ids']))
            mask_batch.append(torch.LongTensor(elem['attention_mask']))
            indice_batch.append(elem['cls_indice'])
            label_batch.append(torch.LongTensor(elem['labels']))

        return {'input_ids': self.pad_batch(ids_batch),
                'token_type_ids': self.pad_batch(seg_batch),
                'attention_mask': self.pad_batch(mask_batch),
                'cls_indice': indice_batch,
                'labels': self.pad_batch(label_batch)}

    def pad_batch(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

def load_dataloader(config, split):
    return DataLoader(Dataset(config.strategy, split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode == 'train' else False, 
                      collate_fn=Collator(config.strategy, config.pad_id), 
                      num_workers=2)
                      