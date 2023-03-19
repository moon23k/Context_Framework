import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(strategy, split)

    @staticmethod
    def load_data(split):
        with open(f"data/{strategy}/{split}.npy", 'rb') as f:
            data = np.load(f, allow_pickle=True)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



class Collator(object):
    def __init__(self, strategy, pad_id):
        self.strategy = strategy
        self.pad_id = pad_id

    def __call__(self, batch):
        if self.strategy == 'fine':
            return self.fine_collate(batch)
        elif self.strategy == 'feat':
            return self.feat_collate(batch)


    def feat_collate(self, batch):
        sent_embs_batch, sent_masks_batch, labels_batch = [], [], []
        for elem in batch:
            sent_embs_batch.append(elem['sent_embs'])
            sent_masks_batch.append(elem['sent_masks']) 
            labels_batch.append(torch.LongTensor(elem['labels']))

        sent_embs_batch = torch.Tensor(sent_embs_batch)
        sent_masks_batch = torch.LongTensor(sent_masks_batch)

        labels_batch = pad_sequence(labels_batch, 
                                    batch_first=True, 
                                    padding_value=self.pad_id)        

        return {'sent_embs': sent_embs_batch,
                'sent_masks': sent_masks_batch,
                'labels': labels_batch}
        


    def fine_collate(self, batch):
        ids_batch, seg_batch, mask_batch, labels_batch = [], [], [], []
        for elem in batch:
            ids_batch.append(torch.LongTensor(elem['input_ids'])) 
            seg_batch.append(torch.LongTensor(elem['token_type_ids']))
            mask_batch.append(torch.LongTensor(elem['attention_mask']))
            labels_batch.append(torch.LongTensor(elem['labels']))

        ids_batch = self.pad_batch(ids_batch)
        seg_batch = self.pad_batch(seg_batch)
        mask_batch = self.pad_batch(mask_batch)    
        labels_batch = self.pad_batch(labels_batch)

        return {'input_ids': ids_batch,
                'token_type_ids': seg_batch,
                'attention_mask': mask_batch,
                'labels': labels_batch}

    def pad_batch(self, batch):
        return pad_sequence(batch, 
                            batch_first=True, 
                            padding_value=self.pad_id)





def load_dataloader(config, split):
    return DataLoader(Dataset(config.strategy, split), 
                      batch_size=128, 
                      shuffle=True if config.mode == 'train' else False, 
                      collate_fn=Collator(config.strategy, config.pad_id), 
                      num_workers=2)