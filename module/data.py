import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.npy", 'rb') as f:
            data = np.load(f, allow_pickle=True)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
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



def load_dataloader(config, split):
    return DataLoader(Dataset(split), 
                      batch_size=128, 
                      shuffle=True if config.mode == 'train' else False, 
                      collate_fn=Collator(config.pad_id), 
                      num_workers=2)