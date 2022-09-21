import json, torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset




def read_data(f_name):
    with open(f"data/{f_name}", 'r') as f:
        data = json.load(f)
    return data



class SumDataset(Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.model_name = config.model_name
        self.data = read_data(f'{split}.json')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['src'], self.data[idx]['trg']



def _collate_fn(batch):    
    src_batch, _src_batch, trg_batch = [], [], []
    max_seq_num, max_seq_len = 0, 0

    for src, trg in batch:
        _src_batch.append(src)
        trg_batch.append(torch.tensor(trg, dtype=torch.long))

        if max_seq_num < len(src):
            max_seq_num = len(src)

        for seq in src:
            if max_seq_len < len(seq):
                max_seq_len = len(seq)
    
    pad_seq = [1 for _ in range(max_seq_len)]
    for _doc in _src_batch:
        doc = []
        for seq in _doc:
            len_gap = max_seq_len - len(seq)
            if len_gap:
                seq += [1] * len_gap
            doc.append(seq)

        num_gap = max_seq_num - len(_doc)
        if num_gap:
            doc.extend([pad_seq for _ in range(num_gap)])

        src_batch.append(doc)

    src_batch = torch.tensor(src_batch, dtype=torch.long)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=1)

    return {'src': src_batch, 'trg': trg_batch}



def load_dataloader(config, split):
    dataset = SumDataset(config, split)    
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=False, 
                      collate_fn=_collate_fn, 
                      num_workers=2)