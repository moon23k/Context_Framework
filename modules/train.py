import time, math, json, torch
import torch.nn as nn
import torch.optim as optim



class Trainer:
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__()
        self.model = model
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.output_dim = config.output_dim
        self.model_name = config.model_name
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, 
                                             label_smoothing=0.1).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=config.learning_rate, 
                                    betas=(0.9, 0.98), 
                                    eps=1e-8)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.ckpt_path = config.ckpt_path
        self.record_path = f"ckpt/{self.model_name}.json"
        self.record_keys = ['epoch', 'train_loss', 'train_ppl',
                            'valid_loss', 'valid_ppl', 
                            'learning_rate', 'train_time']


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
              Train PPL: {record_dict['train_ppl']:.2f}""".replace(' ' * 14, ''))

        print(f"""  >> Valid Loss: {record_dict['valid_loss']:.3f} | \
              Valid PPL: {record_dict['valid_ppl']:.2f}\n""".replace(' ' * 14, ''))


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def train(self):
        best_bleu, records = float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_bleu > val_loss:
                best_bleu = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt_path)
            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        for _, batch in enumerate(self.train_dataloader):
            src = batch['src'].to(self.device)
            trg_input = batch['trg_input'].to(self.device)
            trg_output = batch['trg_output'].to(self.device)
            
            logit = self.model(src, trg_input)
            loss = self.criterion(logit.contiguous().view(-1, self.output_dim),
                                  trg_output.contiguous().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)    
        return epoch_loss, epoch_ppl
    

    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        tot_len = len(self.valid_dataloader)
        
        with torch.no_grad():
            for _, batch in enumerate(self.valid_dataloader):
                src = batch['src'].to(self.device)
                trg_input = batch['trg_input'].to(self.device)
                trg_output = batch['trg_output'].to(self.device)
                
                if self.model_name != 'transformer':
                    logit = self.model(src, trg_input, teacher_forcing_ratio=0.0)
                else:
                    logit = self.model(src, trg_input)

                loss = self.criterion(logit.contiguous().view(-1, self.output_dim),
                                      trg_output.contiguous().view(-1))
                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        
        return epoch_loss, epoch_ppl