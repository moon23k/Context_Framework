import torch, math, time
import torch.nn as nn
import evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        self.device = config.device
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader



    def test(self):
        return


    def metric_score(self, pred, label):
        return