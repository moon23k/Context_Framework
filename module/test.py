import torch, math, time
import torch.nn as nn
import evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.strategy = config.strategy
        self.dataloader = test_dataloader


    def test(self):
        if self.strategy == 'fine':
            return self.fine_test()
        elif self.strategy == 'feat':
            return self.feat_test()

    def fine_test(self):
        return

    def feat_test(self)


    def metric_score(self, pred, label):
        return