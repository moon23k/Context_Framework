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
        self.metric_module = evaluate.load('rouge')


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"    


    def test(self):
        if self.strategy == 'fine':
            return self.fine_test()
        elif self.strategy == 'feat':
            return self.feat_test()


    def metric_score(self, pred, label):
        return score