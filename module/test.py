import torch




class Tester:
	def __init__(config, model, tokenizer, test_dataloader):
		self.model = model
		self.tokenizer = tokenizer
		self.dataloader = test_dataloader


	def test(self):
		return