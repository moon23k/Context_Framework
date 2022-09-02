import torch



class Translator:
	def __init__(self, config, model, src_tokenizer, trg_tokenizer):
		self.model = model
		self.max_len = 100
		self.device = config.device
		self.search = config.search
		self.bos_idx = config.bos_idx
		self.src_tokenizer = src_tokenizer
		self.trg_tokenizer = trg_tokenizer
		self.model_name == config.model_name

	
	def tranaslate(self, config):
		self.model.eval()
		print('Type "quit" to terminate Translation')
		while True:
			user_input = input('please type text >> ')
			if user_input == 'quit':
				print('--- Terminate the Translation ---')
				print('-' * 30)
				break

			src = self.src_tokenizer.Encode(user_input)
			src = torch.LongTensor(src).to(self.device)
			pred_seq = torch.LongTensor([self.bos_idx]).to(self.device)

			for t in range(self.max_len):
				out = self.model(src, pred_seq)
				pred_word = out.argmax(-1)
				pred_seq = torch.cat([pred_seq, pred_word])

			print(f"Original Sentence:   {user_input}")
			print(f'Translated Sequenec: {self.trg_tokenizer.Decode(pred_seq)}\n')
			