from modules.search import RNNSearch, TransSearch



class Summarizer:
    def __init__(self, config, model, tokenizer):
        self.model = model
        self.model_name = config.model_name
        self.search_method = config.search_method

        if config.model_name != 'transformer':
            self.search = RNNSearch(config, model, tokenizer)
        else:
            self.search = TransSearch(config, model, tokenizer)


    def summarize(self):
        self.model.eval()
        print(f'--- Summarization Started on {self.model_name} model! ---')
        print('[ Type "quit" on user input to stop the Process ]')
        
        while True:
            input_seq = input('\nUser Input sentence >> ')
            if input_seq.lower() == 'quit':
                print('\n--- Summarization has terminated! ---')
                break        
            if self.search_method == 'beam':
                output_seq = self.search.beam_search(input_seq)
            else:
                output_seq = self.search.greedy_search(input_seq)
            print(f"Summarized sentence >> {output_seq}")       
