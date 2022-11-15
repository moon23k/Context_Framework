import torch, operator
import torch.nn.functional as F
from queue import PriorityQueue
from collections import namedtuple



class Beam:
    def __init__(self, config, model, tokenizer):
        super(Beam, self).__init__()
        self.beam_size = 4
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.pad_idx = config.pad_idx
        self.model_name = config.model_name
        
        if self.model.training:
            self.model.eval()
        
        if self.model_name != 'transformer':
            self.Node = namedtuple('Node', ['prev_node', 'pred', 'log_prob', 'hidden', 'length'])
        elif self.model_name == 'transformer':
            self.Node = namedtuple('Node', ['prev_node', 'pred', 'log_prob', 'length'])


    def get_score(self, node, max_repeat, min_length=5, alpha=1.2):
        repeat = max([node.pred.tolist().count(token) for token in node.pred.tolist() if token != self.pad_idx])

        if repeat > max_repeat + 5:
            repeat_penalty = -1
        else:
            repeat_penalty = 1
        
        len_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
        score = node.log_prob / len_penalty
        score = score * repeat_penalty
        return score


    def get_nodes(self, hidden=None):
        Node = self.Node
        nodes = PriorityQueue()
        start_tensor = torch.LongTensor([[self.bos_idx]]).to(self.device)

        if self.model_name != 'transformer':
            start_node = Node(prev_node = None, 
                              pred = start_tensor, 
                              log_prob = 0.0, 
                              hidden = hidden,                               
                              length = 0)
        elif self.model_name == 'transformer':
            start_node = Node(prev_node = None,
                              pred = start_tensor,
                              log_prob = 0.0,
                              length = 0)

        for _ in range(self.beam_size):
            nodes.put((0, start_node))        
        return Node, nodes, [], []    


    def get_input_params(self, input_seq):
        input_tokens = self.tokenizer.encode(input_seq)
        input_tensor = torch.LongTensor([input_tokens]).to(self.device)
        max_len = len(input_tokens) + 30
        max_repeat = max([input_tokens.count(token) for token in input_tokens if token != self.pad_idx])
        return input_tensor, max_len, max_repeat        




class RNNSearch(Beam):
    def __init__(self, config, model, tokenizer):
        super(RNNSearch, self).__init__(config, model, tokenizer)

    def beam_search(self, input_seq, topk=1):
        input_tensor, max_len, max_repeat = self.get_input_params(input_seq)
        
        if self.model_name == 'seq2seq':
            hidden = self.model.encoder(input_tensor)
        elif self.model_name == 'attention':
            e_out, hidden = self.model.encoder(input_tensor)

        Node, nodes, end_nodes, top_nodes = self.get_nodes(hidden=hidden)

        for t in range(max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]
            
            for curr_score, curr_node in curr_nodes:
                if curr_node.pred[:, -1].item() == self.eos_idx and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue

                if self.model_name == 'seq2seq':
                    out, hidden = self.model.decoder(curr_node.pred[:, -1], curr_node.hidden)
                elif self.model_name == 'attention':
                    out, hidden = self.model.decoder(curr_node.pred[:, -1], e_out, curr_node.hidden)
                    hidden = hidden.unsqueeze(0)    
                
                logits, preds = torch.topk(out, self.beam_size)
                logits, preds = logits, preds
                log_probs = -F.log_softmax(logits, dim=-1)

                for k in range(self.beam_size):
                    pred = preds[:, k].unsqueeze(0)
                    log_prob = log_probs[:, k].item()
                    pred = torch.cat([curr_node.pred, pred], dim=-1)

                    next_node = Node(prev_node = curr_node,
                                     pred = pred,
                                     log_prob = curr_node.log_prob + log_prob,
                                     hidden = hidden,
                                     length = curr_node.length + 1)
                    
                    next_score = self.get_score(next_node, max_repeat)
                    nodes.put((next_score, next_node))    

                if not t:
                    break

        if len(end_nodes) == 0:
            _, top_node = nodes.get()
        else:
            _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
        
        beam_out = top_node.pred.squeeze(0).tolist()
        return self.tokenizer.decode(beam_out)      
    

    def greedy_search(self, input_seq):
        input_tensor, max_len, _ = self.get_input_params(input_seq)

        output_seq = [[self.pad_idx  if i else self.bos_idx for i in range(max_len)]]        
        output_tensor = torch.LongTensor(output_seq).to(self.device)
        dec_input = output_tensor[:, 0]

        if self.model_name == 'seq2seq':
            hiddens = self.model.encoder(input_tensor)
            for i in range(1, max_len):
                out, hiddens = self.model.decoder(dec_input, hiddens)
                output_tensor[:, i] = out.argmax(-1)
                dec_input = output_tensor[:, i]

        elif self.model_name == 'attention':
            enc_out, hiddens = self.model.encoder(input_tensor)
            for i in range(1, max_len):
                out, hiddens = self.model.decoder(dec_input, enc_out, hiddens)
                hiddens = hiddens.unsqueeze(0)
                output_tensor[:, i] = out.argmax(-1)
                dec_input = output_tensor[:, i]

        return self.tokenizer.decode(output_tensor.squeeze(0).tolist())



class TransSearch(Beam):
    def __init__(self, config, model, tokenizer):
        super(TransSearch, self).__init__(config, model, tokenizer)

    def beam_search(self, input_seq, topk=1):
        Node, nodes, end_nodes, top_nodes = self.get_nodes()
        input_tensor, max_len, max_repeat = self.get_input_params(input_seq)

        e_mask = self.model.pad_mask(input_tensor)
        memory = self.model.encoder(input_tensor, e_mask)
        
        for t in range(max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]

            for curr_score, curr_node in curr_nodes:
                if curr_node.pred[:, -1].item() == self.eos_idx and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue
                
                d_input = curr_node.pred 
                d_mask = self.model.dec_mask(d_input)
                d_out = self.model.decoder(d_input, memory, e_mask, d_mask)
                out = self.model.fc_out(d_out)[:, -1]
                
                logits, preds = torch.topk(out, self.beam_size)
                logits, preds = logits, preds
                log_probs = -F.log_softmax(logits, dim=-1)

                for k in range(self.beam_size):
                    pred = preds[:, k].unsqueeze(0)
                    log_prob = log_probs[:, k].item()
                    pred = torch.cat([curr_node.pred, pred], dim=-1)           
                    
                    next_node = Node(prev_node = curr_node,
                                     pred = pred,
                                     log_prob = curr_node.log_prob + log_prob,
                                     length = curr_node.length + 1)
                    next_score = self.get_score(next_node, max_repeat)                
                    nodes.put((next_score, next_node))
                
                if not t:
                    break

        if len(end_nodes) == 0:
            _, top_node = nodes.get()
        else:
            _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
        
        beam_out = top_node.pred.squeeze(0).tolist()
        return self.tokenizer.decode(beam_out)        


    def greedy_search(self, input_seq):
        input_tensor, max_len, _ = self.get_input_params(input_seq)

        output_seq = [[self.pad_idx  if i else self.bos_idx for i in range(max_len)]]
        output_tensor = torch.LongTensor(output_seq).to(self.device)

        e_mask = self.model.pad_mask(input_tensor)
        memory = self.model.encoder(input_tensor, e_mask)        

        for i in range(1, max_len):
            d_mask = self.model.pad_mask(output_tensor)
            out = self.model.decoder(output_tensor, memory, e_mask, d_mask)
            out = self.model.fc_out(out)
            
            pred = out[:, i].argmax(-1)
            output_tensor[:, i] = pred

            if pred.item() == self.eos_idx:
                break

        return self.tokenizer.decode(output_tensor.squeeze(0).tolist())
