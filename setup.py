import os, json, yaml, nltk 
import sentencepiece as spm
from datasets import load_dataset



def load_data():
	orig_data = load_dataset('cnn_dailymail', '3.0.0')
	train, valid, test = orig_data['train'], orig_data['validation'], orig_data['test']
	
	src_list, trg_list = [], []
	for split in [_train, _valid, _test]:
	    src_list.extend(split['article'])
	    trg_list.extend(split['highlights'])

	with open('data/concat.txt', 'w') as f:
		f.write(src_list + trg_list)

	data = [{'src': src, 'trg': trg for src, trg in zip(src_list, trg_list)}]
	train, valid, test = data[:-6000], data[-6000:-3000], data[-3000:]

	return train, valid, test


def build_vocab():
    assert os.path.exists(f'configs/vocab.yaml')
    assert os.path.exists(f'data/concat.txt')

    with open('configs/vocab.yaml', 'r') as f:
        vocab_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    opt = f"--input=data/concat.txt\
            --model_prefix=data/tokenizer\
            --vocab_size={vocab_dict['vocab_size']}\
            --character_coverage={vocab_dict['coverage']}\
            --model_type={vocab_dict['type']}\
            --unk_id={vocab_dict['unk_id']} --unk_piece={vocab_dict['unk_piece']}\
            --pad_id={vocab_dict['pad_id']} --pad_piece={vocab_dict['pad_piece']}\
            --bos_id={vocab_dict['bos_id']} --bos_piece={vocab_dict['bos_piece']}\
            --eos_id={vocab_dict['eos_id']} --eos_piece={vocab_dict['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove('data/concat.txt')



def save_json(data_obj, data_name):
	with open(f"data/{data_name}", 'w') as f:
		json.dump(data_obj, f)


def load_tokenizer():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tokenizer.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')

	return tokenizer


def main():
	src, trg = load_data()
	build_vocab()
	tokenizer = load_tokenizer()
	
	tokenized_data = [{'src': tokenizer.EncodeAsIds(src), \
					  'trg': tokenizer.EncodeAsIds(trg)}\
					  for src, trg in zip(src, trg)]

	train, valid, test = tokenized_data[:-6000], tokenized_data[-6000:-3000], tokenized_data[-3000:]
	train = train[::3] #downsize
	save_json(train, 'train.json')
	save_json(valid, 'valid.json')
	save_json(test, 'test.json')



if __name__ == 'main':
	main()