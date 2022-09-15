import os, json, yaml
import sentencepiece as spm
from datasets import load_dataset



def load_data(min_len=500, max_len=3000):
    orig_data = load_dataset('cnn_dailymail', '3.0.0')
    train, valid, test = orig_data['train'], orig_data['validation'], orig_data['test']
	
    src_list, trg_list = [], []
    for split in [train, valid, test]:
    	for elem in split:
            if min_len < len(elem['article']) < max_len:
                src_list.append(elem['article'])
                trg_list.append(elem['highlights'])

    with open('data/concat.txt', 'w') as f:
        f.write('\n'.join(src_list + trg_list))

    return src_list, trg_list


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
    src_list, trg_list = load_data()
    print('load dataset completed\n')
    build_vocab()
    print('build vocab completed\n')
    tokenizer = load_tokenizer()
    print('load tokenizer completed\n')

    tokenized_data = [{'src': tokenizer.EncodeAsIds(src),\
                        'trg': tokenizer.EncodeAsIds(trg)}\
                        for src, trg in zip(src_list, trg_list)]
    print('tokenize data completed\n')

    train, valid, test = tokenized_data[:-6000], tokenized_data[-6000:-3000], tokenized_data[-3000:]
    print(f'train_len: {len(train)} valid_len: {len(valid)}, test_len: {len(test)}')
    save_json(train, 'train.json')
    save_json(valid, 'valid.json')
    save_json(test, 'test.json')



if __name__ == '__main__':
    main()
    assert os.path.exists(f'data/train.json')
    assert os.path.exists(f'data/valid.json')
    assert os.path.exists(f'data/test.json')