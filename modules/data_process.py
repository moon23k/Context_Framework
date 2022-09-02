import os, json, yaml
import sentencepiece as spm



def read_text(f_name):
    with open(f"data/{f_name}", 'r') as f:
        data = f.readlines()
    return data


def read_json(split):
    with open(f'data/{split}.json', 'r') as f:
        data = json.load(f)
    return data


def save_json(split, data):
    with open(f"data/{split}.json", 'w') as f:
        json.dump(data, f)    


def text2json(split):
    fname_dict = {'train': 'train', 'valid': 'val', 'test': 'test_2016_flickr'}
    
    src_text = read_text(f"{fname_dict[split]}.en")
    trg_text = read_text(f"{fname_dict[split]}.de")

    json_dict = []
    for src, trg in zip(src_text, trg_text):
        temp_dict = dict()
        temp_dict['src'] = src.strip()
        temp_dict['trg'] = trg.strip()
        json_dict.append(temp_dict)
    
    save_json(split, json_dict)

    os.remove(f'data/{fname_dict[split]}.en')
    os.remove(f'data/{fname_dict[split]}.de')


def create_concats():
    src, trg = [], []
    for split in ['train', 'valid', 'test']:
        data = read_json(split)

        for elem in data:
            src.append(elem['src'])
            trg.append(elem['trg'])

    with open('data/en_concat.txt', 'w') as f:
        f.write('\n'.join(src))
    with open('data/de_concat.txt', 'w') as f:
        f.write('\n'.join(trg))


def build_vocab(lang):
    assert os.path.exists('configs/vocab.yaml')
    assert os.path.exists(f'data/{lang}_concat.txt')
    
    with open('configs/vocab.yaml', 'r') as f:
        vocab_dict = yaml.load(f, Loader=yaml.FullLoader)

    opt = f"--input=data/{lang}_concat.txt\
            --model_prefix=data/{lang}_tokenizer\
            --vocab_size={vocab_dict['vocab_size']}\
            --character_coverage={vocab_dict['coverage']}\
            --model_type={vocab_dict['type']}\
            --unk_id={vocab_dict['unk_id']} --unk_piece={vocab_dict['unk_piece']}\
            --pad_id={vocab_dict['pad_id']} --pad_piece={vocab_dict['pad_piece']}\
            --bos_id={vocab_dict['bos_id']} --bos_piece={vocab_dict['bos_piece']}\
            --eos_id={vocab_dict['eos_id']} --eos_piece={vocab_dict['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove(f'data/{lang}_concat.txt')


def tokenize_data(data, src_tokenizer, trg_tokenizer):
    tokenized_data = []

    for elem in data:
        temp_dict = dict()

        temp_dict['src_seq'] = elem['src']
        temp_dict['trg_seq'] = elem['trg']
        
        temp_dict['src_ids'] = src_tokenizer.EncodeAsIds(elem['src'])
        temp_dict['trg_ids'] = trg_tokenizer.EncodeAsIds(elem['trg'])
        
        tokenized_data.append(temp_dict)
        
    return tokenized_data


def load_tokenizer(lang):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{lang}_tokenizer.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')    
    return tokenizer


if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:
        text2json(split)
    create_concats()
    build_vocab('en')
    build_vocab('de')

    src_tokenizer = load_tokenizer('en')
    trg_tokenizer = load_tokenizer('de')

    train_data = read_json('train')
    valid_data = read_json('valid')
    test_data = read_json('test')

    train_data = tokenize_data(train_data, src_tokenizer, trg_tokenizer)
    valid_data = tokenize_data(valid_data, src_tokenizer, trg_tokenizer)
    test_data = tokenize_data(test_data, src_tokenizer, trg_tokenizer)

    save_json('train', train_data)
    save_json('valid', valid_data)
    save_json('test', test_data)