import os, json, yaml, nltk
import sentencepiece as spm
from datasets import load_dataset



def filter_data(orig_data, min_len=500, max_len=3000):
    train, valid, test = orig_data['train'], orig_data['validation'], orig_data['test']
    
    src_list, trg_list = [], []
    for split in [train, valid, test]:
        for elem in split:
            if min_len < len(elem['article']) < max_len:
                src, trg = elem['article'], elem['highlights']

                trg = re.sub(r'\n', ' ', trg)                 #remove \n
                trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

                src_list.append(src)
                trg_list.append(trg)

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


def tokenize_data(src_list, trg_list, tokenizer):
    tokenized_data = []
    for src, trg in zip(src_list, trg_list):
        temp_dict = dict()
        _src = nltk.tokenize.sent_tokenize(src) #split text into sentences
        temp_dict['src'] = tokenizer.Encode(_src)
        temp_dict['trg'] = tokenizer.Encode(trg)
        tokenized_data.append(temp_dict)

    return tokenized_data


def main():
    orig_data = load_dataset('cnn_dailymail', '3.0.0')
    src_list, trg_list = filter_data(orig_data)
    
    build_vocab()
    tokenizer = load_tokenizer()

    tokenized_data = tokenize_data(src_list, trg_list, tokenizer)
    train, valid, test = tokenized_data[:-6000], tokenized_data[-6000:-3000], tokenized_data[-3000:]

    save_json(train, 'train.json')
    save_json(valid, 'valid.json')
    save_json(test, 'test.json')



if __name__ == '__main__':
    nltk.download('punkt')
    main()
    assert os.path.exists(f'data/train.json')
    assert os.path.exists(f'data/valid.json')
    assert os.path.exists(f'data/test.json')