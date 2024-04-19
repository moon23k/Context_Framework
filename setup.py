import os, json, zipfile, unicodedata
from tqdm import tqdm



def extract_zip_file_names():
    trg_dirs = []
    fnames = []

    #get dir info and save it to lst
    for root, dirs, files in os.walk('data'):
        norm_root = unicodedata.normalize('NFC', root)
        if '라벨링' in norm_root:
            trg_dirs.append(norm_root)

    for dir in trg_dirs:
        for f in os.listdir(dir):
            if 'en' in f:
                fnames.append('/'.join([dir, f]))
    return fnames



def concat_data(zip_file_names):
    concatenated = []

    for zip_file in tqdm(zip_file_names):
        with zipfile.ZipFile(zip_file, 'r') as data_dir:
            for data_file in data_dir.infolist():
                with data_dir.open(data_file) as file:
                    data_str = file.read().decode('utf-8')
                    data_dict = json.loads(data_str)

                    lang_pair = f"{data_dict['sourceLanguage']}{data_dict['targetLanguage']}"
                    x = data_dict['sourceText'] if lang_pair == 'enko' else data_dict['targetText']
                    y = data_dict['targetText'] if lang_pair == 'enko' else data_dict['sourceText']

                    sample = {
                        'domain': data_dict['domain'],
                        'style': data_dict['style'],
                        'x': x,
                        'y': y
                    }

                    concatenated.append(sample)

    return concatenated



def main():
    zip_file_names = extract_zip_file_names()
    concatenated = concat_data(zip_file_names)    