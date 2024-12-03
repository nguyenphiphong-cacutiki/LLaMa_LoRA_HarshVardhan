import json
from datasets import Dataset, load_dataset
# from transformers import load_dataset

def custom_load_dataset(data_path, max_seq_length):
    from utils.count_token_of_sequence import count_token
    import os

    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        results = []
        for data_point in data:
            num = count_token(data_point['text'])
            if num <= max_seq_length:
                results.append(data_point)
    with open('tmp.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False)
    data_loaded = load_dataset('json', data_files='tmp.json', split="train")
    print('num of data sample:', len(data_loaded))
    os.remove('tmp.json')
    return data_loaded
