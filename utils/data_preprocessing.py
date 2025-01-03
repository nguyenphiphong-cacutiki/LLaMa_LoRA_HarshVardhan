import json
import random
from datasets import Dataset, load_dataset
# from transformers import load_dataset

def custom_load_dataset(data_path, max_seq_length, model='llama'):
    from utils.count_token_of_sequence import count_token
    import os

    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        results = []
        for data_point in data:
            num = count_token(data_point['text'], model=model)
            if num <= max_seq_length:
                results.append(data_point)
    with open('tmp.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False)
    data_loaded = load_dataset('json', data_files='tmp.json', split="train")
    print('num of data sample:', len(data_loaded))

    # split data
    train_val_split = data_loaded.train_test_split(test_size=0.25, seed=42)
    train_dataset = train_val_split['train'].shuffle(seed=42)

    val_test_dataset = train_val_split['test'].train_test_split(test_size=0.2, seed=42)
    val_dataset = val_test_dataset['train'].shuffle(seed=42)
    test_dataset = val_test_dataset['test'].shuffle(seed=42)
    os.remove('tmp.json')
    return train_dataset, val_dataset, test_dataset


def custom_load_dataset2(save_dir, data_path, max_seq_length, model='llama'):
    from utils.count_token_of_sequence import count_token
    import os

    train_data = []
    val_data = []
    test_data = []
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        print('origin data length:', len(data))
        train_data = []
        val_data = []
        test_data = []
        for data_point in data:
            num = count_token(data_point['text'], model=model)
            if num <= max_seq_length:
                random_n = random.random()
                if random_n > 0.15:
                    train_data.append(data_point)
                elif random_n > 0.05:
                    val_data.append(data_point)
                else:
                    test_data.append(data_point)
        # print('loaded data length:', len(results))
    train_file = os.path.join(save_dir, 'train_data.json')
    val_file = os.path.join(save_dir, 'val_data.json')
    test_file = os.path.join(save_dir, 'test_data.json')

    with open(train_file, 'w', encoding='utf-8') as json_file:
        json.dump(train_data, json_file, ensure_ascii=False)

    with open(val_file, 'w', encoding='utf-8') as json_file:
        json.dump(val_data, json_file, ensure_ascii=False)

    with open(test_file, 'w', encoding='utf-8') as json_file:
        json.dump(test_data, json_file, ensure_ascii=False)

    train_dataset = load_dataset('json', data_files=train_file, split="train")
    val_dataset = load_dataset('json', data_files=val_file, split="train")
    test_dataset = load_dataset('json', data_files=test_file, split="train")

    return train_dataset, val_dataset, test_dataset