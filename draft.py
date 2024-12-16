
import json
import os
from utils.data_preprocessing import custom_load_dataset
import config_llama as cfg

train_dataset, val_dataset, test_dataset = custom_load_dataset(data_path=cfg.data_path, max_seq_length=cfg.max_seq_length)
# print(test_dataset)
# for example in test_dataset.select(range(5)):
#     print(type(example))
for example in test_dataset['text']:  
    print(example)