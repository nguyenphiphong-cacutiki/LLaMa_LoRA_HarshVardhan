import json
import os
from count_token_of_sequence import count_token
from data_preprocessing import custom_load_dataset
data_path = '/mnt/md1/check_point_text_recognition/data_chatbot/data_llama_7b_chat_hf_time_241216-145514.json'
train_dataset, val_dataset, test_dataset = custom_load_dataset(data_path, max_seq_length=2048)

