import json
import os
from count_token_of_sequence import count_token
data_path = '/mnt/md1/check_point_text_recognition/data_chatbot/data_llama_7b_chat_hf_time_241202-190841.json'
with open(data_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for item in data:
        if count_token(item['text']) < 1536:
            print(item['text'])
            print('-'*100)
            # break