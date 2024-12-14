
import json
import os

data_path = '/mnt/md1/check_point_text_recognition/data_chatbot/data_vietrag7b_time_241204-084854.json'
with open(data_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for i, item in enumerate(data):
        if i % 20 == 0:
            print(item['text'])
            print()