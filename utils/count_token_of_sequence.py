from transformers import AutoTokenizer
import json


model_name_llama = "NousResearch/Llama-2-7b-chat-hf"
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama)

tokenizer_vietrag = AutoTokenizer.from_pretrained('llm4fun/vietrag-7b-v1.0', trust_remote_code = True)

def count_token(text, model='llama'):
    if model == 'llama':
        tokens = tokenizer_llama.encode(text)
        return len(tokens)
    elif model == 'vietrag':
        tokens = tokenizer_vietrag.encode(text)
        return len(tokens)

if __name__ == '__main__':
    data_path = '/mnt/md1/check_point_text_recognition/data_chatbot/data_llama_7b_chat_hf_time_241216-145514.json'
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        print('len of data:', len(data))
        num_greater_than_1024 = 0
        num_greater_than_1536 = 0
        num_greater_than_2048 = 0
        num_greater_than_3072 = 0
        num_other = 0

        for data_point in data:
            num = count_token(data_point['text'])
            if num > 3072:
                num_greater_than_3072 += 1
            if num > 2048:
                num_greater_than_2048 += 1
            if num > 1536:
                num_greater_than_1536 += 1
            if num > 1024:
                num_greater_than_1024 += 1
   
            
        print('num data point have token > 1024:', num_greater_than_1024)
        print('num data point have token > 1536:', num_greater_than_1536)
        print('num data point have token > 2048:', num_greater_than_2048)
        print('num data point have token > 3072:', num_greater_than_3072)
        print('num data point other:', num_other)
        print('check sum: ', num_greater_than_1024+num_greater_than_1536+num_greater_than_2048+num_greater_than_3072+num_other)
