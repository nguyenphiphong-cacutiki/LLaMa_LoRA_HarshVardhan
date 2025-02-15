# 1.Import Necessary libraries
import os
import torch
import time
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    
    )
from peft import LoraConfig, PeftModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
import config_llama as cfg

accelerator = Accelerator()

# 2. Model and Dataset Configuration
model_name = cfg.model_name
# new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/checkpoint-53390"
new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/241224_llama7b_2/ckpt_end_training"
if os.environ.get('IS_DOCKER') is not None:
    new_model = os.path.join('/app/output', '241202_llama7bchathf/checkpoint-2700')
# device_map = {"":0}

# 3. Tokenizer and PEFT configuration
#Load LLama tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. load model for inference
'''
Since the model is loaded in full precision (float32), it requires more memory. 
For large models like LLaMA-2 7B, this can consume significant GPU memory.
'''
# Step 1: Load the base model
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,  # The original base model's name or path
#     device_map=device_map,  # Or specify your device
# )
'''
Mixed Precision: FP16 uses 16-bit floating point numbers, which reduces the memory usage and
allows the model to fit into GPU memory more easily. However, this could potentially reduce 
numerical accuracy slightly, but in most NLP tasks, the difference is negligible.
'''
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=cfg.device_map,
)
# Step 2: Load the fine-tuned LoRA model (saved from trainer.model.save_pretrained)
model = PeftModel.from_pretrained(base_model, new_model)  # `new_model` is the path where you saved the model

# Step 3: Merge the LoRA weights with the base model
model = model.merge_and_unload()
model, tokenizer = accelerator.prepare(model, tokenizer) #Wrap model and tokenizer with Accelerator

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=cfg.max_seq_length)
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=5400)
pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=2048)

def generate(prompt):
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    result = result[0]['generated_text']
    answer = result.split('[/INST]')[1].split('</s>')[0].strip()
    return answer


def direct_inference(load_file=False, prompt=""):
    if load_file:
        prompt_path = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/prompt_for_test.txt"
        while True:
            prompt = input("Type your question: ")
            if prompt != '0':
                with open(prompt_path, 'r') as file:
                    text = file.read().strip()
                start = time.time()
                result = pipe(f"<s>[INST] {text} [/INST]", temperature=0.3)
                result = result[0]['generated_text']
                print('generate text:')
                print(result)
                answer = result.split('[/INST]')[1].split('</s>')[0].strip()
                print('Answer:', answer)
                print('time:', time.time() - start)
            else:
                print('Xin cảm ơn!')
                exit(0)
    else:
        result = pipe(prompt)
        result = result[0]['generated_text']
        answer = result.split('[/INST]')[1].split('</s>')[0].strip()
        return answer


from utils.data_preprocessing import custom_load_dataset
import json
save_path = '/mnt/md1/check_point_text_recognition/ckpt_chatbot/241216_llama7b/ckpt_end_training/compare_results.json'

def make_a_file_compare_step_1():
    
    results = []
    train_dataset, val_dataset, test_dataset = custom_load_dataset(data_path=cfg.data_path, max_seq_length=cfg.max_seq_length)
    with open(save_path, 'w', encoding='utf-8') as json_file:
        for example in test_dataset['text']:  
            prompt = example.split('[/INST]')[0] + '[/INST]'
            answer = example.split('[/INST]')[1].replace('</s>', '')
            base_res = direct_inference(prompt=prompt)
            results.append({
                'prompt': prompt,
                'answer': answer,
                'base':{
                    'response': base_res,
                    'score': ''
                },
                'fine-tuning': {
                    'response': '',
                    'score': ''
                }
            })
            print('done one question.')
        json.dump(results, json_file, ensure_ascii=False)
        
def make_a_file_compare_step_2():
    with open(save_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    for item in data:
        response = direct_inference(prompt=item['prompt'])
        item['fine-tuning']['response'] = response
        # item['base']['response'] = response
        print('done one question.')
    
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False)

def chat():
    while True:
        prompt = input("Type your question: ")
        if prompt != '0':
            start = time.time()
            answer = generate(prompt)
            print('Answer:', answer)
            print('time:', time.time() - start)
        else:
            print('Xin cảm ơn!')
            exit(0)
if __name__ == '__main__':
    # direct_inference(load_file=True)
    # direct_inference(load_file=False)
    chat()
    # make_a_file_compare_step_2()