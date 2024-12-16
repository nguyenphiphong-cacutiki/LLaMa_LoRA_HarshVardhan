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
new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/241202_llama7bchathf/checkpoint-2700"
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


pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=2048)

def generate(prompt):
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    result = result[0]['generated_text']
    answer = result.split('[/INST]')[1].split('</s>')[0].strip()
    return answer

if __name__ == '__main__':
    prompt_path = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/prompt_for_test.txt"
    while True:
        prompt = input("Type your question: ")
        if prompt != '0':
            with open(prompt_path, 'r') as file:
                text = file.read().strip()
            start = time.time()
            result = pipe(f"<s>[INST] {text} [/INST]")
            result = result[0]['generated_text']
            answer = result.split('[/INST]')[1].split('</s>')[0].strip()
            print('Answer:', answer)
            print('time:', time.time() - start)
        else:
            print('Xin cảm ơn!')
            exit(0)