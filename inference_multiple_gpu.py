from accelerate import Accelerator
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
import config_train as cfg


new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/checkpoint-53390"
# Khởi tạo Accelerator
accelerator = Accelerator()

# Load model và tokenizer
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Wrap model và tokenizer bằng Accelerator
model, tokenizer = accelerator.prepare(model, tokenizer)

# Inference
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

while True:
    prompt = input("Type your question: ")
    if prompt != 'exit':
        start = time.time()
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        result = result[0]['generated_text']
        answer = result.split('[/INST]')[1].split('</s>')[0].strip()
        print('Answer:', answer)
        print('time:', time.time() - start)
    else:
        print('Xin cảm ơn!')
        exit(0)
