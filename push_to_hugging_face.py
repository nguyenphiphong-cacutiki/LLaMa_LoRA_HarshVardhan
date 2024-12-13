# 1.Import Necessary libraries
import os
import torch
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM,
    logging,
    )
from peft import LoraConfig, PeftModel
import torch.distributed as dist
from accelerate import Accelerator
import config_vietrag as cfg
from huggingface_hub import upload_folder

accelerator = Accelerator()

# 2. Model and Dataset Configuration
model_name = cfg.model_name
# new_model = "Llama-2-7b-chat-finetune-qlora"
# new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/checkpoint-53390"
new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/241202_llama7bchathf/Llama-2-7b-chat-finetune-qlora"
save_directory = '/mnt/md1/check_point_text_recognition/ckpt_chatbot/241202_llama7bchathf/push_hug_demo'
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

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

repo_name = 'phongnp2010/chatbot-llama-7b-chathf'
upload_folder(
    repo_id=repo_name,
    folder_path=save_directory,
    path_in_repo="."  # Thư mục gốc trong repo (có thể thay đổi nếu cần)
)
# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

