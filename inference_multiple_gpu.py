# 1.Import Necessary libraries
import os
import torch
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

# 2. Model and Dataset Configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "Llama-2-7b-chat-finetune-qlora"
device_map = {"":0}
# config for multi gpu
# Initialize the process group for distributed training
dist.init_process_group(backend='nccl')  # NCCL is commonly used for GPU communication

# Get the rank of the current process and set the device
rank = dist.get_rank()
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')


# 3. Tokenizer and PEFT configuration
#Load LLama tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. load model for inference

# Step 1: Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,  # The original base model's name or path
    device_map=None,  # Or specify your device
).to(device)

# Step 2: Load the fine-tuned LoRA model (saved from trainer.model.save_pretrained)
model = PeftModel.from_pretrained(base_model, new_model).to(device)  # `new_model` is the path where you saved the model

# Step 3: Merge the LoRA weights with the base model
model = model.merge_and_unload()
# Wrap the model with DDP
model = DDP(model, device_ids=[rank], output_device=rank)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# 5. Run text generation pipeline with our next model
prompt = "How can I learn to optimize my webpage for search engines?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
if rank == 0:
    print(result[0]['generated_text'])


# Cleanup and destroy the process group
dist.barrier()
dist.destroy_process_group()
