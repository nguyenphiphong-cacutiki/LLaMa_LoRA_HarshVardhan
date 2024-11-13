# 1.Import Necessary libraries
import os
import torch
from datasets import load_dataset
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
from trl import SFTTrainer, SFTConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# 2. Model and Dataset Configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune-qlora"

# 3. QLoRA parameters
lora_r = 32 #lora attention dimension/ rank
lora_alpha = 8 #lora scaling parameter
lora_dropout = 0.1 #lora dropout probability

# 4. BitsAndBytes Configuration
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# 5. Training Arguments
#output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
#number of training epochs
num_train_epochs = 5
#enable fp16/bf16 training (set bf16 to True when using A100 GPU in google colab)
fp16 = False
bf16 = False
#batch size per GPU for training
per_device_train_batch_size = 1
#batch size per GPU for evaluation
per_device_eval_batch_size = 1
#gradient accumulation steps - No of update steps
gradient_accumulation_steps = 1
#learning rate
learning_rate = 2e-4
#weight decay
weight_decay = 0.001
#Gradient clipping(max gradient Normal)
max_grad_norm = 0.3
#optimizer to use
optim = "paged_adamw_32bit"
#learning rate scheduler
lr_scheduler_type = "cosine"
#seed for reproducibility

#Number of training steps
max_steps = -1
#Ratio of steps for linear warmup
warmup_ratio = 0.03
#group sequnces into batches with same length
group_by_length = True
#save checkpoint every X updates steps
save_steps = 0
#Log at every X updates steps
logging_steps = 100

# 6. SFT parameters
#maximum sequence length to use
max_seq_length = 1024
packing = False
#load the entire model on the GPU
device_map = {"":0}

# 7. Load Dataset and Model
#load dataset
dataset = load_dataset(dataset_name,split = "train")
#load tokenizer and model with QLoRA config
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit = use_4bit,
    bnb_4bit_quant_type = bnb_4bit_quant_type,
    bnb_4bit_compute_dtype = compute_dtype,
    bnb_4bit_use_double_quant = use_nested_quant,)

#cheking GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("="*80)
        print("Your GPU supports bfloat16, you are getting accelerate training with bf16= True")
        print("="*80)

#load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = device_map,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 8. Tokenizer and PEFT configuration
#Load LLama tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#Load QLoRA config
peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r  = lora_r,
    bias = "none",
    task_type = "CAUSAL_LM",
)

# 9. Training
#Set Training parameters
training_arguments = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    fp16 = fp16,
    bf16 = bf16,
    max_grad_norm = max_grad_norm,
    weight_decay = weight_decay,
    lr_scheduler_type = lr_scheduler_type,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    max_steps = max_steps,
    report_to = "tensorboard",
    # eval_strategy="epoch",  # Evaluate at the end of every epoch
    # save_strategy="epoch",  # Save checkpoint at the end of every epoch
    # save_total_limit=3,
    # load_best_model_at_end=True,  # Load the best model at the end of training
    # metric_for_best_model="eval_loss", # Metric used to determine the best model
    # greater_is_better=False, # Since lower eval_loss is better, this is False
)
# Initialize the distributed environment
dist.destroy_process_group()
if dist.is_initialized():
    print("Process group already initialized.")
else:
    dist.init_process_group(backend='nccl')
    print("Process group initialized.")
# Get the local rank from environment variables (set by torchrun or other launcher)
local_rank = int(os.getenv('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
# Initialize DistributedDataParallel
# model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# model = DDP(model).to(device)
#SFT Trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    peft_config = peft_config,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_arguments,
    tokenizer = tokenizer,
    packing = packing,
)
#Start training
trainer.train()
'''
    + After creating the SFTTrainer (which is a trainer designed for fine-tuning models using
    PEFT—Parameter-Efficient Fine-Tuning—with LoRA), you fine-tune the model.
    + This means the trainer.train() step modifies the model variable by applying the LoRA-based
    fine-tuning.
At this point:
    + The base model remains unchanged, but LoRA layers are applied on top of it to store the
    fine-tuning modifications in a separate, efficient manner.
    + The model variable in trainer is now a PEFT model that combines the base model with the
    LoRA fine-tuned layers.
'''

# 10. Saving the model and Testing
#save trained model
trainer.model.save_pretrained(new_model)
'''
+ Here, you save the fine-tuned model, which includes both the base model and LoRA layers.
+ The trainer.model is a PEFT model at this point (a combination of the base model and LoRA
weights). Saving the model preserves the state of the model and LoRA weights, but they are not
merged yet unless you explicitly do so using something like model.merge_and_unload().
'''

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "How can I learn to optimize my webpage for search engines?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

# 11. Store New Llama2 Model
# Reload model in FP16 and merge it with LoRA weights
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map=device_map,
# )
# model = PeftModel.from_pretrained(base_model, new_model)
# model = model.merge_and_unload()

# # Reload tokenizer to save it
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"