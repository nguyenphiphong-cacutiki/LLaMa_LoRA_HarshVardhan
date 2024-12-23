# 1.Import Necessary libraries
import os
import torch
from datasets import load_dataset
import config_llama as cfg
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
from utils.data_preprocessing import custom_load_dataset
# 7. Load Dataset and Model
#load dataset
# dataset = load_dataset(dataset_name,split = "train")
# dataset = load_dataset('json', data_files=cfg.data_path, split="train")

# dataset = custom_load_dataset(data_path=cfg.data_path, max_seq_length=cfg.max_seq_length)

# train_val_split = dataset.train_test_split(test_size=0.25, seed=42)
# train_dataset = train_val_split['train'].shuffle(seed=42)

# val_test_dataset = train_val_split['test'].train_test_split(test_size=0.2, seed=42)
# val_dataset = val_test_dataset['train'].shuffle(seed=42)
# test_dataset = val_test_dataset['test'].shuffle(seed=42)

train_file = '/mnt/md1/check_point_text_recognition/data_chatbot/qa_data_llama/llama_qa_data_241223-163342_train.json'
val_file = '/mnt/md1/check_point_text_recognition/data_chatbot/qa_data_llama/llama_qa_data_241223-163342_val.json'
# train_dataset, val_dataset, test_dataset = custom_load_dataset(data_path=cfg.data_path, max_seq_length=10000)
train_dataset = load_dataset("json", data_files=train_file,split='train')
val_dataset = load_dataset("json", data_files=val_file,split='train')

#load tokenizer and model with QLoRA config
compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = cfg.use_4bit,
    bnb_4bit_quant_type = cfg.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype = compute_dtype,
    bnb_4bit_use_double_quant = cfg.use_nested_quant,)

#cheking GPU compatibility with bfloat16
if compute_dtype == torch.float16 and cfg.use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("="*80)
        print("Your GPU supports bfloat16, you are getting accelerate training with bf16= True")
        print("="*80)

#load base model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    quantization_config = bnb_config,
    device_map = cfg.device_map,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# 8. Tokenizer and PEFT configuration
#Load LLama tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name,trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#Load QLoRA config
peft_config = LoraConfig(
    lora_alpha = cfg.lora_alpha,
    lora_dropout = cfg.lora_dropout,
    r  = cfg.lora_r,
    bias = "none",
    task_type = "CAUSAL_LM",
)

# 9. Training
#Set Training parameters
training_arguments = TrainingArguments(
    output_dir = cfg.output_dir,
    num_train_epochs = cfg.num_train_epochs,
    per_device_train_batch_size = cfg.per_device_train_batch_size,
    gradient_accumulation_steps = cfg.gradient_accumulation_steps,
    optim = cfg.optim,
    # save_steps = save_steps,
    logging_steps = cfg.logging_steps,
    learning_rate = cfg.learning_rate,
    fp16 = cfg.fp16,
    bf16 = cfg.bf16,
    max_grad_norm = cfg.max_grad_norm,
    weight_decay = cfg.weight_decay,
    lr_scheduler_type = cfg.lr_scheduler_type,
    warmup_ratio = cfg.warmup_ratio,
    group_by_length = cfg.group_by_length,
    max_steps = cfg.max_steps,
    report_to = "tensorboard",
    eval_strategy="epoch",  # Evaluate at the end of every epoch
    save_strategy="epoch",  # Save checkpoint at the end of every epoch
    save_total_limit=cfg.save_total_limit,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss", # Metric used to determine the best model
    greater_is_better=False, # Since lower eval_loss is better, this is False

    # add some config
    dataloader_num_workers=1, # Increase the number of workers to increase IO performance if there are multiple GPUs
    local_rank=-1, # Hugging Face Provides Multi-GPU Distribution via Trainer
)

#SFT Trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    peft_config = peft_config,
    dataset_text_field = "text",
    max_seq_length = cfg.max_seq_length,
    args = training_arguments,
    tokenizer = tokenizer,
    packing = cfg.packing,
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
trainer.model.save_pretrained(cfg.new_model)
'''
+ Here, you save the fine-tuned model, which includes both the base model and LoRA layers.
+ The trainer.model is a PEFT model at this point (a combination of the base model and LoRA
weights). Saving the model preserves the state of the model and LoRA weights, but they are not
merged yet unless you explicitly do so using something like model.merge_and_unload().
'''

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


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