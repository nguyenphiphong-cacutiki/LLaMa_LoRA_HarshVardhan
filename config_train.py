import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Model and Dataset Configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
# dataset_name = "mlabonne/guanaco-llama2-1k"
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
import datetime
day = datetime.datetime.now().strftime('%y%m%d')
output_dir = f"/mnt/md1/check_point_text_recognition/ckpt_chatbot/{day}"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
new_model = os.path.join(output_dir, "Llama-2-7b-chat-finetune-qlora")
data_path = '/mnt/md1/check_point_text_recognition/data_chatbot/data_llama_7b_chat_hf_time_241202-190841.json'

#number of training epochs
num_train_epochs = 5

#enable fp16/bf16 training (set bf16 to True when using A100 GPU in google colab)
fp16 = True
bf16 = False

#batch size per GPU for training
per_device_train_batch_size = 1

#batch size per GPU for evaluation
per_device_eval_batch_size = 1

#gradient accumulation steps - No of update steps
gradient_accumulation_steps = 8

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
seed = 1

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
max_seq_length = 1536

packing = False
save_total_limit=3

#load the entire model on the GPU
# device_map = {"":0} 
# device_map = {0: [0], 1: [1]}
device_map = 'auto' # Automatically distribute models across available GPUs