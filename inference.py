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
import config_train as cfg

accelerator = Accelerator()

# 2. Model and Dataset Configuration
model_name = cfg.model_name
# new_model = "Llama-2-7b-chat-finetune-qlora"
# new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/checkpoint-53390"
new_model = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/241202/checkpoint-2700"
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

# 5. Run text generation pipeline with our next model
# prompt = "How can I learn to optimize my webpage for search engines?"

prompt = ""
prompt = '''
Hãy trả lời câu hỏi dựa trên các tài liệu được cung cấp. nếu bạn thấy tài liệu không liên quan đến câu hỏi thì chỉ cần trả lời 'Không có thông tin về câu hỏi này trong tài liệu được cung cấp'.

Các tài liệu:

Tài liệu 1:
Điều 2. Khối lượng kiến thức Học phần Tiếng Anh tăng cường

Học phần Tiếng Anh tăng cường (TC) là học phần Tiếng Anh bổ trợ giúp sinh viên đạt ngưỡng tối thiểu trước khi bắt đầu học các học phần Tiếng Anh trong chương trình đào tạo. Tổng khối lượng học phần Tiếng Anh tăng cường là 6 tín chỉ.

Tài liệu 2:
QUY ĐỊNH

KHỐI LƯỢNG VÀ TỔ CHỨC ĐÀO TẠO HỌC PHẦN TIẾNG ANH TĂNG CƯỜNG

(Ban hành kèm theo Quyết định số: 1767 /QĐ-ĐHTL ngày 29 tháng 08 năm 2019 của Hiệu trưởng Trường Đại học Thủy lợi)

Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng

1. Văn bản này quy định về tổ chức, quản lý học phần Tiếng Anh tăng cường áp dụng đối với sinh viên trình độ Đại học hệ chính quy kể từ khóa tuyển sinh năm 2019 hoặc tuyển sinh Khóa trước 2019 nhưng học theo chương trình của Khóa tuyển sinh 2019 trở về sau.

2. Những sinh viên được nêu ở khoản 1 điều 1 có điểm thi môn Tiếng Anh THPT dưới 4 (< 4 điểm) sẽ thuộc diện bắt buộc tham gia các lớp học Tiếng Anh tăng cường do nhà trường tổ chức; các đối tượng còn lại cũng có thể đăng kí học nếu có nhu cầu.

3. Sinh viên có chứng chỉ ngoại ngữ theo quy định tại Điều 3 của quy định này được miễn học Học phần Tiếng Anh tăng cường và được đăng ký học Học phần Tiếng Anh bắt buộc trong chương trình đào tạo.

 


Câu hỏi:
Mục đích của học phần Tiếng Anh tăng cường là gì?
'''
pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=2048)
while True:
    # prompt = input("Type your question: ")
    if prompt != 'exit':
        start = time.time()
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        result = result[0]['generated_text']
        answer = result.split('[/INST]')[1].split('</s>')[0].strip()
        print('Answer:', answer)
        print('time:', time.time() - start)
        break
    else:
        print('Xin cảm ơn!')
        exit(0)