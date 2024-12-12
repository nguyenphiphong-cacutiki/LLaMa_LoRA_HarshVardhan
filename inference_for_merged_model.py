# 1. Import các thư viện cần thiết
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# 2. Định nghĩa thư mục lưu mô hình đã merge
save_dir = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/241202_llama7bchathf/push_hug_demo"  # Đặt đường dẫn tới thư mục chứa mô hình đã merge và tokenizer

# 3. Load Tokenizer và Model đã merge từ thư mục đã lưu
tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load mô hình đã merge (không cần phải load base model và merge lại)
model = AutoModelForCausalLM.from_pretrained(
    save_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,  # Sử dụng FP16 nếu bạn muốn tiết kiệm bộ nhớ GPU
    device_map="auto",  # Cấu hình auto device map nếu sử dụng nhiều GPU hoặc chuyển sang CPU
)

# 4. Khởi tạo Inference Pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2048)

# 5. Thực hiện inference
while True:
    prompt = input("Type your question (or '0' to exit): ")
    if prompt != '0':
        start = time.time()
        # Chạy inference
        result = pipe(f"<s>[INST] {prompt} [/INST]")  # Cách bạn chuẩn bị input (bằng cách sử dụng [INST] tag)
        generated_text = result[0]['generated_text']
        
        # Trích xuất câu trả lời từ text generated
        answer = generated_text.split('[/INST]')[1].split('</s>')[0].strip()
        
        print(f'Answer: {answer}')
        print(f'Time: {time.time() - start} seconds')
    else:
        print('Thank you!')
        break
