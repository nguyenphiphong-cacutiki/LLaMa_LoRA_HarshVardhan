from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from peft import PeftModel
import torch

# Định nghĩa đường dẫn đến mô hình đã fine-tune và mô hình gốc
model_id = "llm4fun/vietrag-7b-v1.0"
lora_checkpoint_path = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/241213_vietrag7b/checkpoint-1114"

# Tải tokenizer và mô hình gốc
tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

# Tải mô hình đã fine-tune với LoRA
model = PeftModel.from_pretrained(model, lora_checkpoint_path)

# Đặt mô hình vào chế độ đánh giá
model.eval()
# Đảm bảo mô hình sử dụng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)



# Hàm tạo văn bản từ mô hình
def generate(prompt, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    # input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.13,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": False,  # Thay đổi thành True để sử dụng temperature và top_p
            "temperature": 0.2,  # Thêm temperature nếu bạn muốn ngẫu nhiên hóa đầu ra
            "top_p": 0.9,  
            "use_cache": True,
        }
        generated = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

    gen_tokens = generated[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    return output.strip()

if __name__ == '__main__':
    # Tạo prompt cho câu hỏi và ngữ cảnh
    question = "<your-question>"
    context = "<your-context>"
    instruction = 'You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.'
    input = f"Dựa vào một số ngữ cảnh được cho dưới đây, trả lời câu hỏi ở cuối.\n\n{context}\n\nQuestion: {question}"
    prompt_template = (
        "### System:\n"
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )
    prompt = prompt_template.format(instruction=instruction, input=input, output='')
    prompt = '''
    ### Instruction:
    You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.

    ### Input:
    Dựa vào một số ngữ cảnh được cho dưới đây, trả lời câu hỏi ở cuối.

    Tài liệu 1:
    (0, 'Điều 2. Khối lượng kiến thức Học phần Tiếng Anh tăng cường\n\nHọc phần Tiếng Anh tăng cường (TC) là học phần Tiếng Anh bổ trợ giúp sinh viên đạt ngưỡng tối thiểu trước khi bắt đầu học các học phần Tiếng Anh trong chương trình đào tạo. Tổng khối lượng học phần Tiếng Anh tăng cường là 6 tín chỉ.')

    Tài liệu 2:
    Chương 3. XỬ LÝ VI PHẠM

    Điều 12. Xử lý cán bộ, giảng viên vi phạm quy định

    1. Những cán bộ, giảng viên tham gia kỳ thi kết thúc học phần môn học có hành vi vi phạm Quy định (bị phát hiện trong khi làm nhiệm vụ hoặc phát hiện sau kỳ thi), nếu có đủ chứng cứ, tuỳ theo mức độ sẽ bị xử lý kỷ luật theo các hình thức sau đây:

    a. Khiển trách đối với cán bộ, giảng viên vi phạm một trong các lỗi sau:

    - Ra đề thi không phù hợp với nội dung giảng dạy học phần;

    - Để cho sinh viên/học viên không đủ điều kiện dự thi vào phòng thi trực tuyến;

    - Để cho sinh viên tự do sử dụng điện thoại di động hoặc các phương tiện kỹ thuật thu, phát, truyền tin, ghi âm ...trái quy định tại phòng thi trực tuyến, bị cán bộ giám sát thi phát hiện và lập biên bản;

    - Chấm thi hoặc cộng điểm bài thi có nhiều sai sót.
    Question: Câu hỏi 5:  Ai là người quyết định hình thức kỷ luật đối với cán bộ, giảng viên vi phạm quy định trong kỳ thi?  Quyết định đó dựa trên cơ sở gì?

    ### Response:
    '''
    # Tạo câu trả lời
    output = generate(prompt)
    print(output)
