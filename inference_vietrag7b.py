from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from peft import PeftModel
import torch

# Định nghĩa đường dẫn đến mô hình đã fine-tune và mô hình gốc
model_id = "llm4fun/vietrag-7b-v1.0"
lora_checkpoint_path = "/mnt/md1/check_point_text_recognition/ckpt_chatbot/250102_vietrag7b/ckpt_end_training"

# Tải tokenizer và mô hình gốc
tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

# Tải mô hình đã fine-tune với LoRA
# Step 2: Load the fine-tuned LoRA model (saved from trainer.model.save_pretrained)
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

# Step 3: Merge the LoRA weights with the base model
model = model.merge_and_unload()
# Đặt mô hình vào chế độ đánh giá
base_model.eval()
# Đảm bảo mô hình sử dụng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

def format_prompt_for_answer_task(question, context):
    # Tạo prompt cho câu hỏi và ngữ cảnh
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
    return prompt

# Hàm tạo văn bản từ mô hình
def generate(question, context, max_new_tokens=620):
    prompt = format_prompt_for_answer_task(question=question, context=context)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    # input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.13,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": False,  # Thay đổi thành True để sử dụng temperature và top_p
            # "temperature": 0.2,  # Thêm temperature nếu bạn muốn ngẫu nhiên hóa đầu ra
            # "top_p": 0.9,  
            "use_cache": True,
        }
        generated = base_model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

    gen_tokens = generated[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    return output.strip()

if __name__ == '__main__':
    question = '''
trường hợp nào được miễn học phí
'''
    context = '''
Tài liệu :\nĐiều 8. Cơ chế miễn, giảm học phí\n\n1. Đối với sinh viên học theo chương trình truyền thống\n\n- \
Những môn học đăng ký học lần đầu: được miễn giảm học phí ở học kỳ chính theo quy định. Trong học kỳ song song \
với học kỳ chính, học kỳ hè, cũng được xét miễn giảm tương ứng như với học kỳ chính liền kề trước đó.\n\n \
- Những môn học đăng ký học lần thứ hai trở đi: đóng 100% học phí theo quy định.\n\n2. Đối với sinh viên học theo \
chương trình tiên tiến và sinh viên liên thông cao đẳng lên đại học\n\na/ Sinh viên thuộc diện miễn 100% học phí\n\n \
- Đối với các môn học lần đầu: sinh viên được miễn 100% học phí bằng mức học phí quy định của chương trình truyền thống. \
Sinh viên phải đóng phần chênh học phí giữa mức học phí của chương trình tiên tiến hoặc chương trình liên thông với mức \
học phí được miễn.\n\n- Đối với các môn học lần thứ hai trở đi: đóng 100% học phí theo quy định.\n\nb/ Sinh viên thuộc diện \
giảm học phí\n\n- Đối với các các môn học lần đầu: sinh viên được giảm học phí bằng 70% hoặc 50% mức học phí quy định của \
chương trình truyền thống. Sinh viên phải đóng phần chênh học phí giữa mức học phí của chương trình tiên tiến hoặc chương \
trình liên thông với mức học phí được giảm.\n\n- Đối với các môn học lần thứ hai trở đi: đóng 100% học phí theo quy định.\n\n \

'''
    answer = generate(question=question, context=context)
    print('Answer:', answer)
