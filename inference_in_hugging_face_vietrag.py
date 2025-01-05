from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch

model_id = "phongnp2010/vietrag-finetuning"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    torch_dtype=torch.bfloat16,
    device_map='auto',
)


model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"

def format_prompt(question, context):
        instruction = 'You are now a dedicated assistant for TLU. Provide a detailed answer so user don’t need to search \
        outside to understand the answer.'
        input = f"Dựa vào một số Tài liệu được cho dưới đây, trả lời câu hỏi ở cuối. nếu bạn thấy Tài liệu không liên quan đến \
        câu hỏi thì phải giải thích tại sao lại không thể trả lời.\n\n{context}\n\nQuestion: {question}"
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

def generate(question, context, max_new_tokens=620):
    prompt = format_prompt(question=question, context=context)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.13,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": False, 
            "use_cache": True,
        }
        generated = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

    gen_tokens = generated[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]

    return output.strip().replace('</s>', '')

question = '''
Điểm rèn luyện của năm học được tính toán ra sao nếu sinh viên học chưa đủ một năm?
'''

context = '''
Tài liệu :Điều 13. Cách đánh giá kết quả rèn luyện1. Điểm rèn luyện của học kỳ là tổng điểm đạt được của 5 nội dung đánh giá quy định tại điều 4, điều 5, điều 6, điều 7 và điều 8; trong đó điểm thành phần và điểm tổng của các nội dung đánh giá được làm tròn đến 0,5 và không vượt quá điểm quy định.2. Điểm rèn luyện của năm học là trung bình cộng của điểm rèn luyện các học kỳ của năm học đó. Nếu năm học sinh viên học chưa đủ một năm học (như sinh viên ngừng học, sinh viên năm cuối) thì kỳ sinh viên theo học được tính tròn thành 1 năm học.3. Điểm rèn luyện toàn khoá là trung bình cộng của điểm rèn luyện các học kỳ sinh viên theo học trong toàn khoá học.4. Điểm rèn luyện quy đổi được tính từng học kỳ theo công thức sau và làm tròn đến 2 chữ số thập phân:T — i ĐRLgi = 100Trong đó:- r i là điểm rèn luyện của kỳ học thứ i- ĐRLqđ i là điểm rèn luyện quy đổi của kỳ học thứ i
'''

answer = generate(question=question, context=context)
print('Answer:', answer)
