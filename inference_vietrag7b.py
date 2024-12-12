from transformers import GenerationConfig, TextStreamer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch

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

torch_dtype = torch.bfloat16
model_id = "llm4fun/vietrag-7b-v1.0"
device = "cuda"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    torch_dtype=torch_dtype
)

model = model.eval().to(device)

def generate(prompt, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            # temperature=0.2,
            # top_p=0.95,
            # top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )

    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

output = generate(prompt)
