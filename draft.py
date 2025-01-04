



prompt_template = (
    "### System:\n"
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n\n\n"
    "### Instruction:\nYou are now a dedicated assistant for TLU. Provide a detailed answer so user don’t \
    need to search outside to understand the answer.\n\n"
    "### Input:\n{Dựa vào một số Tài liệu được cho dưới đây, trả lời câu hỏi ở cuối. nếu bạn thấy Tài liệu \
    không liên quan đến câu hỏi thì phải giải thích tại sao lại không thể trả lời.\n\n{context}\n\nQuestion: {question}\n\n"
    "### Response:\n{answer}"
)