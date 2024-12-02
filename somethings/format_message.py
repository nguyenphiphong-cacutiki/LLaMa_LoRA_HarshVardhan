from transformers import AutoTokenizer
from textwrap import dedent
from huggingface_hub import login
 
login(
  token="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", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

# Initialize the tokenizer for a specific model, such as LLaMA 2
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

def format_example(row: dict):
    # Create a formatted prompt for the user input
    prompt = dedent(
        f"""
        {row['question']}
        Information:
        ```
        {row['context']}
        ```
        """
    )
    # Define the message template for the chat-based format
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["answer"]},
    ]
    # Apply a chat template provided by the tokenizer to the message
    return tokenizer.apply_chat_template(message, tokenize=False)

if __name__ == '__main__':
    row = {'question': 'some question', 'context': 'some context...........'}
    print(format_example(row))
