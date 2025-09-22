# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch

# Remove bold (**text**) and italic (*text*) Markdown
def remove_markdown_styles(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # italic
    return text


tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
messages = [
    {"role": "user", "content": "What is the capital of France?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)  # Increase this value
output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print("DEBUG OUTPUT:", repr(output_text))  # See the raw output

# regex (case insensitive) to extract text between <|start|>assistant<|channel|>final<|message|> and <|return|>
match = re.search(
    r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>",
    output_text,
    re.DOTALL | re.IGNORECASE,
)

if match:
    answer = match.group(1).strip()
    clean_answer = remove_markdown_styles(answer)
    print(clean_answer)
else:
    print("No final message found.")







# --- Compute and print the loss of a sample sentence ---

def get_ll(args, scoring_model, scoring_tokenizer, text):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids
        return -scoring_model(**tokenized, labels=labels).loss.item()


def get_lls(args, scoring_model, scoring_tokenizer, texts):
    return [get_ll(args, scoring_model, scoring_tokenizer, text) for text in texts]

# Example sentence to compute loss for
sample_sentence = "Paris is the capital of France."

# Prepare dummy args with device info
class Args:
    pass
args = Args()
args.device = model.device

# Compute loss using get_ll
loss = get_ll(args, model, tokenizer, sample_sentence)
print(f"Loss for sentence: '{sample_sentence}' -> {loss}")
