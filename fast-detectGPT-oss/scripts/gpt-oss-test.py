# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

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

outputs = model.generate(**inputs, max_new_tokens=40)
output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
match = re.search(r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>", output_text, re.DOTALL)
if match:
    answer = match.group(1).strip()
    print(answer)
else:
    print("No final message found.")