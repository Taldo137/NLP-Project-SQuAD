import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# ניקוי זיכרון מקדים
gc.collect()

model_name = 'meta-llama/Llama-3.2-3B-Instruct'
print(f"1. Downloading/Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

print("2. Loading Model (Clean Load)...")
# אנחנו משתמשים בשיטה הכי פשוטה: טעינה ל-CPU בזהירות.
# בלי device_map="auto" ובלי offload - זה לפעמים גורם ליותר בעיות מתועלת.
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    dtype=torch.float16, 
    token=True,
    low_cpu_mem_usage=True # זה הדבר היחיד שחובה כדי לא לקרוס
)

model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

print("3. Model Loaded! Generating response...")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Are you working?"},
]

model_input = tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, return_tensors="pt",
    tokenize=False, padding=True, truncation=True)

inputs = tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    max_new_tokens=50, do_sample=False)

response = str(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f'\nSUCCESS: {response}')