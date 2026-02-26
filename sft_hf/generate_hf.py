from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("sft-merged-hf", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("sft-merged-hf")
model.to("mps")
model.eval()

prompt = "Below is an instruction that describes a task.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=50)

print(tokenizer.decode(out[0], skip_special_tokens=True))
