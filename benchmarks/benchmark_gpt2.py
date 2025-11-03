import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scratchformers.models.gpt2 import create_gpt2_from_hf_weights

custom_gpt = create_gpt2_from_hf_weights().to(0)
compiled_custom_gpt = torch.compile(custom_gpt)

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    dtype=torch.float32,
    device_map="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model.eval()

test_str = "Benchmarking to see how this works"
input_ids = tokenizer(test_str, return_tensors="pt").to(model.device)

# Warmup runs
print("Running warmup...")
for _ in range(3):
    _ = model(**input_ids, cache_implementation="static")
    _ = custom_gpt(input_ids["input_ids"])
    _ = compiled_custom_gpt(input_ids["input_ids"])

# Time reference model
num_runs = 10
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(num_runs):
    reference_output = model(**input_ids, cache_implementation="static")
end.record()
torch.cuda.synchronize()
print(f"reference took {start.elapsed_time(end) / num_runs:.4f} ms per forward pass")

# Time custom model
start.record()
for _ in range(num_runs):
    custom_gpt_output = custom_gpt(input_ids["input_ids"])
end.record()
torch.cuda.synchronize()
print(f"custom took {start.elapsed_time(end) / num_runs:.4f} ms per forward pass")

# Time custom model
start.record()
for _ in range(num_runs):
    custom_gpt_output = compiled_custom_gpt(input_ids["input_ids"])
end.record()
torch.cuda.synchronize()
print(f"compiled custom took {start.elapsed_time(end) / num_runs:.4f} ms per forward pass")

assert torch.allclose(reference_output.logits, custom_gpt_output, rtol=1e-2)
