import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scratchformers.models.gpt2 import create_gpt2_from_hf_weights


def test_output_equivalence():
    custom_gpt = create_gpt2_from_hf_weights().to(0)

    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        dtype=torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model.eval()

    test_str = "Test to see whether we get correct logits"
    input_ids = tokenizer(test_str, return_tensors="pt").to(model.device)
    reference_output = model(**input_ids, cache_implementation="static")
    custom_gpt_output = custom_gpt(input_ids["input_ids"])
    assert torch.allclose(reference_output.logits, custom_gpt_output, rtol=1e-2)
