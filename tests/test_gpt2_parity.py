import pytest
import torch

from tests._requirements import require_model_cache

# Runs on CPU but pulls real gpt2 weights via from_pretrained -> `hf`, not `cpu`.
pytestmark = [pytest.mark.hf, pytest.mark.framework_fork, require_model_cache("gpt2")]


def _import_gpt2_modules():
    try:
        from transformers import AutoTokenizer
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as HFOriginalGPT2
        from transformers.models.gpt2_p.modeling_gpt2 import GPT2LMHeadModel as HFModifiedGPT2
    except ImportError as exc:
        pytest.skip(f"modified transformers fork required: {exc}")
    return AutoTokenizer, HFOriginalGPT2, HFModifiedGPT2


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    AutoTokenizer, _, _ = _import_gpt2_modules()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_models(seed: int = 0):
    _, HFOriginalGPT2, HFModifiedGPT2 = _import_gpt2_modules()
    torch.manual_seed(seed)
    hf_original = HFOriginalGPT2.from_pretrained("gpt2")
    torch.manual_seed(seed)
    hf_modified = HFModifiedGPT2.from_pretrained("gpt2")
    return hf_original.eval(), hf_modified.eval()


def generate_text(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def model_logits(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


def test_prefill_logits_identical(gpt2_tokenizer):
    hf_original, hf_modified = build_models()
    prompt = "Transformers are powerful models."
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")

    original_logits = model_logits(hf_original, inputs)
    modified_logits = model_logits(hf_modified, inputs)

    assert torch.allclose(original_logits, modified_logits, atol=1e-5), "Prefill logits should match"


def test_generate_identical_outputs(gpt2_tokenizer):
    hf_original, hf_modified = build_models()
    prompt = "Once upon a time"

    original_text = generate_text(hf_original, gpt2_tokenizer, prompt)
    modified_text = generate_text(hf_modified, gpt2_tokenizer, prompt)

    assert original_text == modified_text, "Generated text should match"
