"""Debug script to identify where monitoring hangs."""
import os
import sys

# Enable debug output
os.environ["MON_ENGINE_DEBUG"] = "1"
os.environ["MON_NATIVE_TO_CPU"] = "1"
os.environ["MON_NATIVE_CALLBACK"] = "1"
os.environ["MON_NATIVE_BUILDER"] = "1"
os.environ["MON_NATIVE_BATCH"] = "0"
os.environ["MON_NATIVE_AUTOCLEAR"] = "1"

import torch
print("[DEBUG] torch imported")

from transformers import AutoTokenizer
print("[DEBUG] AutoTokenizer imported")

from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel
print("[DEBUG] HookedGPT2LMHeadModel imported")

from monitoring import MonitoringConfig, MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection
from monitoring.generate import generate_with_monitoring
print("[DEBUG] monitoring modules imported")

def main():
    device = torch.device("cuda")
    model_name = "gpt2"

    print("[DEBUG] Creating MonitoringEngine...")
    cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )
    engine = MonitoringEngine(
        async_enabled=True,
        config=cfg,
        model_id=model_name,
        db_config=None,  # no-db mode
    )
    print(f"[DEBUG] Engine created, using_native_backend={engine._using_native_backend}")

    print("[DEBUG] Loading model...")
    model = HookedGPT2LMHeadModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model.to(device).eval()
    model.monitoring_engine = engine
    print("[DEBUG] Model loaded")

    print("[DEBUG] Preparing monitoring for model...")
    engine.prepare_for_model(model)
    print("[DEBUG] Monitoring prepared")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("[DEBUG] Tokenizer loaded")

    prompts = ["Hello, world!"]
    print(f"[DEBUG] Encoding {len(prompts)} prompts...")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print(f"[DEBUG] input_ids shape: {input_ids.shape}")

    print("[DEBUG] Starting generate_with_monitoring...")
    sys.stdout.flush()

    with torch.no_grad():
        output = generate_with_monitoring(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"[DEBUG] Generation complete! Output shape: {output.shape}")
    print(f"[DEBUG] Output: {tokenizer.decode(output[0])}")

    print("[DEBUG] Closing engine...")
    engine.close()
    print("[DEBUG] Done!")

if __name__ == "__main__":
    main()
