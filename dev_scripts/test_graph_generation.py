#!/usr/bin/env python3
"""Test script to verify CUDA Graph generation correctness."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_generation(use_graph=False):
    """Compare text generation with and without CUDA Graph."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if not use_graph:
            # Normal generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )
        else:
            # Manual decode loop with graph
            input_ids = inputs["input_ids"]
            
            # Prefill
            outputs = model(input_ids, use_cache=True)
            past = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = [next_token.item()]
            
            # Capture graph for decode
            static_token = torch.empty_like(next_token)
            static_past = tuple(
                (torch.empty_like(k), torch.empty_like(v))
                for k, v in past
            )
            
            # Copy initial values
            static_token.copy_(next_token)
            for (dst_k, dst_v), (src_k, src_v) in zip(static_past, past):
                dst_k.copy_(src_k)
                dst_v.copy_(src_v)
            
            graph = torch.cuda.CUDAGraph()
            
            # Capture
            with torch.cuda.graph(graph):
                graph_outputs = model(
                    static_token,
                    past_key_values=static_past,
                    use_cache=True
                )
                graph_logits = graph_outputs.logits
            
            # Decode loop with replay
            for _ in range(19):  # Generate 19 more tokens
                # Update input
                static_token.copy_(next_token)
                
                # Replay graph
                graph.replay()
                
                # Get next token from graph output
                next_token_logits = graph_logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated.append(next_token.item())
                
                # Update KV cache for next iteration
                next_past = graph_outputs.past_key_values
                for (dst_k, dst_v), (src_k, src_v) in zip(static_past, next_past):
                    # Check if shapes changed
                    if dst_k.shape != src_k.shape:
                        print(f"KV shape changed! {dst_k.shape} -> {src_k.shape}")
                        dst_k.resize_(src_k.shape)
                    dst_k.copy_(src_k)
                    if dst_v.shape != src_v.shape:
                        dst_v.resize_(src_v.shape)
                    dst_v.copy_(src_v)
            
            # Combine tokens
            input_list = input_ids.tolist()[0]
            outputs = torch.tensor([input_list + generated]).to(device)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    print("Testing normal generation...")
    normal_text = test_generation(use_graph=False)
    print(f"Normal: {normal_text}")
    
    print("\nTesting CUDA Graph generation...")
    try:
        graph_text = test_generation(use_graph=True)
        print(f"Graph:  {graph_text}")
        
        if normal_text == graph_text:
            print("\n✅ Generation results match!")
        else:
            print("\n❌ Generation results differ!")
            print("This indicates an issue with KV cache or logits handling in Graph mode.")
    except Exception as e:
        print(f"❌ Graph generation failed: {e}")
        import traceback
        traceback.print_exc()