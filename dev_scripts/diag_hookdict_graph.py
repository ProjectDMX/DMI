"""Diagnose why hook_dict iteration breaks manual/graph mode.

Tests whether forward_hooks on HookPoint modules fire during:
1. Regular forward pass
2. CUDA Graph capture
3. CUDA Graph replay
"""
import sys
import struct
import torch
sys.path.insert(0, ".")

from benchmark.tests.profile_decode import (
    create_hf_gpt2_hooked,
    create_config,
)

METADATA_BYTES = 128
_STRUCT = struct.Struct("<Qqqqqqqqqiii44s")

def decode_slot(buf, slot_id):
    offset = slot_id * METADATA_BYTES
    data = buf[offset:offset+METADATA_BYTES].cpu().numpy().tobytes()
    vals = _STRUCT.unpack(data)
    return vals[0]  # data_ptr

def count_valid_slots(buf, num_slots):
    valid = 0
    for i in range(num_slots):
        dp = decode_slot(buf, i)
        if dp != 0:
            valid += 1
    return valid

def main():
    device = torch.device("cuda")
    args = create_config(
        batch_size=2, prefill_tokens=1, decode_steps=1,
        collect_hidden=True, collect_attention=True,
    )

    model, project_logits = create_hf_gpt2_hooked(args, device)
    model.eval()

    # Check hook_dict
    hook_dict = getattr(model, 'hook_dict', None)
    if hook_dict is None:
        print("ERROR: model has no hook_dict")
        return
    print(f"hook_dict entries: {len(hook_dict)}")

    # Count named_modules
    all_mods = [(n, m) for n, m in model.named_modules() if n]
    print(f"named_modules entries: {len(all_mods)}")

    from monitoring.graph_ops import load_graph_monitor_ops
    ops = load_graph_monitor_ops()

    max_slots = 512

    # --- Test 1: hook_dict with forward_hooks ---
    print("\n=== Test 1: hook_dict forward_hooks ===")
    gpu_buf_hd = torch.zeros(max_slots * METADATA_BYTES, dtype=torch.uint8, device=device)
    handles_hd = []
    anchors_hd = []
    slot_count_hd = 0

    for name, module in hook_dict.items():
        sid = slot_count_hd
        if sid >= max_slots:
            break

        def make_hook(slot_id):
            def hook(mod, inp, out):
                tensor = out if torch.is_tensor(out) else None
                if tensor is None and isinstance(out, (list, tuple)):
                    for item in out:
                        if torch.is_tensor(item):
                            tensor = item
                            break
                if tensor is not None and tensor.is_cuda:
                    ops.record(tensor, gpu_buf_hd, slot_id)
                    if torch.cuda.is_current_stream_capturing():
                        anchors_hd.append(tensor)
            return hook

        h = module.register_forward_hook(make_hook(sid))
        handles_hd.append(h)
        slot_count_hd += 1

    print(f"Registered {slot_count_hd} hooks on HookPoint modules")

    # Forward pass (no graph)
    tokens = torch.randint(0, 50257, (2, 1), device=device)
    with torch.no_grad():
        out = model(tokens, use_cache=True, output_hidden_states=True,
                     output_attentions=True, return_dict=True)

    torch.cuda.synchronize()
    valid_after_fwd = count_valid_slots(gpu_buf_hd, slot_count_hd)
    print(f"After forward pass: {valid_after_fwd}/{slot_count_hd} valid slots")

    # CUDA Graph capture
    gpu_buf_hd.zero_()
    past = out.past_key_values
    static_token = torch.empty_like(tokens)
    static_past = tuple(
        (torch.empty_like(k), torch.empty_like(v)) for k, v in past
    )
    for (dk, dv), (sk, sv) in zip(static_past, past):
        dk.copy_(sk)
        dv.copy_(sv)
    static_token.copy_(tokens)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        cap_out = model(static_token, use_cache=True,
                        past_key_values=static_past,
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict=True)
    torch.cuda.synchronize()

    # Anchor
    if anchors_hd:
        ops.sink(anchors_hd)
        anchors_hd.clear()

    valid_after_capture = count_valid_slots(gpu_buf_hd, slot_count_hd)
    print(f"After CUDA Graph capture: {valid_after_capture}/{slot_count_hd} valid slots")
    print(f"  anchors collected: {len(anchors_hd)} (should be 0 after sink)")

    # CUDA Graph replay
    gpu_buf_hd.zero_()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    valid_after_replay = count_valid_slots(gpu_buf_hd, slot_count_hd)
    print(f"After CUDA Graph replay: {valid_after_replay}/{slot_count_hd} valid slots")

    # Cleanup
    for h in handles_hd:
        h.remove()

    # --- Test 2: named_modules with forward_hooks ---
    print("\n=== Test 2: named_modules forward_hooks ===")
    gpu_buf_nm = torch.zeros(max_slots * METADATA_BYTES, dtype=torch.uint8, device=device)
    handles_nm = []
    anchors_nm = []
    slot_count_nm = 0
    module_set = set()

    for name, module in model.named_modules():
        if not name:
            continue
        if module in module_set:
            continue
        module_set.add(module)
        sid = slot_count_nm
        if sid >= max_slots:
            break

        def make_hook2(slot_id):
            def hook(mod, inp, out):
                tensor = out if torch.is_tensor(out) else None
                if tensor is None and isinstance(out, (list, tuple)):
                    for item in out:
                        if torch.is_tensor(item):
                            tensor = item
                            break
                if tensor is not None and tensor.is_cuda:
                    ops.record(tensor, gpu_buf_nm, slot_id)
                    if torch.cuda.is_current_stream_capturing():
                        anchors_nm.append(tensor)
            return hook

        h = module.register_forward_hook(make_hook2(sid))
        handles_nm.append(h)
        slot_count_nm += 1

    print(f"Registered {slot_count_nm} hooks on all named_modules")

    # Need to re-capture graph with new hooks
    gpu_buf_nm.zero_()
    graph2 = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph2):
        cap_out2 = model(static_token, use_cache=True,
                         past_key_values=static_past,
                         output_hidden_states=True,
                         output_attentions=True,
                         return_dict=True)
    torch.cuda.synchronize()

    if anchors_nm:
        ops.sink(anchors_nm)
        anchors_nm.clear()

    valid_after_capture2 = count_valid_slots(gpu_buf_nm, slot_count_nm)
    print(f"After CUDA Graph capture: {valid_after_capture2}/{slot_count_nm} valid slots")

    gpu_buf_nm.zero_()
    torch.cuda.synchronize()
    graph2.replay()
    torch.cuda.synchronize()

    valid_after_replay2 = count_valid_slots(gpu_buf_nm, slot_count_nm)
    print(f"After CUDA Graph replay: {valid_after_replay2}/{slot_count_nm} valid slots")

    for h in handles_nm:
        h.remove()

    print("\n=== Summary ===")
    print(f"hook_dict:      fwd={valid_after_fwd} capture={valid_after_capture} replay={valid_after_replay}")
    print(f"named_modules:  capture={valid_after_capture2} replay={valid_after_replay2}")

if __name__ == "__main__":
    main()
