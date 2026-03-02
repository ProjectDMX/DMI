
## GPT‑2 (`gpt2_p`) Hook List
Source: `transformers/src/transformers/models/gpt2_p/modeling_gpt2.py`

### Top‑level (GPT2Model)
- `transformer.hook_embed`
- `transformer.hook_pos_embed`
- `transformer.hook_final_ln`

### Per‑block (`transformer.blocks.{layer}`)
**Block‑level HookPoints**
- `transformer.blocks.{layer}.hook_resid_pre`
- `transformer.blocks.{layer}.hook_attn_out`
- `transformer.blocks.{layer}.hook_resid_mid`
- `transformer.blocks.{layer}.hook_ln1`
- `transformer.blocks.{layer}.hook_ln2`
- `transformer.blocks.{layer}.hook_mlp_in`
- `transformer.blocks.{layer}.hook_mlp_out`
- `transformer.blocks.{layer}.hook_resid_post`

**Attention HookPoints (`attn`)**
- `transformer.blocks.{layer}.attn.hook_q`
- `transformer.blocks.{layer}.attn.hook_k`
- `transformer.blocks.{layer}.attn.hook_v`
- `transformer.blocks.{layer}.attn.hook_z`
- `transformer.blocks.{layer}.attn.hook_attn_scores`
- `transformer.blocks.{layer}.attn.hook_pattern`
- `transformer.blocks.{layer}.attn.hook_result`

**Cross‑Attention (only if `add_cross_attention=True`)**
- `transformer.blocks.{layer}.crossattention.hook_q/k/v/z/...` (same set, with `crossattention` prefix)

### LMHead extras (HookedGPT2LMHeadModel)
- `token_ids`
- `final_logits`

## Total hook count (GPT‑2, 12 layers)
- Per layer: 7 (attn) + 8 (block) = **15**
- Top‑level: 3
- LMHead extras: 2

**Total = 3 + 12 * 15 + 2 = 185 HookPoints**

## Qwen3 (`qwen3_p`) Hook List
Source: `transformers/src/transformers/models/qwen3_p/modeling_qwen3.py`

### Top-level (HookedQwen3Model / Qwen3Model)
- `hook_embed`
- `hook_final_ln`

### Per-layer (`layers.{layer}`)
**Layer-level HookPoints**
- `layers.{layer}.hook_resid_pre`
- `layers.{layer}.hook_attn_out`
- `layers.{layer}.hook_resid_mid`
- `layers.{layer}.hook_ln1`
- `layers.{layer}.hook_ln2`
- `layers.{layer}.hook_mlp_in`
- `layers.{layer}.hook_mlp_out`
- `layers.{layer}.hook_resid_post`

**Attention HookPoints (`self_attn`)**
- `layers.{layer}.self_attn.hook_q`
- `layers.{layer}.self_attn.hook_k`
- `layers.{layer}.self_attn.hook_v`
- `layers.{layer}.self_attn.hook_attn_scores`
- `layers.{layer}.self_attn.hook_pattern`
- `layers.{layer}.self_attn.hook_z`
- `layers.{layer}.self_attn.hook_result`

### CausalLM extras (HookedQwen3ForCausalLM)
- `token_ids`
- `final_logits`

### Total hook count (Qwen3 default config, 32 layers)
- Per layer: 7 (attn) + 8 (layer-level) = **15**
- Top-level: 2
- CausalLM extras: 2 (only for `HookedQwen3ForCausalLM`)

**HookedQwen3Model (base) = 2 + 32 * 15 = 482**

**HookedQwen3ForCausalLM = 2 + 32 * 15 + 2 = 484**

Note:
- Qwen3 hook names keep the native `layers.{i}...` format only (no alias expansion).
- `HookedQwen3ForCausalLM` normalizes names by dropping the `model.` prefix, so hook names still match `layers.{i}...`.
- If you instantiate non-hooked classes directly, this hook management path is not used.

### HF output alignment (Qwen3)
- HF `hidden_states` (per-layer output): closest to `layers.{i}.hook_resid_post`
- HF final `last_hidden_state`: closest to `hook_final_ln`
- HF `attentions`: closest to `layers.{i}.self_attn.hook_pattern` (best aligned in eager path)

### Qwen3-specific gaps worth adding (not implemented yet)
1. **RoPE/cache debug hooks**
   - Suggested: `hook_q_rot`, `hook_k_rot`, `hook_k_cached`, `hook_v_cached`
   - Why: Qwen3 decode behavior heavily depends on RoPE + KV cache update; current hooks are before/after but not at cache boundary.
2. **MLP internal split hooks**
   - Suggested: `hook_mlp_gate`, `hook_mlp_up`, `hook_mlp_act`, `hook_mlp_mul`
   - Why: Qwen3 MLP uses gated path (`act(gate_proj(x)) * up_proj(x)`); currently only in/out are visible.
3. **Attention-mask/backend observability**
   - Suggested: mask/metadata hook or debug record (`attention_type`, `sliding_window`, backend)
   - Why: Qwen3 mixes full/sliding attention by layer; behavior differs across eager/sdpa/flash and is hard to diagnose with current tensor-only hooks.
