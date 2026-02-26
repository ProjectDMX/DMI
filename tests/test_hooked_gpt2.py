import torch
import pytest
from transformers import GPT2Config
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError:  # pragma: no cover - optional dependency
    HookedTransformer = None
    HookedTransformerConfig = None


def build_small_model(n_layer: int = 2):
    config = GPT2Config(n_layer=n_layer, n_head=2, n_embd=32, n_positions=32)
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"
    return HookedGPT2Model(config)


def build_matching_hf_tl_pair(n_layer: int = 2):
    if HookedTransformer is None:
        pytest.skip("transformer_lens is required for this test")

    hf_config = GPT2Config(n_layer=n_layer, n_head=2, n_embd=32, n_positions=32)
    hf_config.vocab_size = 50257
    hf_config.attn_implementation = "eager"
    hf_config._attn_implementation = "eager"
    hf_model = HookedGPT2Model(hf_config)

    tl_cfg = HookedTransformerConfig(
        n_layers=n_layer,
        n_ctx=hf_config.n_positions,
        d_model=hf_config.hidden_size,
        n_heads=hf_config.num_attention_heads,
        d_head=hf_config.hidden_size // hf_config.num_attention_heads,
        d_mlp=hf_config.n_inner or 4 * hf_config.hidden_size,
        d_vocab=hf_config.vocab_size,
        act_fn="gelu",
        normalization_type="LN",
        positional_embedding_type="standard",
        device="cpu",
        use_attn_result=True,
        default_prepend_bos=False,
    )
    tl_model = HookedTransformer(tl_cfg)
    return hf_model, tl_model


def expected_shared_hook_names(num_layers: int) -> set[str]:
    names = {"hook_embed", "hook_pos_embed"}
    for layer in range(num_layers):
        prefix = f"blocks.{layer}"
        names.update(
            {
                f"{prefix}.attn.hook_attn_scores",
                f"{prefix}.attn.hook_k",
                f"{prefix}.attn.hook_pattern",
                f"{prefix}.attn.hook_q",
                f"{prefix}.attn.hook_v",
                f"{prefix}.attn.hook_z",
                f"{prefix}.hook_attn_out",
                f"{prefix}.hook_mlp_out",
                f"{prefix}.hook_resid_mid",
                f"{prefix}.hook_resid_post",
                f"{prefix}.hook_resid_pre",
            }
        )
    return names


def attention_result_hook_names(num_layers: int) -> set[str]:
    return {f"blocks.{layer}.attn.hook_result" for layer in range(num_layers)}


def test_hook_aliases_exposed():
    model = build_small_model(n_layer=1)
    assert "blocks.0.hook_resid_pre" in model.hook_dict


def test_run_with_cache_collects_expected_keys():
    model = build_small_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        outputs, cache = model.run_with_cache(input_ids)
    assert outputs.last_hidden_state.shape == (1, 4, model.config.hidden_size)
    assert "hook_embed" in cache
    assert any(key.startswith("blocks.0.attn.hook_q") for key in cache)


def test_manual_hook_capture():
    model = build_small_model()
    captured = {}

    def save_hook(tensor, hook):
        captured[hook.name] = tensor.detach()

    model.add_hook("blocks.0.hook_resid_pre", save_hook)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 3))
    with torch.no_grad():
        model(input_ids)
    model.reset_hooks()

    assert "blocks.0.hook_resid_pre" in captured
    assert captured["blocks.0.hook_resid_pre"].shape == (1, 3, model.config.hidden_size)


@pytest.mark.skipif(HookedTransformer is None, reason="transformer_lens not available in test environment")
def test_shared_hooks_align_with_transformer_lens():
    hf_model, tl_model = build_matching_hf_tl_pair()

    torch.manual_seed(0)
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 5), dtype=torch.long)

    _, hf_cache = hf_model.run_with_cache(input_ids)
    _, tl_cache = tl_model.run_with_cache(input_ids)

    shared_names = set(hf_cache.keys()) & set(tl_cache.keys())
    result_names = attention_result_hook_names(hf_model.config.n_layer)
    expected_names = expected_shared_hook_names(hf_model.config.n_layer)

    assert result_names.issubset(shared_names)
    assert (shared_names - result_names) == expected_names


@pytest.mark.skipif(HookedTransformer is None, reason="transformer_lens not available in test environment")
def test_shared_hook_tensor_shapes_match_transformer_lens():
    hf_model, tl_model = build_matching_hf_tl_pair()

    torch.manual_seed(1)
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 6), dtype=torch.long)

    _, hf_cache = hf_model.run_with_cache(input_ids)
    _, tl_cache = tl_model.run_with_cache(input_ids)

    shared_names = expected_shared_hook_names(hf_model.config.n_layer)
    result_names = attention_result_hook_names(hf_model.config.n_layer)

    assert shared_names.issubset(hf_cache.keys())
    assert shared_names.issubset(tl_cache.keys())
    assert result_names.issubset(hf_cache.keys())
    assert result_names.issubset(tl_cache.keys())

    from collections import Counter

    for name in shared_names:
        hf_shape = tuple(hf_cache[name].shape)
        tl_shape = tuple(tl_cache[name].shape)
        assert Counter(hf_shape) == Counter(tl_shape), f"shape mismatch for {name}: {hf_shape} vs {tl_shape}"

    # TransformerLens keeps per-head structure for hook_result; verify consistency without requiring identical shape
    for name in result_names:
        hf_shape = hf_cache[name].shape
        tl_shape = tl_cache[name].shape
        assert hf_shape[0] == tl_shape[0]  # batch
        assert hf_shape[1] == tl_shape[1]  # sequence length
        assert hf_shape[-1] == tl_shape[-1]  # model width
        assert tl_shape[2] == hf_model.config.num_attention_heads
