"""Tests for the DMI hook compiler (monitoring.compiler)."""
from __future__ import annotations

import ast
import io
import os
import tempfile
import textwrap
from contextlib import redirect_stdout

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_GPT2_SRC = os.path.join(
    ROOT, "integration/vllm/vllm/model_executor/models/gpt2.py")
VLLM_GPT2_GT = os.path.join(
    ROOT, "integration/vllm/vllm/model_executor/models/gpt2_p.py")
HF_GPT2_SRC = os.path.join(
    ROOT, "integration/transformers/src/transformers/models/gpt2/modeling_gpt2.py")
HF_GPT2_GT = os.path.join(
    ROOT, "integration/transformers/src/transformers/models/gpt2_p/modeling_gpt2.py")
HF_QWEN3_SRC = os.path.join(
    ROOT, "integration/transformers/src/transformers/models/qwen3/modeling_qwen3.py")
HF_QWEN3_GT = os.path.join(
    ROOT, "integration/transformers/src/transformers/models/qwen3_p/modeling_qwen3.py")

# ---------------------------------------------------------------------------
# Fixtures — GPT2 vLLM spec (matches real source structure)
# ---------------------------------------------------------------------------
GPT2_VLLM_SPEC = textwrap.dedent("""\
    from monitoring.compiler.dsl import H, spec

    @spec(source="{source}", framework="vllm")
    class GPT2Attention:
        def forward(self, hidden_states):
            (qkv, _) = self.c_attn(hidden_states)
            q, k, v = qkv.chunk(chunks=3, dim=-1)
            H("q", q)
            H("k", k)
            H("v", v)
            attn_output = self.attn(q, k, v)
            H("z", attn_output)
            (attn_output, _) = self.c_proj(attn_output)

    class GPT2MLP:
        def forward(self, hidden_states):
            (hidden_states, _) = self.c_fc(hidden_states)
            hidden_states = self.act(hidden_states)
            H("post", hidden_states)
            (hidden_states, _) = self.c_proj(hidden_states)

    class GPT2Block:
        def forward(self, hidden_states):
            H("resid_pre", hidden_states)
            hidden_states = self.ln_1(hidden_states)
            H("ln1", hidden_states)
            attn_output = self.attn(hidden_states=hidden_states)
            H("attn_out", attn_output)
            hidden_states = attn_output + residual
            H("resid_mid", hidden_states)
            hidden_states = self.ln_2(hidden_states)
            H("ln2", hidden_states)
            H("mlp_in", hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            H("mlp_out", feed_forward_hidden_states)

    class GPT2Model:
        def forward(self, input_ids, position_ids, intermediate_tensors, inputs_embeds):
            if get_pp_group().is_first_rank:
                inputs_embeds = self.embed_input_ids(input_ids)
                H("embed", inputs_embeds)
                position_embeds = self.wpe(position_ids)
                H("pos_embed", position_embeds)
                hidden_states = inputs_embeds + position_embeds

            for layer in islice(self.h, self.start_layer, self.end_layer):
                hidden_states = layer(hidden_states)

            H("resid_final", hidden_states)
            hidden_states = self.ln_f(hidden_states)
            H("final_ln", hidden_states)
""")


# ---------------------------------------------------------------------------
# HF GPT-2 spec fixture
# ---------------------------------------------------------------------------
GPT2_HF_SPEC = textwrap.dedent("""\
    from monitoring.compiler.dsl import H, spec

    @spec(source="{source}", framework="hf")
    class GPT2Attention:
        def forward(self, hidden_states, past_key_values, cache_position, attention_mask):
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            key_states = key_states.view(shape_kv)
            H("k", key_states)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.view(shape_kv)
            H("v", value_states)
            value_states = value_states.transpose(1, 2)
            query_states = query_states.view(shape_q)
            H("q", query_states)
            query_states = query_states.transpose(1, 2)
            attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask)
            H("z", attn_output)
            attn_output = attn_output.reshape(attn_output.shape)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

    @spec(source="{source}", framework="hf")
    class GPT2MLP:
        def forward(self, hidden_states):
            hidden_states = self.c_fc(hidden_states)
            hidden_states = self.act(hidden_states)
            H("post", hidden_states)
            hidden_states = self.c_proj(hidden_states)
            hidden_states = self.dropout(hidden_states)

    @spec(source="{source}", framework="hf")
    class GPT2Block:
        def forward(self, hidden_states, past_key_values, cache_position, attention_mask):
            H("resid_pre", hidden_states)
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            H("ln1", hidden_states)
            attn_output, self_attn_weights = self.attn(hidden_states, past_key_values, cache_position, attention_mask)
            H("attn_out", attn_output)
            hidden_states = attn_output + residual
            H("resid_mid", hidden_states)
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            H("ln2", hidden_states)
            H("mlp_in", hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            H("mlp_out", feed_forward_hidden_states)
            hidden_states = residual + feed_forward_hidden_states

    @spec(source="{source}", framework="hf")
    class GPT2Model:
        def forward(self, input_ids, position_ids):
            inputs_embeds = self.wte(input_ids)
            H("embed", inputs_embeds)
            position_embeds = self.wpe(position_ids)
            H("pos_embed", position_embeds)
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)
            for i, block in enumerate(self.h):
                outputs = block(hidden_states)
                hidden_states = outputs[0]
            H("resid_final", hidden_states)
            hidden_states = self.ln_f(hidden_states)
            H("final_ln", hidden_states)
""")


# ---------------------------------------------------------------------------
# HF Qwen3 spec fixture
# ---------------------------------------------------------------------------
QWEN3_HF_SPEC = textwrap.dedent("""\
    from monitoring.compiler.dsl import H, spec

    @spec(source="{source}", framework="hf")
    class Qwen3Attention:
        def forward(self, hidden_states, position_embeddings, attention_mask):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
            H("q", query_states)
            query_states = query_states.transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
            H("k", key_states)
            key_states = key_states.transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape)
            H("v", value_states)
            value_states = value_states.transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask)
            H("z", attn_output)
            attn_output = attn_output.reshape(input_shape)
            attn_output = self.o_proj(attn_output)

    @spec(source="{source}", framework="hf")
    class Qwen3MLP:
        def forward(self, x):
            x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            H("post", x)
            x = self.down_proj(x)

    @spec(source="{source}", framework="hf")
    class Qwen3DecoderLayer:
        def forward(self, hidden_states, attention_mask, position_embeddings):
            H("resid_pre", hidden_states)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            H("ln1", hidden_states)
            hidden_states, self_attn_weights = self.self_attn(hidden_states, position_embeddings, attention_mask)
            H("attn_out", hidden_states)
            hidden_states = residual + hidden_states
            H("resid_mid", hidden_states)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            H("ln2", hidden_states)
            H("mlp_in", hidden_states)
            hidden_states = self.mlp(hidden_states)
            H("mlp_out", hidden_states)
            hidden_states = residual + hidden_states

    @spec(source="{source}", framework="hf")
    class Qwen3Model:
        def forward(self, input_ids):
            inputs_embeds = self.embed_tokens(input_ids)
            H("embed", inputs_embeds)
            hidden_states = inputs_embeds
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(hidden_states)
            H("resid_final", hidden_states)
            hidden_states = self.norm(hidden_states)
            H("final_ln", hidden_states)
""")


def _write_spec(source_path: str, spec_text: str = None) -> str:
    """Write spec to a temp file, return its path."""
    if spec_text is None:
        spec_text = GPT2_VLLM_SPEC.format(source=source_path)
    fd, path = tempfile.mkstemp(suffix=".py")
    os.write(fd, spec_text.encode())
    os.close(fd)
    return path


def _get_hook_attrs(tree: ast.Module, class_name: str) -> list[str]:
    """Extract self.hook_xxx = HookPoint() attr names from __init__."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    attrs = []
                    for stmt in ast.walk(item):
                        if (isinstance(stmt, ast.Assign)
                                and len(stmt.targets) == 1
                                and isinstance(stmt.targets[0], ast.Attribute)
                                and getattr(stmt.targets[0].value, 'id', '') == 'self'
                                and stmt.targets[0].attr.startswith('hook_')):
                            attrs.append(stmt.targets[0].attr)
                    return attrs
    return []


def _walk_body_for_hooks(body: list[ast.stmt]) -> list[tuple[str, str]]:
    """Recursively walk body in execution order, extract self.hook_xxx() calls."""
    calls = []
    for stmt in body:
        # Side-effect style: self.hook_xxx(arg)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == 'self'
                    and call.func.attr.startswith('hook_')):
                arg = ast.unparse(call.args[0]) if call.args else ""
                calls.append((call.func.attr, arg))
        # Assignment style (HF): x = self.hook_xxx(arg)
        if isinstance(stmt, ast.Assign):
            val = stmt.value
            if (isinstance(val, ast.Call)
                    and isinstance(val.func, ast.Attribute)
                    and isinstance(val.func.value, ast.Name)
                    and val.func.value.id == 'self'
                    and val.func.attr.startswith('hook_')):
                arg = ast.unparse(val.args[0]) if val.args else ""
                calls.append((val.func.attr, arg))
        if isinstance(stmt, ast.For):
            calls.extend(_walk_body_for_hooks(stmt.body))
        elif isinstance(stmt, ast.If):
            calls.extend(_walk_body_for_hooks(stmt.body))
            calls.extend(_walk_body_for_hooks(stmt.orelse))
    return calls


def _get_hook_calls(tree: ast.Module, class_name: str) -> list[tuple[str, str]]:
    """Extract ordered [(hook_attr, arg_name), ...] from forward()."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return _walk_body_for_hooks(item.body)
    return []


# ===================================================================
# parse_spec tests
# ===================================================================

class TestParseSpec:
    def test_extracts_hooks(self):
        from monitoring.compiler.compiler import parse_spec
        spec_path = _write_spec(VLLM_GPT2_SRC)
        try:
            spec = parse_spec(spec_path)
            classes = {sc.name: sc for sc in spec.classes}
            assert set(classes.keys()) == {
                "GPT2Attention", "GPT2MLP", "GPT2Block", "GPT2Model"}

            attn = classes["GPT2Attention"]
            assert [h.name for h in attn.hooks] == ["q", "k", "v", "z"]

            block = classes["GPT2Block"]
            assert [h.name for h in block.hooks] == [
                "resid_pre", "ln1", "attn_out", "resid_mid",
                "ln2", "mlp_in", "mlp_out"]

            model = classes["GPT2Model"]
            assert [h.name for h in model.hooks] == [
                "embed", "pos_embed", "resid_final", "final_ln"]
        finally:
            os.unlink(spec_path)

    def test_source_and_framework(self):
        from monitoring.compiler.compiler import parse_spec
        spec_path = _write_spec(VLLM_GPT2_SRC)
        try:
            spec = parse_spec(spec_path)
            assert spec.source == VLLM_GPT2_SRC
            assert spec.framework == "vllm"
        finally:
            os.unlink(spec_path)

    def test_anchors(self):
        from monitoring.compiler.compiler import parse_spec
        spec_path = _write_spec(VLLM_GPT2_SRC)
        try:
            spec = parse_spec(spec_path)
            block = {sc.name: sc for sc in spec.classes}["GPT2Block"]
            # resid_pre has no anchor (first in forward)
            assert block.hooks[0].name == "resid_pre"
            assert block.hooks[0].anchor_before is None
            # ln1 anchored to self.ln_1(hidden_states)
            assert block.hooks[1].name == "ln1"
            assert "ln_1" in block.hooks[1].anchor_before
        finally:
            os.unlink(spec_path)

    def test_anchor_kinds(self):
        from monitoring.compiler.compiler import parse_spec
        spec_path = _write_spec(VLLM_GPT2_SRC)
        try:
            spec = parse_spec(spec_path)
            model = {sc.name: sc for sc in spec.classes}["GPT2Model"]
            hooks = {h.name: h for h in model.hooks}
            # embed is inside if body, after a stmt
            assert hooks["embed"].anchor_kind == "stmt"
            # resid_final is after the for loop
            assert hooks["resid_final"].anchor_kind == "after_block"
        finally:
            os.unlink(spec_path)


# ===================================================================
# Composition analysis tests
# ===================================================================

class TestComposition:
    def _get_tree(self):
        with open(VLLM_GPT2_SRC) as f:
            return ast.parse(f.read())

    def test_find_composition(self):
        from monitoring.compiler.compiler import _find_composition
        tree = self._get_tree()
        spec_names = {"GPT2Attention", "GPT2MLP", "GPT2Block", "GPT2Model"}
        comp = _find_composition(tree, spec_names)

        assert "attn" in comp["GPT2Block"]
        assert comp["GPT2Block"]["attn"] == ("GPT2Attention", False)
        assert "mlp" in comp["GPT2Block"]
        assert comp["GPT2Block"]["mlp"] == ("GPT2MLP", False)

        # GPT2Model uses make_layers → GPT2Block is a layer
        assert "h" in comp["GPT2Model"]
        assert comp["GPT2Model"]["h"] == ("GPT2Block", True)

    def test_find_layer_iter(self):
        from monitoring.compiler.compiler import _find_layer_iter
        tree = self._get_tree()
        info = _find_layer_iter(tree, "GPT2Model")
        assert info is not None
        layer_attr, start_attr, end_attr = info
        assert layer_attr == "h"
        assert start_attr == "start_layer"
        assert end_attr == "end_layer"

    def test_resolve_path(self):
        from monitoring.compiler.compiler import _find_composition, _resolve_path
        tree = self._get_tree()
        spec_names = {"GPT2Attention", "GPT2MLP", "GPT2Block", "GPT2Model"}
        comp = _find_composition(tree, spec_names)

        assert _resolve_path(comp, "GPT2Block", "GPT2Block") == "layer"
        assert _resolve_path(comp, "GPT2Block", "GPT2Attention") == "layer.attn"
        assert _resolve_path(comp, "GPT2Block", "GPT2MLP") == "layer.mlp"
        assert _resolve_path(comp, "GPT2Block", "GPT2Model") is None


# ===================================================================
# Interleave tests
# ===================================================================

class TestInterleave:
    def test_interleave_gpt2_block(self):
        from monitoring.compiler.compiler import (
            parse_spec, _find_composition, _interleave_hooks)
        spec_path = _write_spec(VLLM_GPT2_SRC)
        try:
            spec = parse_spec(spec_path)
            spec_map = {sc.name: sc for sc in spec.classes}
            with open(VLLM_GPT2_SRC) as f:
                tree = ast.parse(f.read())
            comp = _find_composition(tree, set(spec_map.keys()))
            hooks = _interleave_hooks(spec_map, comp, "GPT2Block")

            names = [h.name for h, _ in hooks]
            owners = [owner for _, owner in hooks]

            # Expected FIFO order
            assert names == [
                "resid_pre", "ln1",
                "q", "k", "v", "z",         # from GPT2Attention
                "attn_out", "resid_mid",
                "ln2", "mlp_in",
                "post",                       # from GPT2MLP
                "mlp_out",
            ]
            assert owners[2:6] == ["GPT2Attention"] * 4
            assert owners[10] == "GPT2MLP"
        finally:
            os.unlink(spec_path)


# ===================================================================
# compile_spec integration tests
# ===================================================================

@pytest.fixture
def compiled_gpt2():
    """Compile GPT-2 vLLM spec and return (output_text, AST)."""
    from monitoring.compiler.compiler import compile_spec
    spec_path = _write_spec(VLLM_GPT2_SRC)
    try:
        result = compile_spec(spec_path)
    finally:
        os.unlink(spec_path)
    tree = ast.parse(result)
    return result, tree


class TestCompileHookDeclarations:
    """Check that __init__ gets the right HookPoint declarations."""

    def test_attention_hooks(self, compiled_gpt2):
        _, tree = compiled_gpt2
        attrs = _get_hook_attrs(tree, "GPT2Attention")
        assert set(attrs) == {"hook_q", "hook_k", "hook_v", "hook_z"}

    def test_mlp_hooks(self, compiled_gpt2):
        _, tree = compiled_gpt2
        attrs = _get_hook_attrs(tree, "GPT2MLP")
        assert attrs == ["hook_post"]

    def test_block_hooks(self, compiled_gpt2):
        _, tree = compiled_gpt2
        attrs = _get_hook_attrs(tree, "GPT2Block")
        assert set(attrs) == {
            "hook_resid_pre", "hook_ln1", "hook_attn_out",
            "hook_resid_mid", "hook_ln2", "hook_mlp_in", "hook_mlp_out"}

    def test_model_hooks(self, compiled_gpt2):
        _, tree = compiled_gpt2
        attrs = _get_hook_attrs(tree, "GPT2Model")
        assert set(attrs) == {
            "hook_embed", "hook_pos_embed",
            "hook_resid_final", "hook_final_ln"}


class TestCompileHookCalls:
    """Check that forward() gets hook calls with correct arguments and order."""

    def test_attention_calls(self, compiled_gpt2):
        _, tree = compiled_gpt2
        calls = _get_hook_calls(tree, "GPT2Attention")
        assert calls == [
            ("hook_q", "q"),
            ("hook_k", "k"),
            ("hook_v", "v"),
            ("hook_z", "attn_output"),
        ]

    def test_mlp_calls(self, compiled_gpt2):
        _, tree = compiled_gpt2
        calls = _get_hook_calls(tree, "GPT2MLP")
        assert calls == [("hook_post", "hidden_states")]

    def test_block_calls(self, compiled_gpt2):
        _, tree = compiled_gpt2
        calls = _get_hook_calls(tree, "GPT2Block")
        expected_names = [
            "hook_resid_pre", "hook_ln1", "hook_attn_out",
            "hook_resid_mid", "hook_ln2", "hook_mlp_in", "hook_mlp_out"]
        assert [c[0] for c in calls] == expected_names

    def test_model_calls(self, compiled_gpt2):
        _, tree = compiled_gpt2
        calls = _get_hook_calls(tree, "GPT2Model")
        names = [c[0] for c in calls]
        assert "hook_embed" in names
        assert "hook_pos_embed" in names
        assert "hook_resid_final" in names
        assert "hook_final_ln" in names
        # resid_final must come before final_ln
        assert names.index("hook_resid_final") < names.index("hook_final_ln")


class TestCompileImports:
    """Check that the monitoring imports are added correctly."""

    def test_hookpoint_import(self, compiled_gpt2):
        text, _ = compiled_gpt2
        assert "from monitoring.hook_points import HookPoint" in text

    def test_ring_transport_import(self, compiled_gpt2):
        text, _ = compiled_gpt2
        assert "from monitoring.ring_transport import" in text
        assert "HookSpec," in text
        assert "HOOK_TYPE_Q," in text

    def test_existing_imports_intact(self, compiled_gpt2):
        text, _ = compiled_gpt2
        # The multi-line from .utils import (...) must not be broken
        assert "from .utils import (" in text
        assert "    AutoWeightsLoader," in text


class TestCompileHookSpecs:
    """Check the generated get_hook_specs() method."""

    def _get_specs_body(self, tree: ast.Module) -> ast.FunctionDef:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GPT2Model":
                for item in node.body:
                    if (isinstance(item, ast.FunctionDef)
                            and item.name == "get_hook_specs"):
                        return item
        pytest.fail("get_hook_specs not found on GPT2Model")

    def test_exists(self, compiled_gpt2):
        _, tree = compiled_gpt2
        self._get_specs_body(tree)

    def test_hook_order(self, compiled_gpt2):
        """Verify FIFO order of HookSpec entries matches ground truth."""
        text, tree = compiled_gpt2
        func = self._get_specs_body(tree)
        src = ast.unparse(func)

        import re
        expected_per_layer = [
            "RESID_PRE", "LN1",
            "Q", "K", "V", "Z",
            "ATTN_OUT", "RESID_MID",
            "LN2", "MLP_IN", "MLP_POST", "MLP_OUT",
        ]

        for stmt in ast.walk(func):
            if isinstance(stmt, ast.For):
                loop_src = ast.unparse(stmt)
                loop_types = re.findall(r'HOOK_TYPE_(\w+)', loop_src)
                assert loop_types == expected_per_layer
                break
        else:
            pytest.fail("No for loop in get_hook_specs")

    def test_mlp_path(self, compiled_gpt2):
        text, _ = compiled_gpt2
        assert "layer.mlp.hook_post" in text

    def test_attn_path(self, compiled_gpt2):
        text, _ = compiled_gpt2
        assert "layer.attn.hook_q" in text

    def test_layer_iteration(self, compiled_gpt2):
        text, _ = compiled_gpt2
        assert "range(self.start_layer, self.end_layer)" in text


# ===================================================================
# Extractor tests
# ===================================================================

class TestExtractor:
    def test_extract_gpt2(self):
        from monitoring.compiler.extractor import extract_skeleton
        skeleton = extract_skeleton(VLLM_GPT2_SRC)
        assert "class GPT2Attention" in skeleton
        assert "class GPT2MLP" in skeleton
        assert "class GPT2Block" in skeleton
        assert "class GPT2Model" in skeleton
        assert "self.c_attn" in skeleton
        assert "self.ln_1" in skeleton
        assert "self.attn" in skeleton

    def test_skeleton_parseable(self):
        from monitoring.compiler.extractor import extract_skeleton
        skeleton = extract_skeleton(VLLM_GPT2_SRC)
        ast.parse(skeleton)


# ===================================================================
# Ground truth comparison
# ===================================================================

class TestGroundTruth:
    """Compare compiled output against hand-written ground truth."""

    def test_gpt2_hook_declarations_match(self, compiled_gpt2):
        _, compiled_tree = compiled_gpt2
        with open(VLLM_GPT2_GT) as f:
            gt_tree = ast.parse(f.read())

        for cls_name in ["GPT2Attention", "GPT2MLP", "GPT2Block", "GPT2Model"]:
            compiled_attrs = set(_get_hook_attrs(compiled_tree, cls_name))
            gt_attrs = set(_get_hook_attrs(gt_tree, cls_name))
            assert compiled_attrs == gt_attrs, (
                f"{cls_name}: compiled {compiled_attrs} != ground truth {gt_attrs}")

    def test_gpt2_hook_calls_match(self, compiled_gpt2):
        _, compiled_tree = compiled_gpt2
        with open(VLLM_GPT2_GT) as f:
            gt_tree = ast.parse(f.read())

        for cls_name in ["GPT2MLP", "GPT2Block"]:
            compiled_calls = [c[0] for c in _get_hook_calls(compiled_tree, cls_name)]
            gt_calls = [c[0] for c in _get_hook_calls(gt_tree, cls_name)]
            assert compiled_calls == gt_calls, (
                f"{cls_name}: compiled {compiled_calls} != ground truth {gt_calls}")

    def test_gpt2_hook_specs_order(self, compiled_gpt2):
        import re
        text, _ = compiled_gpt2
        with open(VLLM_GPT2_GT) as f:
            gt_text = f.read()

        def extract_loop_types(src):
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef)
                        and node.name == "get_hook_specs"):
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.For):
                            return re.findall(
                                r'HOOK_TYPE_(\w+)', ast.unparse(stmt))
            return []

        compiled_order = extract_loop_types(text)
        gt_order = extract_loop_types(gt_text)
        assert compiled_order == gt_order


# ===================================================================
# Target validation tests
# ===================================================================

class TestTargetValidation:
    def test_error_on_tuple_unpack_without_target(self):
        from monitoring.compiler.compiler import parse_spec
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="dummy.py", framework="vllm")
            class Foo:
                def forward(self, x):
                    a, b = self.split(x)
                    H("hook1")
        """)
        spec_path = _write_spec("dummy.py", spec_text)
        try:
            with pytest.raises(ValueError, match="tuple unpack"):
                parse_spec(spec_path)
        finally:
            os.unlink(spec_path)

    def test_error_on_no_target(self):
        from monitoring.compiler.compiler import compile_spec
        # H("hook1") as first statement — no target can be inferred, no explicit target
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="{source}", framework="vllm")
            class GPT2Block:
                def forward(self, hidden_states):
                    H("hook1")
                    hidden_states = self.ln_1(hidden_states)
        """.format(source=VLLM_GPT2_SRC))
        spec_path = _write_spec(VLLM_GPT2_SRC, spec_text)
        try:
            with pytest.raises(ValueError, match="no target"):
                compile_spec(spec_path)
        finally:
            os.unlink(spec_path)

    def test_error_on_nonexistent_target(self):
        from monitoring.compiler.compiler import compile_spec
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="{source}", framework="vllm")
            class GPT2Attention:
                def forward(self, hidden_states):
                    (qkv, _) = self.c_attn(hidden_states)
                    H("q", q_head)
        """.format(source=VLLM_GPT2_SRC))
        spec_path = _write_spec(VLLM_GPT2_SRC, spec_text)
        try:
            with pytest.raises(ValueError, match="not bound"):
                compile_spec(spec_path)
        finally:
            os.unlink(spec_path)


# ===================================================================
# Control flow tests
# ===================================================================

class TestControlFlow:
    def test_hook_after_for_loop(self):
        """H() after a for loop should be inserted after the loop, not inside it."""
        from monitoring.compiler.compiler import compile_spec
        spec_path = _write_spec(VLLM_GPT2_SRC)
        try:
            result = compile_spec(spec_path)
        finally:
            os.unlink(spec_path)

        # Use AST to find the for loop and hook calls in GPT2Model.forward
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GPT2Model":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "forward":
                        # Find for loop end_lineno
                        for_end = None
                        for stmt in item.body:
                            if isinstance(stmt, ast.For):
                                for_end = stmt.end_lineno
                        assert for_end is not None, "for loop not found"

                        # Find hook_resid_final and ln_f lines
                        resid_final_line = None
                        ln_f_line = None
                        for stmt in ast.walk(item):
                            if (isinstance(stmt, ast.Call)
                                    and isinstance(stmt.func, ast.Attribute)
                                    and stmt.func.attr == "hook_resid_final"):
                                resid_final_line = stmt.lineno
                            if (isinstance(stmt, ast.Call)
                                    and isinstance(stmt.func, ast.Attribute)
                                    and stmt.func.attr == "ln_f"):
                                ln_f_line = stmt.lineno

                        assert resid_final_line is not None, "hook_resid_final not found"
                        assert ln_f_line is not None, "self.ln_f not found"
                        assert resid_final_line > for_end, (
                            f"hook_resid_final ({resid_final_line}) should be after "
                            f"for loop end ({for_end})")
                        assert resid_final_line < ln_f_line, (
                            f"hook_resid_final ({resid_final_line}) should be before "
                            f"self.ln_f ({ln_f_line})")
                        return
        pytest.fail("GPT2Model.forward not found")

    def test_hook_block_entry(self):
        """H() as first statement in if body should stay inside the if."""
        from monitoring.compiler.compiler import parse_spec
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="{source}", framework="vllm")
            class GPT2Model:
                def forward(self, input_ids, position_ids, intermediate_tensors, inputs_embeds):
                    if get_pp_group().is_first_rank:
                        H("block_start", inputs_embeds)
                        inputs_embeds = self.embed_input_ids(input_ids)
        """.format(source=VLLM_GPT2_SRC))
        spec_path = _write_spec(VLLM_GPT2_SRC, spec_text)
        try:
            spec = parse_spec(spec_path)
            model = {sc.name: sc for sc in spec.classes}["GPT2Model"]
            h = model.hooks[0]
            assert h.anchor_kind == "block_entry"
            assert h.anchor_branch == "body"
        finally:
            os.unlink(spec_path)

    def test_hook_else_block_entry(self):
        """H() as first statement in else branch should be in orelse."""
        from monitoring.compiler.compiler import parse_spec
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="dummy.py", framework="vllm")
            class Foo:
                def forward(self, x):
                    if cond:
                        a = foo(x)
                    else:
                        H("else_hook", x)
                        b = bar(x)
        """)
        spec_path = _write_spec("dummy.py", spec_text)
        try:
            spec = parse_spec(spec_path)
            foo = {sc.name: sc for sc in spec.classes}["Foo"]
            h = foo.hooks[0]
            assert h.name == "else_hook"
            assert h.anchor_kind == "block_entry"
            assert h.anchor_branch == "orelse"
        finally:
            os.unlink(spec_path)

    def test_while_not_supported(self):
        from monitoring.compiler.compiler import parse_spec
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="dummy.py", framework="vllm")
            class Foo:
                def forward(self, x):
                    while True:
                        x = self.step(x)
                    H("done", x)
        """)
        spec_path = _write_spec("dummy.py", spec_text)
        try:
            with pytest.raises(NotImplementedError, match="while"):
                parse_spec(spec_path)
        finally:
            os.unlink(spec_path)

    def test_duplicate_block_headers(self):
        """Two blocks with same header should get different ordinals."""
        from monitoring.compiler.compiler import parse_spec
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="dummy.py", framework="vllm")
            class Foo:
                def forward(self, x):
                    if cond:
                        a = foo(x)
                    H("after_first", a)
                    if cond:
                        b = bar(x)
                    H("after_second", b)
        """)
        spec_path = _write_spec("dummy.py", spec_text)
        try:
            spec = parse_spec(spec_path)
            foo = {sc.name: sc for sc in spec.classes}["Foo"]
            assert len(foo.hooks) == 2
            h1, h2 = foo.hooks
            assert h1.anchor_kind == "after_block"
            assert h2.anchor_kind == "after_block"
            assert h1.block_ordinal == 0
            assert h2.block_ordinal == 1
        finally:
            os.unlink(spec_path)


class TestHookSpecsOrder:
    def test_after_block_before_loop_in_specs(self):
        """after_block hook before the layer loop should be in pre, not post."""
        from monitoring.compiler.compiler import compile_spec
        import re
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="{source}", framework="vllm")
            class GPT2Block:
                def forward(self, hidden_states):
                    H("resid_pre", hidden_states)
                    residual = hidden_states
                    hidden_states = self.ln_1(hidden_states)
                    attn_output = self.attn(hidden_states=hidden_states)
                    hidden_states = attn_output + residual
                    residual = hidden_states
                    hidden_states = self.ln_2(hidden_states)
                    feed_forward_hidden_states = self.mlp(hidden_states)
                    hidden_states = residual + feed_forward_hidden_states

            @spec(source="{source}", framework="vllm")
            class GPT2Model:
                def forward(self, input_ids, position_ids, intermediate_tensors, inputs_embeds):
                    if get_pp_group().is_first_rank:
                        inputs_embeds = self.embed_input_ids(input_ids)
                        H("embed", inputs_embeds)
                        position_embeds = self.wpe(position_ids)
                        H("pos_embed", position_embeds)
                        hidden_states = inputs_embeds + position_embeds

                    H("resid_pre_loop", hidden_states)

                    for layer in islice(self.h, self.start_layer, self.end_layer):
                        hidden_states = layer(hidden_states)

                    H("resid_final", hidden_states)
                    hidden_states = self.ln_f(hidden_states)
                    H("final_ln", hidden_states)
        """.format(source=VLLM_GPT2_SRC))
        spec_path = _write_spec(VLLM_GPT2_SRC, spec_text)
        try:
            result = compile_spec(spec_path)
        finally:
            os.unlink(spec_path)

        tree = ast.parse(result)
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "get_hook_specs"):
                src = ast.unparse(node)
                types = re.findall(r'HOOK_TYPE_(\w+)', src)
                # Find the for loop in get_hook_specs
                has_loop = False
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.For):
                        has_loop = True
                        loop_src = ast.unparse(stmt)
                        loop_types = re.findall(r'HOOK_TYPE_(\w+)', loop_src)
                        break
                assert has_loop, "No for loop in get_hook_specs"

                # Check overall order: pre_loop types come before loop types
                # Find index of first loop type in overall types
                first_loop_idx = None
                for lt in loop_types:
                    if lt in types:
                        idx = types.index(lt)
                        if first_loop_idx is None or idx < first_loop_idx:
                            first_loop_idx = idx

                # RESID_FINAL should be after the loop section
                assert "RESID_FINAL" in types
                resid_final_idx = types.index("RESID_FINAL")
                assert resid_final_idx > first_loop_idx, (
                    "RESID_FINAL should be after per-layer hooks")

                # EMBED and POS_EMBED should be before the loop section
                assert "EMBED" in types
                assert types.index("EMBED") < first_loop_idx, (
                    "EMBED should be before per-layer hooks")
                return
        pytest.fail("get_hook_specs not found")


# ===================================================================
# dtype tests
# ===================================================================

class TestDtype:
    def test_dtype_in_hook_spec(self):
        """dtype= in H() should propagate to HookSpec generation."""
        from monitoring.compiler.compiler import compile_spec
        spec_text = textwrap.dedent("""\
            from monitoring.compiler.dsl import H, spec

            @spec(source="{source}", framework="vllm")
            class GPT2Model:
                def forward(self, input_ids, position_ids, intermediate_tensors, inputs_embeds):
                    if get_pp_group().is_first_rank:
                        inputs_embeds = self.embed_input_ids(input_ids)
                        H("embed", inputs_embeds, dtype=torch.float32)
                        position_embeds = self.wpe(position_ids)
                        H("pos_embed", position_embeds)
                        hidden_states = inputs_embeds + position_embeds

                    for layer in islice(self.h, self.start_layer, self.end_layer):
                        hidden_states = layer(hidden_states)

                    H("resid_final", hidden_states)
                    hidden_states = self.ln_f(hidden_states)
                    H("final_ln", hidden_states)
        """.format(source=VLLM_GPT2_SRC))
        spec_path = _write_spec(VLLM_GPT2_SRC, spec_text)
        try:
            result = compile_spec(spec_path)
        finally:
            os.unlink(spec_path)

        tree = ast.parse(result)
        # Find get_hook_specs and check for dtype keyword in HookSpec calls
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "get_hook_specs"):
                # Find HookSpec calls with dtype
                found_dtype = False
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Call):
                        func_name = None
                        if isinstance(stmt.func, ast.Name):
                            func_name = stmt.func.id
                        elif isinstance(stmt.func, ast.Attribute):
                            func_name = stmt.func.attr
                        if func_name == "HookSpec":
                            for kw in stmt.keywords:
                                if kw.arg == "dtype":
                                    found_dtype = True
                                    assert "float32" in ast.unparse(kw.value)
                assert found_dtype, "No HookSpec with dtype= found in get_hook_specs"
                break
        else:
            pytest.fail("get_hook_specs not found")


# ===================================================================
# HF GPT-2 ground truth comparison
# ===================================================================

class TestGroundTruthHFGPT2:
    """Compare compiled HF GPT-2 output against hand-written ground truth."""

    @pytest.fixture(autouse=True)
    def _compile(self):
        from monitoring.compiler.compiler import compile_spec
        spec_text = GPT2_HF_SPEC.format(source=HF_GPT2_SRC)
        spec_path = _write_spec(HF_GPT2_SRC, spec_text)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.compiled_text = compile_spec(spec_path)
            self.compile_output = buf.getvalue()
            self.compiled_tree = ast.parse(self.compiled_text)
        finally:
            os.unlink(spec_path)
        with open(HF_GPT2_GT) as f:
            self.gt_tree = ast.parse(f.read())

    def test_hf_gpt2_compile_without_warnings(self):
        assert self.compile_output.strip() == ""

    def test_hf_gpt2_hook_declarations_match(self):
        for cls_name in ["GPT2Attention", "GPT2MLP", "GPT2Block", "GPT2Model"]:
            compiled_attrs = set(_get_hook_attrs(self.compiled_tree, cls_name))
            gt_attrs = set(_get_hook_attrs(self.gt_tree, cls_name))
            # GT may have extra hooks (attn_scores, pattern) the compiler doesn't generate
            assert compiled_attrs <= gt_attrs, (
                f"{cls_name}: compiled {compiled_attrs} not subset of GT {gt_attrs}")
            # But compiled must have at least the core hooks
            assert len(compiled_attrs) > 0, f"{cls_name}: no hooks generated"

    def test_hf_gpt2_hook_calls_match(self):
        # Only compare MLP and Block (Attention has complex branching)
        for cls_name in ["GPT2MLP"]:
            compiled_calls = [c[0] for c in _get_hook_calls(self.compiled_tree, cls_name)]
            gt_calls = [c[0] for c in _get_hook_calls(self.gt_tree, cls_name)]
            assert compiled_calls == gt_calls, (
                f"{cls_name}: compiled {compiled_calls} != GT {gt_calls}")
        # Block: compiled hooks are subset of GT (multi-line attn call can't be anchored)
        compiled_calls = set(c[0] for c in _get_hook_calls(self.compiled_tree, "GPT2Block"))
        gt_calls = set(c[0] for c in _get_hook_calls(self.gt_tree, "GPT2Block"))
        assert compiled_calls <= gt_calls, (
            f"GPT2Block: compiled {compiled_calls} not subset of GT {gt_calls}")
        assert len(compiled_calls) >= 5, (
            f"GPT2Block: too few hooks compiled: {compiled_calls}")

    def test_hf_gpt2_hook_specs_order(self):
        import re

        def extract_loop_types(src):
            tree = ast.parse(src) if isinstance(src, str) else src
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef)
                        and node.name == "get_hook_specs"):
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.For):
                            types = re.findall(
                                r'HOOK_TYPE_(\w+)', ast.unparse(stmt))
                            return types
            return []

        compiled_order = extract_loop_types(self.compiled_text)
        with open(HF_GPT2_GT) as f:
            gt_order = extract_loop_types(f.read())
        # Filter out ATTN_SCORES and PATTERN from GT (compiler doesn't generate these)
        gt_filtered = [t for t in gt_order if t not in ("ATTN_SCORES", "PATTERN")]
        assert compiled_order == gt_filtered, (
            f"compiled {compiled_order} != GT (filtered) {gt_filtered}")


# ===================================================================
# HF Qwen3 ground truth comparison
# ===================================================================

class TestGroundTruthHFQwen3:
    """Compare compiled HF Qwen3 output against hand-written ground truth."""

    @pytest.fixture(autouse=True)
    def _compile(self):
        from monitoring.compiler.compiler import compile_spec
        spec_text = QWEN3_HF_SPEC.format(source=HF_QWEN3_SRC)
        spec_path = _write_spec(HF_QWEN3_SRC, spec_text)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.compiled_text = compile_spec(spec_path)
            self.compile_output = buf.getvalue()
            self.compiled_tree = ast.parse(self.compiled_text)
        finally:
            os.unlink(spec_path)
        with open(HF_QWEN3_GT) as f:
            self.gt_tree = ast.parse(f.read())

    def test_hf_qwen3_compile_without_warnings(self):
        assert self.compile_output.strip() == ""

    def test_hf_qwen3_hook_declarations_match(self):
        for cls_name in ["Qwen3Attention", "Qwen3MLP", "Qwen3DecoderLayer", "Qwen3Model"]:
            compiled_attrs = set(_get_hook_attrs(self.compiled_tree, cls_name))
            gt_attrs = set(_get_hook_attrs(self.gt_tree, cls_name))
            assert compiled_attrs <= gt_attrs, (
                f"{cls_name}: compiled {compiled_attrs} not subset of GT {gt_attrs}")
            assert len(compiled_attrs) > 0, f"{cls_name}: no hooks generated"

    def test_hf_qwen3_hook_calls_match(self):
        compiled_mlp_calls = _get_hook_calls(self.compiled_tree, "Qwen3MLP")
        assert compiled_mlp_calls == [("hook_post", "x")]

        # DecoderLayer: compiled hooks are subset of GT (multi-line self_attn can't be anchored)
        compiled_calls = set(c[0] for c in _get_hook_calls(self.compiled_tree, "Qwen3DecoderLayer"))
        gt_calls = set(c[0] for c in _get_hook_calls(self.gt_tree, "Qwen3DecoderLayer"))
        assert compiled_calls <= gt_calls, (
            f"Qwen3DecoderLayer: compiled {compiled_calls} not subset of GT {gt_calls}")
        assert "hook_attn_out" in compiled_calls
        assert len(compiled_calls) >= 6, (
            f"Qwen3DecoderLayer: too few hooks compiled: {compiled_calls}")

        model_calls = _get_hook_calls(self.compiled_tree, "Qwen3Model")
        model_names = [c[0] for c in model_calls]
        assert "hook_resid_final" in model_names
        assert "hook_final_ln" in model_names
        assert model_names.index("hook_resid_final") < model_names.index("hook_final_ln")

    def test_hf_qwen3_hook_specs_order(self):
        import re

        def extract_loop_types(src):
            tree = ast.parse(src) if isinstance(src, str) else src
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef)
                        and node.name == "get_hook_specs"):
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.For):
                            types = re.findall(
                                r'HOOK_TYPE_(\w+)', ast.unparse(stmt))
                            return types
            return []

        compiled_order = extract_loop_types(self.compiled_text)
        with open(HF_QWEN3_GT) as f:
            gt_order = extract_loop_types(f.read())
        gt_filtered = [t for t in gt_order if t not in ("ATTN_SCORES", "PATTERN")]
        assert compiled_order == gt_filtered, (
            f"compiled {compiled_order} != GT (filtered) {gt_filtered}")
