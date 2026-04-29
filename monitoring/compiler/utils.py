from __future__ import annotations

import ast
import re

from .types import HookDef


_HOOK_TYPE_MAP = {
    "q": "HOOK_TYPE_Q", "k": "HOOK_TYPE_K", "v": "HOOK_TYPE_V",
    "z": "HOOK_TYPE_Z", "attn_scores": "HOOK_TYPE_ATTN_SCORES",
    "pattern": "HOOK_TYPE_PATTERN", "attn_out": "HOOK_TYPE_ATTN_OUT",
    "resid_pre": "HOOK_TYPE_RESID_PRE", "resid_mid": "HOOK_TYPE_RESID_MID",
    "resid_final": "HOOK_TYPE_RESID_FINAL",
    "ln1": "HOOK_TYPE_LN1", "ln2": "HOOK_TYPE_LN2",
    "mlp_in": "HOOK_TYPE_MLP_IN", "mlp_out": "HOOK_TYPE_MLP_OUT",
    "mlp_post": "HOOK_TYPE_MLP_POST", "post": "HOOK_TYPE_MLP_POST",
    "embed": "HOOK_TYPE_EMBED", "pos_embed": "HOOK_TYPE_POS_EMBED",
    "final_ln": "HOOK_TYPE_FINAL_LN",
    "token_ids": "HOOK_TYPE_TOKEN_IDS", "final_logits": "HOOK_TYPE_FINAL_LOGITS",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _block_header(stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.For):
        return f"for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}"
    if isinstance(stmt, ast.If):
        return f"if {ast.unparse(stmt.test)}"
    return ast.unparse(stmt)


def _hook_attr(name: str) -> str:
    return f"hook_{name}" if not name.startswith("hook_") else name


def _make_hook_call(hook: HookDef, framework: str) -> str:
    attr = _hook_attr(hook.name)
    if hook.target is None:
        raise ValueError(f"Hook '{hook.name}' has no target variable")
    if framework == "hf":
        return f"{hook.target} = self.{attr}({hook.target})"
    return f"self.{attr}({hook.target})"


def _call_func_name(node: ast.expr):
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
