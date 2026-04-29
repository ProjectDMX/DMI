from __future__ import annotations

import ast
from typing import Optional

from .types import SpecInfo
from .utils import _HOOK_TYPE_MAP, _call_func_name, _hook_attr
from .locator import _find_control_flow_in_method, _find_line


def _find_composition(tree: ast.Module, spec_names: set[str]) -> dict:
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in spec_names:
            continue
        result[node.name] = {}
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != '__init__':
                continue
            for stmt in ast.walk(item):
                if isinstance(stmt, ast.Assign):
                    _detect_child(stmt, spec_names, result[node.name])
    return result


def _detect_child(stmt: ast.Assign, spec_names: set[str], out: dict):
    tgt = stmt.targets[0] if len(stmt.targets) == 1 else None
    val = stmt.value

    if (isinstance(tgt, ast.Attribute)
            and getattr(tgt.value, 'id', None) == 'self'
            and isinstance(val, ast.Call)):
        name = _call_func_name(val)
        if name in spec_names:
            out[tgt.attr] = (name, False)
        if _call_func_name(val) == 'ModuleList':
            for child in ast.walk(val):
                if isinstance(child, ast.Call):
                    n = _call_func_name(child)
                    if n and n in spec_names:
                        out[tgt.attr] = (n, True)
                        return

    if (isinstance(tgt, ast.Tuple) and isinstance(val, ast.Call)
            and _call_func_name(val) == 'make_layers'):
        elts = tgt.elts
        if len(elts) >= 3 and isinstance(elts[-1], ast.Attribute):
            attr = elts[-1].attr
            for child in ast.walk(val):
                if isinstance(child, ast.Call):
                    n = _call_func_name(child)
                    if n and n in spec_names:
                        out[attr] = (n, True)
                        return


def _find_layer_iter(tree: ast.Module, class_name: str) -> Optional[tuple]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != '__init__':
                continue
            for stmt in ast.walk(item):
                if not isinstance(stmt, ast.Assign):
                    continue
                tgt = stmt.targets[0] if len(stmt.targets) == 1 else None
                val = stmt.value
                if (isinstance(tgt, ast.Tuple) and isinstance(val, ast.Call)
                        and _call_func_name(val) == 'make_layers'):
                    elts = tgt.elts
                    if len(elts) >= 3:
                        return (
                            elts[2].attr if isinstance(elts[2], ast.Attribute) else None,
                            elts[0].attr if isinstance(elts[0], ast.Attribute) else None,
                            elts[1].attr if isinstance(elts[1], ast.Attribute) else None,
                        )
                if (isinstance(tgt, ast.Attribute)
                        and getattr(tgt.value, 'id', None) == 'self'
                        and isinstance(val, ast.Call)
                        and _call_func_name(val) == 'ModuleList'):
                    return (tgt.attr, None, None)
    return None


def _split_at_layer_loop(tree: ast.Module, lines: list[str],
                         class_name: str, hooks: list) -> tuple:
    loop_line = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                for stmt in item.body:
                    if isinstance(stmt, ast.For):
                        loop_line = stmt.lineno
                        break
    if loop_line is None:
        return hooks, []

    pre, post = [], []
    for h in hooks:
        if h.anchor_kind == "after_block":
            loc = _find_control_flow_in_method(tree, class_name, h.anchor_header, h.block_ordinal)
            if loc and loc[1] < loop_line:
                pre.append(h)
            else:
                post.append(h)
            continue
        if h.anchor_kind == "block_entry":
            loc = _find_control_flow_in_method(tree, class_name, h.anchor_header, h.block_ordinal)
            if loc and loc[0] < loop_line:
                pre.append(h)
            else:
                post.append(h)
            continue
        if h.anchor_before is None:
            pre.append(h)
            continue
        idx = _find_line(lines, h.anchor_before)
        if idx is not None and idx + 1 < loop_line:
            pre.append(h)
        else:
            post.append(h)
    return pre, post


def _interleave_hooks(spec_map: dict, composition: dict,
                      class_name: str) -> list[tuple]:
    sc = spec_map.get(class_name)
    if sc is None:
        return []
    children = composition.get(class_name, {})
    result = []
    for hook in sc.hooks:
        if hook.anchor_before:
            for attr, (child_cls, _) in children.items():
                if f'self.{attr}(' in hook.anchor_before:
                    result.extend(_interleave_hooks(spec_map, composition, child_cls))
                    break
        result.append((hook, class_name))
    return result


def _resolve_path(composition: dict, from_class: str,
                  target_class: str, prefix: str = 'layer') -> Optional[str]:
    if from_class == target_class:
        return prefix
    for attr, (child_cls, _) in composition.get(from_class, {}).items():
        if child_cls == target_class:
            return f'{prefix}.{attr}'
        sub = _resolve_path(composition, child_cls, target_class, f'{prefix}.{attr}')
        if sub:
            return sub
    return None


def _generate_hook_specs(spec: SpecInfo, tree: ast.Module,
                         lines: list[str]) -> Optional[tuple[str, int]]:
    spec_names = {sc.name for sc in spec.classes}
    spec_map = {sc.name: sc for sc in spec.classes}

    composition = _find_composition(tree, spec_names)
    all_children = {c for d in composition.values() for c, _ in d.values()}
    roots = [sc.name for sc in spec.classes if sc.name not in all_children]
    if not roots:
        return None
    root = roots[-1]
    root_sc = spec_map[root]

    layer_info = _find_layer_iter(tree, root)
    layer_class = None
    if layer_info:
        for _, (cls, is_layer) in composition.get(root, {}).items():
            if is_layer:
                layer_class = cls
                break

    if layer_info and layer_class:
        pre, post = _split_at_layer_loop(tree, lines, root, root_sc.hooks)
        per_layer = _interleave_hooks(spec_map, composition, layer_class)
    else:
        pre, post, per_layer = root_sc.hooks, [], []

    insert_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == root:
            insert_line = node.end_lineno
            break
    if insert_line is None:
        return None

    i1, i2, i3 = "    ", "        ", "            "
    code = [f"\n{i1}def get_hook_specs(self) -> list[HookSpec]:"]
    code.append(f"{i2}specs: list[HookSpec] = []")

    for h in pre:
        dtype_arg = f", dtype={h.dtype}" if h.dtype else ""
        code.append(f"{i2}specs.append(HookSpec("
                    f"{_HOOK_TYPE_MAP.get(h.name, repr(h.name))}, "
                    f"self.{_hook_attr(h.name)}{dtype_arg}))")

    if layer_info and per_layer:
        la, sa, ea = layer_info
        if sa and ea:
            code.append(f"{i2}for i in range(self.{sa}, self.{ea}):")
            code.append(f"{i3}layer = self.{la}[i]")
        else:
            code.append(f"{i2}for i, layer in enumerate(self.{la}):")
        for h, owner in per_layer:
            path = _resolve_path(composition, layer_class, owner) or 'layer'
            dtype_arg = f", dtype={h.dtype}" if h.dtype else ""
            code.append(f"{i3}specs.append(HookSpec("
                        f"{_HOOK_TYPE_MAP.get(h.name, repr(h.name))}, "
                        f"{path}.{_hook_attr(h.name)}, layer_no=i{dtype_arg}))")

    for h in post:
        dtype_arg = f", dtype={h.dtype}" if h.dtype else ""
        code.append(f"{i2}specs.append(HookSpec("
                    f"{_HOOK_TYPE_MAP.get(h.name, repr(h.name))}, "
                    f"self.{_hook_attr(h.name)}{dtype_arg}))")

    code.append(f"{i2}return specs")
    return "\n".join(code) + "\n", insert_line


def _make_import_block(spec: SpecInfo) -> str:
    all_names = [h.name for sc in spec.classes for h in sc.hooks]
    types = sorted({_HOOK_TYPE_MAP[n] for n in all_names if n in _HOOK_TYPE_MAP})

    lines = ["from monitoring.hook_points import HookPoint"]
    if types:
        lines.append("from monitoring.ring_transport import (")
        lines.append("    HookSpec,")
        for t in types:
            lines.append(f"    {t},")
        lines.append(")")
    return "\n".join(lines)
