"""DMI Hook Compiler — compile a spec into a hooked model file.

Reads a spec file (with H() markers) and the original model source,
then produces a new source file with HookPoint declarations, forward()
hook calls, and get_hook_specs() generated automatically.
"""
from __future__ import annotations

import ast
import os
from itertools import groupby
from typing import Optional, Sequence

from .types import HookDef, HookInsertionPlan, SpecClass, SpecInfo
from .utils import _block_header, _hook_attr
from .locator import _find_control_flow_in_method, _find_line, _get_indent, _plan_hook_insertion
from .generator import (
    _find_composition,
    _find_layer_iter,
    _generate_hook_specs,
    _interleave_hooks,
    _make_import_block,
    _resolve_path,
)


def parse_spec(spec_path: str) -> SpecInfo:
    with open(spec_path) as f:
        tree = ast.parse(f.read(), filename=spec_path)

    info = SpecInfo(source="", framework="hf")

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                fname = getattr(dec.func, "id", None) or getattr(dec.func, "attr", None)
                if fname == "spec":
                    for kw in dec.keywords:
                        if kw.arg == "source" and isinstance(kw.value, ast.Constant):
                            info.source = kw.value.value
                        if kw.arg == "framework" and isinstance(kw.value, ast.Constant):
                            info.framework = kw.value.value
                    if not info.source and dec.args and isinstance(dec.args[0], ast.Constant):
                        info.source = dec.args[0].value

        sc = SpecClass(name=node.name)
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                hooks = _extract_hooks(item.body)
                for h in hooks:
                    h.class_name = node.name
                    if h.target is None and h.anchor_kind == "stmt" and h.anchor_before:
                        try:
                            prev_ast = ast.parse(h.anchor_before, mode="exec").body[0]
                        except SyntaxError:
                            prev_ast = None
                        if isinstance(prev_ast, ast.Assign) and len(prev_ast.targets) == 1:
                            target = prev_ast.targets[0]
                            if isinstance(target, ast.Name):
                                h.target = target.id
                            elif isinstance(target, ast.Tuple):
                                raise ValueError(
                                    f"H(\"{h.name}\") after tuple unpack "
                                    f"'{h.anchor_before}' — must specify target"
                                )
                sc.hooks.extend(hooks)

        if sc.hooks:
            info.classes.append(sc)

    return info


def _extract_hooks(body: list[ast.stmt],
                   parent_block: Optional[ast.stmt] = None,
                   parent_branch: Optional[str] = None,
                   header_counts: Optional[dict] = None,
                   block_ordinals: Optional[dict] = None) -> list[HookDef]:
    if header_counts is None:
        header_counts = {}
    if block_ordinals is None:
        block_ordinals = {}
    results = []
    last_non_h = None

    for stmt in body:
        if isinstance(stmt, ast.While):
            raise NotImplementedError("while loops not supported in spec")
        if isinstance(stmt, ast.For):
            last_non_h = stmt
            header = _block_header(stmt)
            ordinal = header_counts.get(header, 0)
            header_counts[header] = ordinal + 1
            block_ordinals[id(stmt)] = ordinal
            results.extend(_extract_hooks(
                stmt.body,
                parent_block=stmt,
                parent_branch="body",
                header_counts=header_counts,
                block_ordinals=block_ordinals,
            ))
        elif isinstance(stmt, ast.If):
            last_non_h = stmt
            header = _block_header(stmt)
            ordinal = header_counts.get(header, 0)
            header_counts[header] = ordinal + 1
            block_ordinals[id(stmt)] = ordinal
            results.extend(_extract_hooks(
                stmt.body,
                parent_block=stmt,
                parent_branch="body",
                header_counts=header_counts,
                block_ordinals=block_ordinals,
            ))
            results.extend(_extract_hooks(
                stmt.orelse,
                parent_block=stmt,
                parent_branch="orelse",
                header_counts=header_counts,
                block_ordinals=block_ordinals,
            ))
        else:
            hook = _parse_h_call(stmt)
            if hook:
                if last_non_h is not None:
                    if isinstance(last_non_h, (ast.For, ast.If)):
                        hook.anchor_kind = "after_block"
                        hook.anchor_header = _block_header(last_non_h)
                        hook.block_ordinal = block_ordinals[id(last_non_h)]
                    else:
                        hook.anchor_kind = "stmt"
                        hook.anchor_before = ast.unparse(last_non_h)
                elif parent_block is not None:
                    hook.anchor_kind = "block_entry"
                    hook.anchor_header = _block_header(parent_block)
                    hook.anchor_branch = parent_branch
                    hook.block_ordinal = block_ordinals[id(parent_block)]
                results.append(hook)
            else:
                last_non_h = stmt
    return results


def _parse_h_call(stmt: ast.stmt) -> Optional[HookDef]:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    if not (isinstance(call.func, ast.Name) and call.func.id == "H"):
        return None
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None

    name = call.args[0].value
    target = call.args[1].id if len(call.args) >= 2 and isinstance(call.args[1], ast.Name) else None
    dtype = None
    for kw in call.keywords:
        if kw.arg == "dtype":
            dtype = ast.unparse(kw.value)
    return HookDef(name=name, target=target, dtype=dtype)


def _make_hook_decl_block(sc: SpecClass, indent: str) -> str:
    seen = set()
    lines = []
    for hook in sc.hooks:
        attr = _hook_attr(hook.name)
        if attr in seen:
            continue
        seen.add(attr)
        lines.append(f"{indent}self.{attr} = HookPoint()\n")
    return "".join(lines)


def _find_import_insert(lines: list[str]) -> int:
    import_insert = 0
    in_multiline = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_multiline:
            import_insert = i + 1
            if ')' in stripped:
                in_multiline = False
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_insert = i + 1
            if '(' in stripped and ')' not in stripped:
                in_multiline = True
    return import_insert


def compile_spec(
    spec_path: str,
    source_path: Optional[str] = None,
    framework: Optional[str] = None,
) -> str:
    spec = parse_spec(spec_path)
    if source_path is None:
        source_path = spec.source
    if framework is None:
        framework = spec.framework

    if not source_path or not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source not found: {source_path!r}")

    with open(source_path) as f:
        lines = f.readlines()

    tree = ast.parse("".join(lines), filename=source_path)
    spec_map = {sc.name: sc for sc in spec.classes}
    insertions: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in spec_map:
            continue

        sc = spec_map[node.name]

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                indent = _get_indent(lines[item.body[0].lineno - 1])
                block = _make_hook_decl_block(sc, indent)
                if block:
                    insertions.append((item.end_lineno, block))
                break

        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "forward":
                continue

            fwd_start = item.lineno - 1
            fwd_end = item.end_lineno
            search_from = fwd_start

            for hook in sc.hooks:
                plan = _plan_hook_insertion(
                    tree,
                    lines,
                    node.name,
                    item,
                    hook,
                    framework,
                    search_from,
                    fwd_start,
                    fwd_end,
                )
                if plan.insertion is not None:
                    insertions.append(plan.insertion)
                if plan.search_from is not None:
                    search_from = plan.search_from
                if plan.warning:
                    print(plan.warning)

    hook_specs = _generate_hook_specs(spec, tree, lines)
    if hook_specs:
        method_text, insert_line = hook_specs
        insertions.append((insert_line, method_text))

    insertions.append((_find_import_insert(lines), _make_import_block(spec) + "\n\n"))

    sorted_ins = sorted(insertions, key=lambda x: x[0], reverse=True)
    for _, group in groupby(sorted_ins, key=lambda x: x[0]):
        items = list(group)
        line_idx = items[0][0]
        combined = "".join(text for _, text in items)
        lines.insert(line_idx, combined)

    return "".join(lines)


def main(args: Optional[Sequence[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Compile hook spec into hooked model")
    parser.add_argument("spec", help="Spec file with H() markers")
    parser.add_argument("--source", default=None, help="Original model source (overrides @spec)")
    parser.add_argument(
        "--framework",
        default=None,
        help="Framework name (overrides @spec). 'hf' uses assignment-style hooks, others use side-effect style.",
    )
    parser.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    ns = parser.parse_args(args)

    result = compile_spec(ns.spec, source_path=ns.source, framework=ns.framework)
    if ns.output:
        with open(ns.output, "w") as f:
            f.write(result)
        print(f"Wrote hooked model to {ns.output}")
    else:
        print(result)
