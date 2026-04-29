from __future__ import annotations

import ast
import copy
import re
from typing import Optional

from .types import HookDef, HookInsertionPlan
from .utils import _block_header, _make_hook_call, _norm


def _find_line(lines: list[str], anchor: str, start: int = 0,
               end: Optional[int] = None) -> Optional[int]:
    anchor_n = _norm(anchor)
    end = end or len(lines)

    m = re.match(r'([\w,\s]+?)\s*=\s*(.*)', anchor_n)
    anchor_target = _norm(m.group(1)) if m else None
    anchor_rhs = m.group(2) if m else anchor_n
    anchor_first_target = None
    if anchor_target and "," in anchor_target:
        anchor_first_target = anchor_target.split(",", 1)[0].strip()
    call_match = re.search(r'(?:self\.)?(\w+)\s*\(', anchor_rhs)
    anchor_call = call_match.group(1) if call_match else None

    for i in range(start, end):
        if _norm(lines[i]) == anchor_n:
            return i

    if anchor_target and anchor_call:
        for i in range(start, end):
            line_n = _norm(lines[i])
            if (re.match(rf'{re.escape(anchor_target)}\s*=', line_n) or
                re.match(rf'\({re.escape(anchor_target)},', line_n)):
                if anchor_call in line_n:
                    return i
            if anchor_first_target:
                if (re.match(rf'{re.escape(anchor_first_target)}\s*,', line_n) or
                    re.match(rf'\({re.escape(anchor_first_target)}\s*,', line_n)):
                    if anchor_call in line_n:
                        return i

    if anchor_target and '+' in anchor_rhs:
        for i in range(start, end):
            line_n = _norm(lines[i])
            if re.match(rf'{re.escape(anchor_target)}\s*=', line_n) and '+' in line_n:
                return i

    return None


def _get_indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _names_from_target(node) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple):
        return {n for elt in node.elts for n in _names_from_target(elt)}
    if isinstance(node, ast.Starred) and isinstance(node.value, ast.Name):
        return {node.value.id}
    return set()


def _collect_bound_names(func_node: ast.FunctionDef, up_to_lineno: int) -> set[str]:
    names = {arg.arg for arg in func_node.args.args}
    for stmt in ast.walk(func_node):
        if not hasattr(stmt, 'lineno') or stmt.lineno >= up_to_lineno:
            continue
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                names.update(_names_from_target(t))
        elif isinstance(stmt, ast.For):
            names.update(_names_from_target(stmt.target))
        elif isinstance(stmt, ast.AugAssign):
            names.update(_names_from_target(stmt.target))
    return names


def _find_control_flow_in_method(
    tree: ast.Module, class_name: str, header: str,
    ordinal: int = 0,
) -> Optional[tuple[int, int, int, int]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "forward":
                continue
            blocks = []
            for stmt in ast.walk(item):
                if isinstance(stmt, (ast.For, ast.If)):
                    blocks.append(stmt)
            blocks.sort(key=lambda s: s.lineno)
            count = 0
            for b in blocks:
                if _headers_match(b, header):
                    if count == ordinal:
                        body_start = b.body[0].lineno if b.body else b.lineno + 1
                        orelse_start = (b.orelse[0].lineno
                                        if hasattr(b, 'orelse') and b.orelse
                                        else 0)
                        return (b.lineno, b.end_lineno, body_start, orelse_start)
                    count += 1
    return None


def _find_stmt_end_lineno(func: ast.FunctionDef, lineno: int) -> Optional[int]:
    candidates = []
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.stmt):
            continue
        if isinstance(stmt, (ast.For, ast.If, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stmt_lineno = getattr(stmt, "lineno", None)
        stmt_end = getattr(stmt, "end_lineno", None)
        if stmt_lineno is None or stmt_end is None:
            continue
        if stmt_lineno <= lineno <= stmt_end:
            candidates.append((stmt_end - stmt_lineno, stmt_end))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _headers_match(block: ast.stmt, header: str) -> bool:
    block_header = _block_header(block)
    if _norm(block_header) == _norm(header):
        return True
    if not isinstance(block, ast.For):
        return False
    block_parts = _parse_for_header(block_header)
    header_parts = _parse_for_header(header)
    if block_parts is None or header_parts is None:
        return False
    (block_target, block_iter), (header_target, header_iter) = block_parts, header_parts
    if _norm(block_target) != _norm(header_target):
        return False
    block_iter_n = _norm(block_iter)
    header_iter_n = _norm(header_iter)
    return (
        block_iter_n == header_iter_n
        or block_iter_n.startswith(f"{header_iter_n}[")
        or header_iter_n.startswith(f"{block_iter_n}[")
    )


def _parse_for_header(header: str) -> Optional[tuple[str, str]]:
    match = re.match(r"for\s+(.+?)\s+in\s+(.+)", header.strip())
    if match is None:
        return None
    return match.group(1), match.group(2)


def _try_rewrite_call_arg_hook(
    func: ast.FunctionDef,
    lines: list[str],
    hook: HookDef,
    framework: str,
    start_lineno: int,
) -> bool:
    if not hook.anchor_before or hook.target is None:
        return False
    try:
        prev_stmt = ast.parse(hook.anchor_before, mode="exec").body[0]
    except SyntaxError:
        return False
    if not isinstance(prev_stmt, ast.Assign) or len(prev_stmt.targets) != 1:
        return False
    if not (isinstance(prev_stmt.targets[0], ast.Name) and prev_stmt.targets[0].id == hook.target):
        return False
    anchor_expr = prev_stmt.value

    candidate_stmts = sorted(
        (
            stmt for stmt in ast.walk(func)
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.value, ast.Call)
            and stmt.lineno >= start_lineno
            and stmt.lineno == stmt.end_lineno
        ),
        key=lambda stmt: stmt.lineno,
    )
    for stmt in candidate_stmts:
        call = stmt.value
        for i, arg in enumerate(call.args):
            if ast.dump(arg, include_attributes=False) != ast.dump(anchor_expr, include_attributes=False):
                continue
            new_call = copy.deepcopy(call)
            new_call.args[i] = ast.Name(id=hook.target, ctx=ast.Load())
            indent = _get_indent(lines[stmt.lineno - 1])
            lhs = ", ".join(ast.unparse(t) for t in stmt.targets)
            hook_code = _make_hook_call(hook, framework)
            replacement = (
                f"{indent}{hook.target} = {ast.unparse(anchor_expr)}\n"
                f"{indent}{hook_code}\n"
                f"{indent}{lhs} = {ast.unparse(new_call)}\n"
            )
            lines[stmt.lineno - 1] = replacement
            return True
    return False


def _validate_target_bound(
    func: ast.FunctionDef,
    class_name: str,
    hook: HookDef,
    up_to_lineno: int,
) -> None:
    if hook.target is None:
        return
    bound = _collect_bound_names(func, up_to_lineno)
    if hook.target not in bound:
        raise ValueError(
            f"Hook '{hook.name}' references '{hook.target}' "
            f"which is not bound in {class_name}.forward()"
        )


def _build_hook_insertion(
    lines: list[str],
    func: ast.FunctionDef,
    class_name: str,
    hook: HookDef,
    framework: str,
    insert_line_idx: int,
    indent_line_idx: int,
    validate_lineno: int,
) -> tuple[int, str]:
    _validate_target_bound(func, class_name, hook, validate_lineno)
    indent = _get_indent(lines[indent_line_idx])
    return insert_line_idx, f"{indent}{_make_hook_call(hook, framework)}\n"


def _locate_stmt_anchor(lines: list[str], anchor_before: str, search_from: int,
                        fwd_start: int) -> Optional[int]:
    idx = _find_line(lines, anchor_before, start=search_from)
    if idx is None:
        idx = _find_line(lines, anchor_before, start=fwd_start)
    return idx


def _plan_block_hook_insertion(
    tree: ast.Module,
    lines: list[str],
    class_name: str,
    func: ast.FunctionDef,
    hook: HookDef,
    framework: str,
) -> HookInsertionPlan:
    loc = _find_control_flow_in_method(tree, class_name, hook.anchor_header, hook.block_ordinal)
    if loc is None:
        return HookInsertionPlan(
            warning=f"WARNING: could not find block for hook '{hook.name}' in {class_name}.forward()"
        )

    header_line, end_line, body_start, orelse_start = loc
    if hook.anchor_kind == "after_block":
        insertion = _build_hook_insertion(
            lines, func, class_name, hook, framework, end_line, header_line - 1, end_line + 1
        )
        return HookInsertionPlan(insertion=insertion)

    if hook.anchor_kind == "block_entry":
        entry = orelse_start if hook.anchor_branch == "orelse" and orelse_start > 0 else body_start
        insertion = _build_hook_insertion(
            lines, func, class_name, hook, framework, entry - 1, entry - 1, entry
        )
        return HookInsertionPlan(insertion=insertion)

    raise ValueError(f"unsupported block anchor kind: {hook.anchor_kind}")


def _plan_stmt_hook_insertion(
    lines: list[str],
    class_name: str,
    func: ast.FunctionDef,
    hook: HookDef,
    framework: str,
    search_from: int,
    fwd_start: int,
    fwd_end: int,
) -> HookInsertionPlan:
    if hook.anchor_before is None:
        target_line = func.body[0].lineno - 1
        insertion = _build_hook_insertion(
            lines, func, class_name, hook, framework, target_line, target_line, func.body[0].lineno
        )
        return HookInsertionPlan(insertion=insertion)

    idx = _locate_stmt_anchor(lines, hook.anchor_before, search_from, fwd_start)
    if idx is None:
        if _try_rewrite_call_arg_hook(func, lines, hook, framework, start_lineno=search_from + 1):
            return HookInsertionPlan()
        return HookInsertionPlan(
            warning=f"WARNING: could not find anchor for hook '{hook.name}' in "
                    f"{class_name}.forward(): {hook.anchor_before!r}"
        )

    stmt_end_lineno = _find_stmt_end_lineno(func, idx + 1)
    if stmt_end_lineno is not None:
        insert_after = stmt_end_lineno - 1
    else:
        insert_after = idx
        while insert_after + 1 < fwd_end:
            next_line = lines[insert_after + 1] if insert_after + 1 < len(lines) else ""
            next_stripped = next_line.strip()
            if not next_stripped or _get_indent(next_line) <= _get_indent(lines[idx]):
                break
            if len(_get_indent(next_line)) > len(_get_indent(lines[idx])):
                insert_after += 1
            else:
                break

    insertion = _build_hook_insertion(
        lines, func, class_name, hook, framework, insert_after + 1, idx, insert_after + 2
    )
    return HookInsertionPlan(insertion=insertion, search_from=insert_after + 1)


def _plan_hook_insertion(
    tree: ast.Module,
    lines: list[str],
    class_name: str,
    func: ast.FunctionDef,
    hook: HookDef,
    framework: str,
    search_from: int,
    fwd_start: int,
    fwd_end: int,
) -> HookInsertionPlan:
    if hook.anchor_kind in {"after_block", "block_entry"}:
        return _plan_block_hook_insertion(tree, lines, class_name, func, hook, framework)
    return _plan_stmt_hook_insertion(
        lines, class_name, func, hook, framework, search_from, fwd_start, fwd_end
    )
