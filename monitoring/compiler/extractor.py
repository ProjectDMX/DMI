"""DMI Hook Compiler — skeleton extractor.

Parses a model source file and outputs a simplified spec skeleton
containing only the data-flow statements of each class's forward().
"""
from __future__ import annotations

import ast
import os
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# AST classification
# ---------------------------------------------------------------------------

def _rooted_in_self(node: ast.AST) -> bool:
    """Check whether an attribute chain starts with ``self``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == "self"


def _has_self_call(node: ast.AST) -> bool:
    """Return True if *node* contains a ``self.xxx(...)`` call anywhere."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and _rooted_in_self(func):
                return True
    return False


def _is_data_flow(stmt: ast.stmt) -> bool:
    """Decide whether a statement belongs in the skeleton."""
    if isinstance(stmt, ast.Return):
        return True
    if isinstance(stmt, ast.For):
        return True

    # Assignments: keep if RHS has self.xxx(...), a top-level call, or a binop
    if isinstance(stmt, ast.Assign) and stmt.value is not None:
        v = stmt.value
        if _has_self_call(v):
            return True
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name):
            return True
        if isinstance(v, ast.BinOp):
            return True
        if isinstance(v, ast.Tuple):
            return True
        return False

    # Bare expressions: self.xxx(...)
    if isinstance(stmt, ast.Expr) and _has_self_call(stmt.value):
        return True

    return False


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

def _simplify_call(node: ast.Call) -> ast.Call:
    """Keep leading positional args and simple keyword args."""
    new_args = []
    for i, arg in enumerate(node.args):
        if isinstance(arg, ast.Name) or (isinstance(arg, ast.Attribute) and i < 4):
            new_args.append(arg)
        elif isinstance(arg, ast.Call) and i < 3:
            new_args.append(_simplify_call(arg))
        else:
            new_args.append(ast.Constant(value=...))
            break
    # Keep keyword args that are simple names (e.g. hidden_states=hidden_states)
    new_kw = [kw for kw in node.keywords
              if isinstance(kw.value, ast.Name) and kw.arg is not None]
    return ast.copy_location(
        ast.Call(func=node.func, args=new_args, keywords=new_kw), node)


def _simplify(node: ast.expr) -> ast.expr:
    """Recursively simplify an expression."""
    if isinstance(node, ast.Call):
        s = _simplify_call(node)
        s.args = [_simplify(a) for a in s.args]
        return s
    if isinstance(node, ast.BinOp):
        return ast.BinOp(left=_simplify(node.left), op=node.op,
                         right=_simplify(node.right))
    return node


def _unparse_stmt(stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.Assign):
        lhs = ", ".join(ast.unparse(t) for t in stmt.targets)
        return f"{lhs} = {ast.unparse(_simplify(stmt.value))}"
    if isinstance(stmt, ast.Expr):
        return ast.unparse(_simplify(stmt.value))
    if isinstance(stmt, ast.Return):
        return f"return {ast.unparse(stmt.value)}" if stmt.value else "return"
    return ast.unparse(stmt)


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------

def _extract_forward(func: ast.FunctionDef) -> list[str]:
    """Return skeleton lines for one forward() method."""
    sig = ast.unparse(func.args)
    lines = [f"def forward({sig}):"]

    def _walk_body(body: Sequence[ast.stmt], indent: int):
        prefix = "    " * indent
        for stmt in body:
            if isinstance(stmt, ast.For):
                lines.append(f"{prefix}for {ast.unparse(stmt.target)} "
                             f"in {ast.unparse(stmt.iter)}:")
                _walk_body(stmt.body, indent + 1)
            elif isinstance(stmt, ast.If):
                # Flatten: extract data-flow stmts from both branches
                _walk_body(stmt.body, indent)
                _walk_body(stmt.orelse, indent)
            elif _is_data_flow(stmt):
                lines.append(f"{prefix}{_unparse_stmt(stmt)}")

    _walk_body(func.body, 1)
    return lines


def _find_forward(cls: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            return item
    return None


def extract_skeleton(source_path: str) -> str:
    """Parse *source_path* and return a spec skeleton string."""
    with open(source_path) as f:
        tree = ast.parse(f.read(), filename=source_path)

    classes = [(node.name, _find_forward(node))
               for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and _find_forward(node)]

    if not classes:
        raise ValueError(f"No classes with forward() found in {source_path}")

    basename = os.path.basename(source_path)
    out = [
        f"# Auto-generated spec from: {basename}",
        '# Add H("name", variable) where you want hooks, then run: dmi compile',
        "",
        "from monitoring.compiler.dsl import H, spec",
        "",
        f'@spec(source="{basename}")',
    ]

    for i, (name, fwd) in enumerate(classes):
        if i > 0:
            out.append("")
        out.append(f"class {name}:")
        for line in _extract_forward(fwd):
            out.append(f"    {line}")
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args: Optional[Sequence[str]] = None):
    import argparse
    p = argparse.ArgumentParser(description="Extract hook spec skeleton")
    p.add_argument("source", help="Model source file")
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    ns = p.parse_args(args)

    result = extract_skeleton(ns.source)
    if ns.output:
        with open(ns.output, "w") as f:
            f.write(result)
        print(f"Wrote spec skeleton to {ns.output}")
    else:
        print(result)
