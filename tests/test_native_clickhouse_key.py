"""Exercise the native live-table key parser without a server or CUDA build."""

from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.cpu


@pytest.fixture(scope="module")
def key_parser(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    build = tmp_path_factory.mktemp("clickhouse_key")
    source = build / "parse.cpp"
    source.write_text('''
#include "clickhouse_key.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    auto names = dmx_host::detail::ParseIdentifierKey(argv[1]);
    if (!names) return 2;
    for (const auto& name : *names) std::cout << name << '\\n';
}
''')
    executable = build / "parse"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
         f"-I{root / 'native' / 'csrc'}", str(source), "-o", str(executable)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return executable


@pytest.mark.parametrize("expression, expected", [
    ("model_id", ["model_id"]),
    ("`model_id`", ["model_id"]),
    ("(model_id)", ["model_id"]),
    ("(`model_id`)", ["model_id"]),
    (" \t( `model_id` )\n", ["model_id"]),
    ("model_id, phase", ["model_id", "phase"]),
    ("`model_id`, `phase`", ["model_id", "phase"]),
    ("(model_id, phase)", ["model_id", "phase"]),
    (" (`model_id`, `phase`) ", ["model_id", "phase"]),
    ("(_model_1, phase2)", ["_model_1", "phase2"]),
    ("", []),
    (" \t\n", []),
])
def test_plain_identifier_lists(key_parser, expression, expected):
    result = subprocess.run(
        [str(key_parser), expression], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, expression
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize("expression", [
    "()", "( )", "(model_id, )", "model_id,", "model_id, ",
    ",model_id", "(,model_id)", "(model_id,,phase)",
    "toDate(x)", "(toDate(x))", "tuple(model_id)",
    "((model_id))", "((model_id, phase))", "(model_id, (phase))",
    "(model_id), phase", "(model_id), (phase)",
    "(model_id", "model_id)", "(", ")",
    "model_id + 1", "(model_id + 1)", "model_id phase", "(model_id) suffix",
    "`model_id", "`model``id`", "`model(id)`", "1model", "(1model)",
])
def test_expressions_and_malformed_lists_are_rejected(key_parser, expression):
    result = subprocess.run(
        [str(key_parser), expression], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 2, expression
