# Code organization

DMI uses one canonical Python package and keeps its original import paths as a
compatibility layer. New code should import from `dmi`; existing applications
using `monitoring` or `integration.hf_adapter` continue to resolve to the same
implementation objects.

```text
dmi/
├── adapters/                 Framework-neutral adapter contracts
│   └── huggingface/          HuggingFace adapter and generation helpers
├── api/v1/                   Stable integration facade for framework plugins
├── hooks/                    Hook primitives, ABI catalog, and selection policy
├── storage/                  ClickHouse access and captured-state reassembly
├── transport/                Ring transport and lazy native-extension loader
├── config.py                 User-facing configuration models
└── engine.py                 Runtime orchestration

monitoring/                   DMI 1.x Python shims and native C++/CUDA sources
integration/                  DMI 1.x HuggingFace shims and external submodules
examples/                     Runnable end-user examples
benchmarks/                   Reproduction and profiling tools
tests/                        Unit, compatibility, native, and end-to-end tests
docs/                         Architecture and usage documentation
```

## Import policy

- Use `dmi` for core configuration and engine APIs.
- Use `dmi.api.v1` from framework integrations that need the stable facade.
- Use `dmi.adapters.huggingface` for the HuggingFace adapter and generation
  helpers.
- Treat `monitoring.*` and the root `integration` Python modules as deprecated
  compatibility paths. They are module aliases, so class identity, module
  state, monkeypatching, and serialized qualified names continue to behave as
  they did before the reorganization.
- Keep framework-independent code out of adapter packages. Keep native loading
  isolated in `dmi.transport.native` so ordinary imports work without CUDA or a
  built extension.

## Native source compatibility

The native C++/CUDA sources and Makefile intentionally remain under
`monitoring/` during the 1.x compatibility window. This preserves existing
build commands and extension lookup locations:

```bash
make -C monitoring
```

Hook IDs and their metadata form a native ABI. The pure-Python catalog in
`dmi/hooks/catalog.py` mirrors `monitoring/csrc/ring/tensor_meta.h`, allowing
CPU-only imports; a native test verifies that both tables agree whenever the
extension is available.

## Change checklist

Before merging an organizational change:

1. Preserve or deliberately version every public import path.
2. Keep compatibility shims free of independent mutable state.
3. Run `python -m compileall -q dmi monitoring integration benchmarks examples`.
4. Run `python -m pytest -m cpu -q`, then `python -m pytest -q` where the
   required native and framework dependencies are available.
5. Build a wheel and inspect it for both the canonical package and the 1.x
   compatibility namespaces.
