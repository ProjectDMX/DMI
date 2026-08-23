# Code organization

DMI uses a `src` layout with one canonical Python namespace. Compiled backend
sources and external repositories stay outside the installable package.

```text
src/dmi/
├── adapters/
│   ├── base.py                 Framework-neutral adapter contract
│   ├── types.py                Per-step adapter data models
│   └── huggingface/
│       ├── adapter.py          Hugging Face adapter implementation
│       ├── generation.py       Monitored generation entry points
│       └── model_shape.py      Hugging Face model-shape conversion
├── api/v1/                     Stable framework-integration facade
├── hooks/
│   ├── catalog.py              Native ABI hook catalog
│   ├── specs.py                Hook types and analytical shape planning
│   ├── dispatch.py             Producer dispatch and hook installation
│   ├── point.py                HookPoint model primitives
│   └── selection.py            Presets and PP/TP selection policy
├── storage/                    ClickHouse access and tensor reassembly
├── transport/                  Ring runtime and native-extension loader
├── config.py                   User-facing configuration models
└── engine.py                   Runtime orchestration

native/                         C++/CUDA backend sources and Makefile
third_party/                    External Git submodules
examples/                       Runnable end-user examples
benchmarks/                     Reproduction and profiling tools
tests/
└── native/ring/                Standalone CUDA ring tests
docs/                           Architecture and usage documentation
```

## Boundaries

- Import runtime functionality only through the `dmi` namespace. Install the
  checkout with `pip install -e .` during development.
- Framework plugins use `dmi.api.v1` as their stable integration boundary.
- Framework-specific behavior stays in `dmi.adapters`; hook specifications,
  storage, and transport remain framework-neutral.
- `dmi.hooks.specs` describes what is captured, `dmi.hooks.dispatch` connects
  specifications to HookPoints, and `dmi.transport.ring` owns runtime movement.
- Native loading stays isolated in `dmi.transport.native`, so importing `dmi`
  does not require CUDA or a compiled extension.
- `third_party` is never discovered or distributed as part of the DMI package.

## Native backend

Build from the repository root or directly from the native source directory:

```bash
make
# equivalent to: make -C native
```

The build keeps an intermediate copy under `native/` and writes the importable
`_native_backend` shared library to `src/dmi/`. The native hook ABI in
`native/csrc/ring/tensor_meta.h` is mirrored by `src/dmi/hooks/catalog.py`.

## Change checklist

1. Keep Python imports rooted at `dmi`; never add repository-root namespaces.
2. Keep generated artifacts and external dependencies out of `src/dmi`.
3. Run `python -m compileall -q src/dmi tests benchmarks examples`.
4. Run `python -m pytest -m cpu -q`, followed by `python -m pytest -q` where
   native and framework dependencies are available.
5. Verify the source distribution contains `dmi` and required native sources;
   verify source and wheel distributions both exclude `third_party` contents.
