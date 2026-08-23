# Code organization

DMI has one canonical Python namespace. Runtime code belongs under `dmi`,
compiled backend sources belong under `native`, and external repositories are
isolated under `third_party`.

```text
dmi/
├── adapters/                 Framework-neutral contracts and implementations
│   └── huggingface/          HuggingFace adapter and generation helpers
├── api/v1/                   Stable integration facade for framework plugins
├── hooks/                    Hook primitives, ABI catalog, and selection policy
├── storage/                  ClickHouse access and captured-state reassembly
├── transport/                Ring transport and native-extension loader
├── config.py                 User-facing configuration models
└── engine.py                 Runtime orchestration

native/                       C++/CUDA backend sources and Makefile
third_party/                  Git submodules and other external source trees
examples/                     Runnable end-user examples
benchmarks/                   Reproduction and profiling tools
tests/                        Unit, native, integration, and end-to-end tests
docs/                         Architecture and usage documentation
```

## Boundaries

- Import runtime functionality only from `dmi` or one of its documented
  subpackages.
- Framework plugins use `dmi.api.v1` as their stable integration boundary.
- Framework-specific behavior stays inside `dmi.adapters`; reusable hooks,
  transport, and storage code must remain framework-neutral.
- Native loading stays isolated in `dmi.transport.native`, so ordinary package
  imports do not require CUDA or a compiled shared library.
- Code under `third_party` is owned by its upstream repository. The DMI package
  must not discover or distribute those trees as Python namespaces.

## Native backend

Build the C++/CUDA backend from the repository root or directly from its source
directory:

```bash
make
# equivalent to: make -C native
```

The build keeps an intermediate copy in `native/` and places the importable
`_native_backend` shared library in `dmi/`. Hook IDs and metadata form a native
ABI; `dmi/hooks/catalog.py` mirrors `native/csrc/ring/tensor_meta.h`, and the
native test suite verifies both definitions agree when the extension exists.

## Change checklist

1. Keep every Python import rooted at `dmi`.
2. Keep generated artifacts and external dependencies out of runtime packages.
3. Run `python -m compileall -q dmi benchmarks examples`.
4. Run `python -m pytest -m cpu -q`, followed by `python -m pytest -q` where
   native and framework dependencies are available.
5. Build the source distribution and confirm that it contains `dmi` and
   `native`, but no external submodule contents.
