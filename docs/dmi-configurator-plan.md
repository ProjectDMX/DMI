# DMI-configurator — Design & Implementation Plan

Status: **phases 1-7 implemented** (phase 8, runtime policy, deferred by design)
Branch: `feat/dmi-configurator`
Verified against: `cb4e490`

## Running it

```bash
pip install -e ".[ui]"
dmi ui ./Qwen3-8B
```

Then open <http://127.0.0.1:8000>. `dmi ui` takes the model however you have
it — a model directory, a `config.json`, a Hugging Face model id (needs
`transformers`), or a DMI descriptor YAML:

```bash
dmi ui ./Qwen3-8B/config.json
dmi ui Qwen/Qwen3-8B
dmi ui examples/model_descriptors/llama3-8b.yaml
```

To start from an existing configuration and save back to it:

```bash
dmi ui ./Qwen3-8B --config attention-debug.dmi.yaml
```

Descriptors are generated, never hand-typed:

```bash
dmi describe-model ./Qwen3-8B --output qwen3-8b.yaml
```

Without an install, `python -m dmi.cli ...` works from a checkout with `src` on
`PYTHONPATH`. `--host` and `--port` move the bind address; the default is
loopback only.

Using the configuration from Python, with no browser involved:

```python
from dmi.configuration import compile_config, load_config, ModelContext

config = load_config("attention-debug.dmi.yaml")
compiled = compile_config(config, ModelContext(specs=model.get_hook_specs(), shape=cfg))
```

---

## 1. Purpose

One job:

> Turn a model descriptor plus a user's visual selections into a validated DMI
> runtime YAML configuration.

The architecture visualization is the main interaction surface. The generated
YAML is the product artifact. This is a configuration tool, not an
observability dashboard.

## 2. The central architectural decision

DMI does **not** switch from Python configuration to YAML. YAML is added as a
serialization format in front of the existing configuration mechanisms, via a
loader/compiler pair:

```text
             DMI-configurator
                    │
                    ▼
              user config.yaml
                    │
                    ▼
              DMI config loader          load_config()   — knows YAML, not runtime
                    │
                    ▼
                 DMIConfig               canonical in-process representation
                    │
                    ▼
             DMI config compiler         compile_config() — knows runtime, not YAML
                    │
                    ▼
             CompiledDMIConfig
                    │
         ┌──────────┼───────────┐
         ▼          ▼           ▼
    HookSpecs  CaptureSchedule Runtime
         └──────────┴───────────┘
                    ▼
               DMI runtime
```

The separation is the point: `load_config()` knows nothing about the runtime,
`compile_config()` knows nothing about YAML. Existing DMI machinery remains the
execution layer.

Both configuration paths feed the same object:

```text
        ┌── existing integration arguments ──┐
        │                                    │
        ▼                                    │
   DMIConfig builder                         │
        ▲                                    │
        │                                    │
        └────────── YAML loader ─────────────┘
                         │
                         ▼
                    DMI runtime
```

What is explicitly rejected: `yaml.safe_load()` producing a dict that every
module then indexes into (`config["observations"]["hooks"]`). One parser, one
typed object, everything else downstream.

## 3. Three distinct concepts

| Concept | Answers | Owner |
|---|---|---|
| Model descriptor | What is this model? | new, authored per model |
| DMI hook catalog | What can DMI observe? | `src/dmi/hooks/catalog.py` (existing, authoritative) |
| User configuration | What does the user want DMI to do? | new, `DMIConfig` |

The frontend must never contain the authoritative hook list. It derives its
choices from the catalog through a backend adapter.

## 4. Model descriptor

Declarative. No screen coordinates, no frontend-specific information.

**Derived from the framework, not written by hand.** The framework already
knows the model, and DMI already reads a Hugging-Face-shaped config:
`make_model_shape_from_hf_config` extracts every topology field except the
layer count. `dmi describe-model` adds the layer count and identity, reusing
that extractor rather than duplicating it, and `dmi ui` accepts the same
framework sources directly so a descriptor file is optional.

A descriptor file still earns its place: the configurator runs where the model
is not loaded — pick layers on a laptop, run the capture on a cluster. The
runtime never reads it; `compile_config` takes its shape from the adapter's
live `detect_model_shape(model)`. So a stale descriptor can mislead you while
authoring, but it cannot corrupt a capture.

```yaml
schema_version: 1
model:
  id: qwen3-8b
  name: Qwen3 8B
  architecture: decoder_transformer
topology:
  num_layers: 32
  hidden_size: 4096
  num_attention_heads: 32
  num_kv_heads: 8
  intermediate_size: 14336
  num_experts: 0
  top_k: 0
```

From this the UI infers: 32 layers, attention exists, MLP exists, MoE
unavailable, top-k expert observations unavailable.

Architecture types: `decoder_transformer` for v1. Later
`encoder_decoder_transformer`, `vision_transformer`, `moe_transformer`.

## 5. User configuration

```yaml
version: 1
observations:
  layers:
    start: 8
    end: 15
  hooks:
    - resid_pre
    - q
    - k
    - v
    - pattern
    - mlp_out
schedule:
  step_stride: 4
  request_stride: 1
  capture_prefill: true
  capture_decode: true
policy:
  objective: balanced
```

Human-readable structure is the canonical new format. The flat
`dmx_hook_selection: q,k,v` form stays supported through a compatibility
adapter, not as the canonical representation.

Two files stay separate: `qwen3-8b.model.yaml` describes what the model is;
`attention-debug.dmi.yaml` describes what to capture. The configurator uses
both at design time; the runtime needs only the latter plus whatever model
information the framework already supplies.

### Versioning

`version: 1` is mandatory. Research configuration files outlive the code that
generated them, so the loader dispatches on version and raises
`UnsupportedConfigVersion` rather than guessing.

## 6. Typed schema

```python
@dataclass
class LayerSelection:
    start: int
    end: int

@dataclass
class ObservationConfig:
    hooks: list[str]
    layers: LayerSelection | None

class RuntimePolicy(Enum):
    COMPLETENESS = "completeness"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

@dataclass
class DMIConfig:
    version: int
    observations: ObservationConfig
    schedule: CaptureSchedule
    runtime: RuntimeConfig
    policy: RuntimePolicy | None = None
```

Layer selection is structured. Never a string `"8-15"`, and never encoded into
hook syntax such as `pattern@8-15` or `pattern[8:15]`.

## 7. Compilation

```python
@dataclass
class CompiledDMIConfig:
    hook_specs: list[HookSpec]
    schedule: CaptureSchedule
    runtime: RuntimeConfig
    policy: RuntimePolicy | None
```

```python
def compile_config(config, model_context):
    specs = select_hook_specs(
        model_context.specs,
        ",".join(config.observations.hooks),
        cfg=model_context.shape,
    )
    specs = filter_by_layers(specs, config.observations.layers)
    return CompiledDMIConfig(
        hook_specs=specs,
        schedule=config.schedule,
        runtime=config.runtime,
        policy=config.policy,
    )
```

`compile_config()` returns objects DMI actually needs, not another dictionary.

### Layer filtering

The only genuinely new runtime operation. Applied after hook resolution, as a
pure filter on `HookSpec.layer_no`:

```text
selection string → HookSpecs → filter by layer_no → selected HookSpecs
```

## 8. Precedence

```text
explicit CLI override  >  YAML config  >  DMI defaults
```

Overrides stay minimal initially to avoid configuration becoming confusing.

## 9. Runtime policy

Policy expresses what DMI should optimize when observation and inference
performance come into tension.

```text
Completeness   protect observation delivery
Balanced       bounded observation degradation
Performance    protect the serving path
```

This is a runtime semantics feature, not a hook preset. Phasing:

- **A** — expose the concept in the configuration model only.
- **B** — define actual runtime behavior (backlog, drain prioritization,
  dropping, backpressure, ring pressure, flush behavior).
- **C** — connect policy to the runtime's resource management.
- **D** — serialize as `policy.objective`.

**Rule:** do not ship a UI where selecting "Performance" has no effect. Until
the runtime contract exists, the UI must not claim the control changes DMI
behavior.

## 10. Backend

FastAPI, localhost only. No database, no authentication, no websockets, no
persistent service.

```text
GET  /api/model
GET  /api/catalog
POST /api/validate
POST /api/config/serialize
POST /api/config/parse
```

The backend invokes the same Python configuration objects the runtime uses, so
what the UI calls valid is exactly what DMI calls valid. That is the main
reason the schema lives inside DMI rather than in a separate web project.

## 11. Frontend

HTML, CSS, JavaScript, SVG. No React, no Node build chain, no diagram
framework. Install experience must stay:

```bash
pip install "dmi[ui]"
dmi ui qwen3-8b.yaml
```

SVG because each architecture object becomes a real DOM element
(`<g data-layer="12">`) supporting click, hover, selected state, tooltips,
labels, accessibility, and animation.

### Layout

```text
┌─────────────────────────────────────────────────────────────────┐
│ DMI-configurator                         Open   Save   Export   │
│ Qwen3 8B · Decoder Transformer · 32 layers            ✓ Valid   │
├───────────────────────────────────────────┬─────────────────────┤
│          MODEL ARCHITECTURE               │ SELECTED COMPONENT  │
│              Embedding                    │ Attention           │
│                  │                        │ □ Q  □ K  □ V       │
│        ┌─────────▼─────────┐              │ □ Pattern           │
│        │ Attention         │              │                     │
│        │  Q K V Pattern    │              │ Layers              │
│        └─────────┬─────────┘              │ [ 8 ─────── 15 ]    │
│        ┌─────────▼─────────┐              │                     │
│        │ MLP / MoE         │              │                     │
│        └─────────┬─────────┘              │                     │
│                  ⋮                        │                     │
├───────────────────────────────────────────┴─────────────────────┤
│ CAPTURE                                                         │
│ Every [4] steps   Request [1]   ☑ Prefill   ☑ Decode            │
│ SYSTEM OBJECTIVE                                                │
│ ○ Completeness     ● Balanced      ○ Performance                │
│ Configuration valid                             [Generate YAML] │
└─────────────────────────────────────────────────────────────────┘
```

Single page, not a wizard.

### Architecture renderer

Driven by structured metadata, not hardcoded HTML:

```javascript
renderArchitecture({architecture: "decoder_transformer", num_layers: 32})
```

Selection events `selectLayer(12)`, `selectLayerRange(8, 15)`,
`selectComponent("attention")`, `selectHook("pattern")` update central state.

Component states: `normal`, `hovered`, `selected`, `partially-selected`,
`unavailable`. Visual feedback replaces filling the diagram with checkboxes.

### Frontend state

Small and DMI-logic-free:

```javascript
{
  observations: {layers: {start: 8, end: 15}, hooks: ["q", "k", "v"]},
  schedule: {stepStride: 4, requestStride: 1, capturePrefill: true, captureDecode: true},
  policy: "balanced"
}
```

### Validation UX

Visible but quiet. Header shows `● Valid` or `⚠ 2 issues`. Errors attach to the
control that caused them — "Router logits: unavailable for this model", not
`ValidationError: hook spec invalid...`.

## 12. Round-tripping

```text
parse(serialize(config)) == config
```

after canonical normalization. This is what makes the YAML a configuration
artifact rather than a one-way export.

## 13. Repository structure

```text
src/dmi/configuration/
    __init__.py
    schema.py          canonical typed representation
    manifest.py        load/validate model descriptors
    validation.py      layer ranges, hook availability, capabilities, schedule
    yaml.py            serialization only
    compiler.py        DMIConfig -> CompiledDMIConfig
    compatibility.py   to_legacy_hook_selection()
src/dmi/ui/
    app.py
    routes.py
    static/{index.html,app.js,architecture.js,styles.css}
```

Public API:

```python
load_config(path) -> DMIConfig
dump_config(config, path)
validate_config(config, model)
compile_config(config, model) -> CompiledDMIConfig
```

## 14. Implementation sequence

| Phase | Content | Gate |
|---|---|---|
| 1 | `DMIConfig`, `ModelManifest`, `LayerSelection`, `ObservationConfig`, `RuntimePolicy` + validation | schema frozen |
| 2 | YAML load/save/round-trip | usable config format with no browser |
| 3 | Compatibility — structured observations into existing hook selection | existing public API unchanged |
| 4 | Descriptor + catalog adapter with availability metadata | UI model |
| 5 | Static UI | no build process |
| 6 | FastAPI | UI validity == DMI validity |
| 7 | CLI | `dmi ui model.yaml` |
| 8 | Runtime policy | only once semantics are implemented |

The critical first milestone is **not the visual UI**. It is freezing the
`ModelManifest` + `DMIConfig` + YAML contract. Once those are correct the
architecture UI is a thin, replaceable layer over a stable configuration model.

## 15. Testing

- **Descriptor** — valid, invalid, unsupported architecture, missing topology.
- **Configuration** — valid, invalid layer selection, unknown hook,
  unavailable hook, invalid schedule.
- **Serialization** — config → YAML, YAML → config, round trip.
- **Integration** — descriptor → catalog → available hooks → configuration →
  legacy selector.
- **Golden files** — `tests/golden/qwen3-{basic,attention,moe}.yaml` compared
  against expected normalized output, to catch accidental changes to the
  configuration contract.

## 16. MVP definition of done

```text
✓ Load model descriptor            ✓ Validate configuration
✓ Render model architecture        ✓ Generate YAML
✓ Display model layer count        ✓ Reload YAML
✓ Select one or more layers        ✓ Preserve config through round-trip
✓ Select supported observations    ✓ Keep dmx_hook_selection working
✓ Disable unavailable observations ✓ Run from `dmi ui MODEL.yaml`
✓ Configure step frequency         ✓ No Node/React build
✓ Configure request frequency      ✓ Localhost-only by default
✓ Configure prefill/decode
```

## 17. Explicitly out of scope for the MVP

Grafana integration (stays downstream), runtime dashboard, payload estimator,
live model introspection, model editing, drag-and-drop model editing,
experiment management, authentication, multi-user server.

Model descriptor generation is **not** deferred: `dmi describe-model` ships,
and `dmi ui` reads framework configs directly.

---

## Appendix A — Verification against the codebase

Checked at `cb4e490`. The plan's architecture holds; these specifics do not.

### Confirmed

- **`CaptureSchedule`** — `src/dmi/config.py:10`. All eight fields exist
  exactly as assumed: `step_stride`, `step_offset`, `warmup_steps`,
  `capture_prefill`, `capture_decode`, `request_stride`, `request_offset`,
  `warmup_requests`, with validation in `__post_init__`. Reusable as-is.
- **Hook catalog** — `src/dmi/hooks/catalog.py:24`. 23 hooks, each carrying
  `(id, act_name, short_name, per_layer, group, tp_sharded, shape_class,
  pp_stage)`. Enough to drive UI grouping and layer applicability directly.
- **`HookSpec.layer_no`** — `src/dmi/hooks/specs.py:195`. Already present
  (`-1` for global hooks). Layer filtering is a pure function over an existing
  field; no changes to existing code.
- **Availability logic already exists** — `select_hook_specs`
  (`src/dmi/hooks/selection.py:112`) already suppresses `mlp_post` when
  `intermediate_dim == 0`, `router_logits` when `num_experts == 0`, and
  `topk_ids`/`topk_weights` when `top_k == 0`. The configurator's "unavailable
  + reason" surface should be derived from this, not reimplemented.
- **`pyyaml>=6.0.0` and `pydantic>=2.11.0` are already core dependencies.**
  Only `fastapi` and `uvicorn` are new, so the `[ui]` extra is small.

### Corrections

1. **`select_hook_specs` has a different signature than the plan assumes.** It
   is `select_hook_specs(specs, mode, cfg=None)` — it *filters a list of
   already-constructed specs*. It does not build specs from a model config.
   Specs come from `model.get_hook_specs()`. `compile_config()` therefore
   cannot run without a model-bound spec list, which means it needs a
   `model_context`, not just a descriptor.

2. **There is no CLI.** No `[project.scripts]` in `pyproject.toml`, no
   `argparse`, no `def main()` anywhere in `src/`. `dmi ui` and
   `dmi run --config` would both be the *first* DMI launcher, not additions to
   an existing one. The plan's caution about not introducing a second launcher
   resolves cleanly: there is no first one to conflict with.

3. **`dmx_hook_selection` is not a `src/dmi` API.** It appears only in
   `README.md` and `docs/vllm.md` as a vLLM `additional_config` key consumed by
   the vendored fork in `third_party/vllm-integration`. The real in-tree
   interface is `BackendAdapter.attach_model(model, hook_selection: str)` at
   `src/dmi/adapters/base.py:153`. Compatibility work targets that signature.

4. **`ModelShapeConfig` has no `num_layers`** (`src/dmi/hooks/specs.py:171`).
   DMI core never knows the layer count — it consumes a spec list the adapter
   builds by walking the model. `num_layers` is genuinely new information that
   only the configurator needs, for rendering and range validation. The
   descriptor is therefore *not* a serialized `ModelShapeConfig`.

5. **Naming diverges.** `ModelShapeConfig` uses `hidden_dim` /
   `intermediate_dim` / `num_heads`; the proposed descriptor uses HF naming
   (`hidden_size` / `intermediate_size` / `num_attention_heads`). One of the
   two has to win at the boundary.

6. **MoE hooks are `GROUP_OTHER`, not `GROUP_MLP`.** `router_logits`,
   `topk_ids`, and `topk_weights` are catalog group `GROUP_OTHER`, so a UI that
   groups strictly by catalog group will not place them under "MLP / MoE" as
   the layout assumes. The configurator needs its own presentation grouping
   layered over the catalog groups.

7. **`RuntimeConfig` overlaps `RingConfig`.** The existing runtime
   configuration surface (`docs/config.md`) is `RingConfig`, a native C++
   struct with ~15 transport-level parameters. The plan's caution about
   including runtime fields "only if genuinely part of the supported contract"
   applies directly.

8. **Test layout convention is flat.** Tests are `tests/test_*.py`, not nested
   packages, and are marker-gated (`cpu`, `gpu`, `e2e`, ...). New tests should
   be `tests/test_configuration_*.py` marked `cpu`.

9. **No YAML anywhere yet.** Nothing in `src/` or `tests/` imports `yaml`.
   This would be the codebase's first YAML configuration surface.

### Integration point

`BackendAdapter.attach_model` (`src/dmi/adapters/base.py:153`) is the single
chokepoint where configuration becomes runtime state:

```python
specs = model.get_hook_specs()
specs = apply_hook_selection(specs, hook_selection, cfg=cfg)
specs = filter_by_pp_rank(specs, self.is_pp_first(), self.is_pp_last())
specs = filter_by_tp_rank(specs, tp)
install_ring_hooks(specs, ring_payload=self.transport._ring_payload)
```

Layer filtering inserts as a fourth filter in this chain, immediately after
`apply_hook_selection`. Purely additive. The only signature change required
anywhere is giving `attach_model` a way to accept a `CompiledDMIConfig`
alongside the existing `hook_selection: str` default.

---

## Appendix B — Decisions taken

The five open questions were resolved as follows when phases 1-7 were built.

1. **Descriptor field naming — HF names.** Descriptors use `hidden_size`,
   `intermediate_size`, `num_attention_heads`, because they are authored from
   HF configs. The translation into `ModelShapeConfig`'s `hidden_dim` /
   `intermediate_dim` is isolated in `manifest.to_model_shape_config()`, the
   only place the two vocabularies meet.

2. **The descriptor carries a full topology block**, not just `num_layers`.
   The redundancy with `ModelShapeConfig` is worth it: the configurator runs
   with no model loaded, and a descriptor that only carried the layer count
   could not compute availability or head geometry on its own.

3. **`ModelContext` = spec list + `ModelShapeConfig`.** Forced by
   `select_hook_specs(specs, mode, cfg)`, which filters specs the model already
   produced. `compile_config` cannot run from a descriptor alone.

4. **`layers.end` is inclusive.** `LayerSelection(8, 15)` selects eight layers.
   The UI labels it "Layers 8-15" and the label must not lie. Tested directly,
   including that global hooks (`layer_no == -1`) are never swept up by a range.

5. **No `runtime` block in YAML v1.** The existing runtime surface is
   `RingConfig`, a native struct of transport parameters. Exposing a subset
   would imply a support contract that does not exist. `DMIConfig` has no
   `runtime` field.

### What landed

| Area | Module |
|---|---|
| Typed schema | `dmi/configuration/schema.py` |
| Descriptors | `dmi/configuration/manifest.py` |
| Catalog projection | `dmi/configuration/catalog_adapter.py` |
| Diagram metadata | `dmi/configuration/architecture.py` |
| Validation | `dmi/configuration/validation.py` |
| YAML | `dmi/configuration/yaml.py` |
| Legacy bridge | `dmi/configuration/compatibility.py` |
| Compilation | `dmi/configuration/compiler.py` |
| Layer filter | `dmi/hooks/selection.py` (`filter_by_layers`) |
| Backend | `dmi/ui/app.py`, `dmi/ui/server.py` |
| Front end | `dmi/ui/static/` (no build step) |
| Descriptor derivation | `dmi/configuration/introspect.py` |
| CLI | `dmi/cli.py` |

The only change to existing code is additive: `filter_by_layers` and
`hook_belongs_to_layers` in `dmi/hooks/selection.py`, following the same
convention as `filter_by_pp_rank` and `filter_by_tp_rank` (disable the dropped
spec's hook point, raise on unbound specs).

### Still open

* **Wiring into `attach_model`.** `compile_config` produces the filtered spec
  list, but `BackendAdapter.attach_model(model, hook_selection: str)` has not
  been given a way to accept a `CompiledDMIConfig`. Until it does, a layer range
  authored in the UI is not applied by an integration that calls `attach_model`
  directly. This is the one signature change the design calls for.
* **Runtime policy (phase 8).** `policy.objective` round-trips and is shown in
  the UI behind an explicit notice that it does not yet change behaviour.
* **Architecture coverage.** Only `decoder_transformer` is supported.
  Encoder-decoder configs are detected and refused rather than mis-rendered;
  vision and encoder-decoder layouts are future node tables.
