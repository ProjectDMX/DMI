# vLLM 0.27.1 DMI patch replay ledger

This ledger records how the DMI-only vLLM 0.25.1 patch stack was replayed onto
the immutable vLLM `v0.27.1` base
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. A clean Git replay is only a
syntactic result; the final disposition also records the semantic target-port
work and regressions.

| vLLM 0.25.1 DMI commit | vLLM 0.27.1 replay commit | disposition |
| --- | --- | --- |
| `5e5ea135635b` | `398aa9c45951` | replayed; hooked model/registry behavior re-audited |
| `8cd08cf3cc9f` | `f9712e693d44` | replayed; token-ID dtype retained |
| `e98c92551e4c` | `e47e5f9e1b5c` | replayed; reference variants re-imported |
| `fcf9f29ea2d8` | `df79f3922fac` | replayed; hook-selection behavior covered by focused tests |
| `22aef14eec73` | `010134c2cffc` | replayed; GPT-2 reference loader later adapted in `fdfe631884ae` |
| `6c24f51a7ad2` | `25011721e75c` | replayed; reference model paths re-imported |
| `a9ba67d1d838` | `70337a4629c8` | replayed; residual/hook semantics retained |
| `b88ba13ad264` | `cbad4996f10c` | replayed; MLP-post inventory retained |
| `4eb1673b3b8a` | `836e423138e1` | replayed; TP behavior requires target GPU matrix |
| `208490f17a18` | `c9fa9517a730` | replayed; Llama target source is unchanged |
| `12b857db3759` | `c2a706299bb1` | replayed; actual-token annotations retained |
| `57310abd3e85` | `4ae85e8ab707` | replayed; 0.27.1 `FusedMoEFactory`/runner router contract revalidated |
| `e1757a13ae26` | `f4f51db23e82` | replayed; canonical inventories revalidated |
| `a5fadb124e63` | `545eeadbad45` | replayed; reference source cleanup retained |
| `03eb163e6275` | `a7059c274795` | replayed; Qwen2 compile-dimension contract later adapted in `fdfe631884ae` |
| `2228df2b07eb` | `e5e6c78cdb0e` | replayed 0.25.1 port basis; target-specific model drift superseded by `fdfe631884ae` |
| `6f1fce945c54` | `de52ce06a728` | replayed; packed Llama loader remains semantically unchanged |
| target-only | `fdfe631884ae` | adapted GPT-2 `AutoWeightsLoader`, Qwen2 dynamic positions, and Qwen3 per-layer sliding window for 0.27.1 |

The resulting versioned integration branch is `dmi-v0.27.1`. No tag is cut
until the final support matrix and exact root gitlink have been approved.
