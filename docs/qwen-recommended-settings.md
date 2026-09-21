# Qwen Flash Next: theoretical starting settings

These are **experimental starting points, not measured optimal presets**. They
assume one active request, Whallm's pinned MXFP4 installed model, a local SSD,
and an otherwise mostly idle Mac. No complete checkpoint was run for this work.
The 96 GB profile is a planning scenario, not a claim that a device was tested.
The App does not automatically apply these profiles or change saved preferences.

[Machine-readable runtime fragments](qwen-recommended-settings.json) contain
only RuntimeConfig keys. They are not a complete server catalog: select a profile's
`runtime` object, not the whole document, when constructing a catalog. Tests
check parameter types/dependencies and budget arithmetic, not model fit or speed.
The new controls and source audit are described in [the implementation note](qwen-next-throughput.md).

## Starting point for 8K-32K conversations

| App setting | M2 Max, 64 GB | M5 Pro, 64 GB | M5 Ultra, 96 GB planning |
| --- | ---: | ---: | ---: |
| Expert cache GiB | 24 | 24 | 40 |
| Memory limit GiB (MLX, not process RSS) | 48 | 48 | 72 |
| Read workers | 8 | 16 | 16 |
| Prefetch read workers | 2 | 2 | 2 |
| Prefill step size | 512 | 1024 | 1024 |
| QSA queries per chunk | 16 | 16 | 16 |
| Use fused QSA SDPA | on, experimental | on, experimental | on, experimental |
| Use indexed QSA decode | on, experimental | on, experimental | on, experimental |
| Prompt cache | memory | memory | memory |
| Prompt cache entries / GiB | 1 / 4 | 1 / 4 | 1 / 8 |

The identical 64 GB expert budgets are intentional: GPU generation alone does
not create more RAM. Worker counts and chunk sizes are hypotheses to A/B, not
chip-specific measured tuning. Start at the stated budget and decrease it by
4-8 GiB when running other memory-heavy applications. Only try 28-32 GiB on a
64 GB dedicated server after measuring the complete workload with headroom.
Do not set an expert budget equal to the MLX memory limit.

Use the following common settings: layer-major prefill **on**, whole-layer
expert batching **on**, next-layer prefetch **on**, grouped expert prefill **on**,
ready-expert decode **on**, LRU eviction, pooled QSA keys **on**, optimized N-gram
lookup **on**, compiled Qwen tensor operations **on**, phase memory **on**.
Keep expert waves **0**, N-gram backend **mmap**, N-gram row-cache payload **0 MiB**,
KV and index compression **off**, ANE prefill share **0%**, and power-saving read
throttling **off**. These separate exact scheduling changes from additional
quantization and accelerator experiments. mmap and cached expert reads do not
mean the entire N-gram table is resident.

## Adjust the two different chunk controls independently

`Prefill step size` controls model chunks; `QSA queries per chunk` controls the
inner attention loop. They are not interchangeable with expert-wave slots.
A larger QSA chunk never increases the model's selected attention budget.

For contexts at or below about 8K, try **4 QSA queries**: the initial and source-recheck
virtual runner trials showed 16/32 slower at 8K. At about 32K, try **32 queries**:
those trials showed better component p50 than 4, but higher additional allocations.
Use 16 as an intermediate starting experiment, not a universally best value.
At 64K and above, sweep 4/16/32 with the same memory and output-quality gates.
The extended benchmark covers narrow (1/3-query) calls at 128K, but not large
prefill batches or full-model memory at that context length.
Keep model prefill at 512 on the M2 starting profile; test 256/512/1024 on the
same prompt before increasing it. No benefit from 4096 is assumed.

The runtime reduces requested QSA chunks according to a 256 MiB *per-chunk
workspace estimate*. It is **not** a 256 MiB peak-allocation guarantee. Pending
lazy graph chunks, token-major cache copies, SDPA scratch and retained outputs
can coexist. The source recheck's 128-query/32K case used about **1100 MiB additional
MLX allocation** with chunk 32. Do not size a model from the per-chunk estimate
alone. The App's memory overview is also an uncalibrated structural estimate.

## Why the expert cache starts here

The installed model uses 2,611,200 bytes per routed expert blob, 48 layers and
512 experts/layer; the manifest remains authoritative. Thus:

```
slots = floor(expert_cache_GiB * 1073741824 / 2611200)
24 GiB -> 9868 slots
40 GiB -> 16448 slots
all routed experts -> 24576 blobs -> about 59.77 GiB
```

More cache can reduce demand reads only when it retains experts that will be
reused. It is not a proportional speed multiplier. Common tensors, KV/index
history, recurrent states, prompt snapshots, activations, allocator buffers,
CPU allocations, OS pages and other applications still require memory. The
48/72 GiB MLX limits are starting ceilings; wiring remains bounded by the actual
device's reported recommended working set. They are not whole-process limits.
The profile does not change privileged macOS wired-limit settings.

## MTP is a second experiment, not the initial baseline

First measure with **MTP off**. Existing full-model MTP versus non-MTP output
behavior has not been newly qualified by these tests. For a separate coding
trial with the installed native MTP sidecar, enable MTP and its custom strategy:
**2 draft tokens**, **2 consecutive zero-acceptance rounds**, and **0.5 GiB MTP
expert cache** on the 64 GB profiles. On the 96 GB planning profile, try **1.3 GiB**
(large enough in payload terms for the 512-expert MTP layer). Main-model expert
cache and memory limits are unchanged. The App's inactive auxiliary budget is
not counted until MTP is enabled.

Test draft depth 3 only after depth 2 improves accepted output tokens/second at
the same quality. Additional draft/verify reads can outweigh acceptance gains.
The App currently disables ordinary prompt-cache reuse for MTP; repeated-turn
TTFT can therefore worsen even when one long answer decodes faster. Recheck both.
Do not set fewer generated tokens, change sampling, or skip reasoning to report
an engine speed improvement.

## Rollback and promotion gates

Record the exact commit, chip, physical RAM, model/quantization, total context,
prompt/cache state, configuration, input/output hashes, TTFT, prefill/decode
rates, p50/p95 over repeated requests, peak process/MLX memory, memory pressure
and swap. Compare prose, code and Traditional Chinese tasks. Component timing
percentiles are not end-user latency percentiles.

If output-quality or numerical checks fail, turn off indexed QSA first; then
restore query chunk 4 and fused SDPA off. Keep compression and MTP off while
isolating the cause. Stop on nonfinite logits, failed cancellation/continuation,
unexpected cache changes or sustained swap. No physical-device throughput or
quality improvement is guaranteed by this document.
