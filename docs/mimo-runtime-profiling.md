# MiMo layer diagnostics

The MiMo runtime changes are included in the 2026-09-23 local App build, not a
public release. The diagnostic command below remains a checkout tool; its hooks
are opt-in and never attached by the server or model loader. It reuses the
existing profile collector with a MiMo-specific attachment; no new dependencies
are needed.

## Run

Use a complete installed MiMo model, one process at a time:

```sh
PYTHONPATH=runtime:. .venv/bin/python Scripts/profile_mimo_runtime.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --output scratch/mimo-baseline --mode baseline

PYTHONPATH=runtime:. .venv/bin/python Scripts/profile_mimo_runtime.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --output scratch/mimo-host --mode host --timeline

PYTHONPATH=runtime:. .venv/bin/python Scripts/profile_mimo_runtime.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --output scratch/mimo-sync --mode sync
```

Output paths must not exist. Defaults: Code 1024 input tokens, 32 output tokens,
seed42, temperature0, 771 slots /9.6GiB, four readers, LRU, bounded layer-major
Prefill enabled, Prompt Cache and speculation off. These are explicit diagnostic
settings, **not a change to App defaults**. Use `--no-layer-major` for the original
chunked path, `--context novel`, `--max-tokens 128`, or `--config FILE.json` for
RuntimeConfig overrides. When changing legacy slots in a JSON override, also set
`expert_cache_bytes` to null so the byte budget does not override the slot count.

The runner stops at900s,40GiB sampled footprint, or256MiB growth in system swap
usage by default. It does not purge OS caches. Later requests in `--requests N`
retain expert slots; compare matching request indices.

## Outputs and interpretation

- `result.json`: effective settings, environment, source/catalog/manifest hashes,
  corpus/input/output hashes, output tokens, request diagnostics, and profile rows.
- `layers.csv` (host/sync): 48 backbone layers per phase/request, attention type,
  wall/owner CPU, expert hits/misses, logical bytes, read/wait/pack/eviction time,
  and nested Attention/QKV/SDPA/output projection/KV/router/routed-expert spans.
- `samples.jsonl`: process footprint, CPU/fault counters, VM/swap and device-wide
  GPU utilization. These are not per-layer memory ownership or GPU occupancy.
- `source.tar.gz`, `input_tokens.json`: source snapshot and exact input IDs.
- `timeline.jsonl` with `--timeline`: Mach-clock phase/decoder intervals, not a
  Metal kernel trace.

Backbone IDs are **0–47**. Layer0 is dense; expert-cache ID is backbone ID minus1.
Global and sliding attention are labeled separately. Miss fractions count unique
expert lookups per layer union, not all token/expert assignments during Prefill.
First yielded token splits
Prefill/Decode; pipelined work may straddle that boundary.

**Use baseline mode for performance comparisons.** Host keeps lazy evaluation:
router materialization can be charged to routed experts, and later cache eval
can carry earlier GPU work. Sync forces evaluation and barriers, disturbing
normal overlap; it measures CPU/I/O/GPU-wait scopes, **not GPU kernel time**.
Inclusive child spans/counters are nested and must not be added to their parent.
CPU and wall snapshots have slightly different boundaries, which matters for
small spans. Hooks are owner-thread-only; use a dedicated process because some
class/function hooks are process-global. Hooks are restored on cleanup.

## Findings and optimization (2026-09-23)

M2 Max/64GiB, Code1K/32, bounded Prefill, four readers/771 slots:
normal Decode14.27s included11.40s waiting for expert futures (~80%). In the host
observer, routed experts including I/O took12.90s; expert wait was10.15s.
Layers3/6 missed97.6%/97.2% of selected expert lookups. Other early MoE layers
were also costly; this was not one broken layer. Dense layer0 took about0.020s
across31 observed Decode calls. Attention is not the leading bottleneck in this
workload. Sync Attention3.47s/router0.48s are perturbed scope times, not amounts
that can simply be subtracted from normal inference.

Two setting candidates were not adopted: eight readers did not reduce bytes or
show reliable speed improvement; LFU reduced Decode misses by only0.45% in the
screen. Timing drift affected CPU, packing, and unchanged Decode paths, not only
expert I/O. A native sample showed MLX evaluation/Metal waits and dispatch, but
neither that sample nor system telemetry established a thermal, App Nap, or
specific driver cause. Temporary `caffeinate -di` did not eliminate the drift.

A deterministic work reduction **was implemented**: the last layer of each
bounded text-Prefill block now populates only KV state. Its discarded SDPA/output
projection and MoE are skipped; fused QKV, K-RoPE and value scaling remain. The
normal forward/Decode and media paths are unchanged. It is active only when the
existing layer-major path is used; the App default remains off.

Complete-model checks retained exact output tokens:

| Workload | Before Prefill logical GiB | After | Saved |
| --- | ---: | ---: | ---: |
| Code1K/128 | 140.200 | 138.021 | 2.179GiB |
| Novel1K/32 | 111.102 | 109.408 | 1.693GiB |

Four unused final-layer expert calls and8184 routed assignments disappear for
these1023-token Prefills. Code Decode bytes remain identical; Novel Decode has
one extra miss (12.75MiB) due to changed cache residency. No swap was observed.
**This is verified less work, not a verified wall-time speedup:** Code total
95.77→373.96s was dominated by severe drift; Novel159.29→159.07s was effectively
unchanged. Neither result justifies a default change or a claimed speedup.
Python634 tests pass, including exact GA/SWA cache state, continuation, block
boundaries, cancellation and profiler restoration. Full-model long-context and
media performance have not been validated.

The main opportunity remains expert I/O/cache behavior. Potential next work is
overlapping Prefill's required expert reads with computation rather than waiting
for the entire union, and controlled cache-capacity comparisons accounting for
OS-cache displacement. These are hypotheses, not implemented or measured gains.
A stable timing environment is needed before accepting further speed defaults.
