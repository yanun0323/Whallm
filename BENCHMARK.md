# Benchmark

Recorded historical measurements for the V4.1 prefill change, v1.1.7, v1.1.4,
and v1.1.0. No new full-model performance measurements are claimed for v1.1.8. Each section states its workload and measurement conditions.

## Historical PR measurements: DeepSeek V4.1 prefill on M5 Max

The following measurements were supplied by the author of [PR #11](https://github.com/yanun0323/Whallm/pull/11).
They describe the pre-merge implementation, not the merged `develop` runtime.
The merge retains request-scoped buffer cleanup and the DSpark/CED restriction
from `develop`; no full-model performance or memory measurement was rerun during
integration. The uncommitted source and truncated output hashes below limit
reproducibility; do not treat these rows as current release validation.

These rows were measured on 2026-09-17 from the source tree at commit
`7887a15` plus the uncommitted V4.1 prefill changes (region-major contiguous
expert layer buffers, pooled layer buffers, attention chunks with the MoE
batched over 4,096 tokens, and layer-major prefill with next-layer prefetch on
by default). The runtime ran from a Python 3.13.14 virtual environment with
MLX 0.32.2, not from the packaged App, through the same `run_trial` function as
the built-in **Throughput** benchmark.

Environment: MacBook Pro, Apple M5 Max, 128 GB unified memory, macOS 27.0,
internal SSD (about 13 GB/s for expert reads with the page cache bypassed).
Workload: **Code** context, output limit **128 tokens**, one run per row.
Configuration: 4,608 slots, LRU expert eviction, bf16 KV cache, 100 GiB memory
limit, 4 read workers, 2 prefetch read workers, prompt cache off, DSpark off.
Cache state: the expert slot cache is released at the start of every
layer-major prefill and prefill reads bypass the page cache, so prefill rows
do not depend on earlier runs; decode starts with an empty slot cache. Output
hashes are the first 16 hex digits of the SHA-256 over the generated token ids.

| Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) | Expert bytes read | Output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 27129 | 37.8 | 4.03 | 92.60 | 383 GB | 98b10a8ed5f8acf6 |
| 4096 | 34126 | 120.1 | 3.74 | 92.67 | 415 GB | bb2be29d9724a930 |
| 8192 | 51023 | 160.5 | 4.15 | 92.77 | 383 GB | 09a7a573f977090c |
| 16384 | 103425 | 158.4 | 4.09 | 92.97 | 383 GB | 104ced323450be0a |

Expert bytes read cover prefill and decode; every layer-major prefill reads the
complete 289 GB expert set once. Peak MLX memory is set by the 4,608 slots that
decode refills after each prefill.

Control rows from the same harness and machine with the v1.1.7 runtime
(token-major V4.1 prefill, the previous default), same configuration:

| Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) | Expert bytes read | Output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 53938 | 19.0 | 4.20 | 92.60 | 506 GB | f623db1be43cfd08 |
| 4096 | 88111 | 46.5 | 4.10 | 92.68 | 738 GB | 7fd55b255f3b160d |

For reference, the v1.1.7 App's Throughput benchmark on this Mac with the same
settings reported TTFT 51672 ms (19.8 tok/s), 98919 ms (41.4 tok/s),
219828 ms (37.3 tok/s), and 472847 ms (34.6 tok/s) for 1,024, 4,096, 8,192,
and 16,384 input tokens. Those App rows were not produced by this harness, so
their commit and cache state are not recorded here.

DSpark rows from the same harness and configuration with DSpark enabled
(768 DSpark slots, confidence threshold 0.6). Decode counts accepted draft
tokens as generated tokens, so it is not comparable with the rows above. The
control row uses the previous token-major prefill, which DSpark required before
this change:

| Prefill | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) | Expert bytes read | Output hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| layer-major | 1024 | 23186 | 44.2 | 3.89 | 93.58 | 382 GB | b1f87721338c7e39 |
| layer-major | 4096 | 33614 | 121.9 | 5.20 | 96.05 | 419 GB | 3da271deec0f6e61 |
| token-major (control) | 1024 | 44091 | 23.2 | 4.65 | 95.87 | 477 GB | de4d738a8356a3ff |

## Current Throughput sampling

Throughput fixes **`temperature=0` and `seed=42`** for every trial, regardless
of saved model temperature, Qwen adaptive temperature, or sampling values sent
to the benchmark endpoint. Chat and other generation endpoints are unchanged.
Other sampling parameters and model options still apply; fixed sampling does
not guarantee identical output across different acceleration paths.

Results report `temperature`, `seed`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, and `repetition_penalty`. JSON exports retain all of them;
text and Markdown exports also show Temperature and Seed. Missing fields from
older servers remain unknown, not retroactively labeled 0/42. Historical
measurements below have not been rerun and do not inherit this new policy.

## Current Throughput memory metric

Current source uses **Peak Memory** (GiB), not Peak MLX. Each trial samples
macOS physical footprint at a nominal 10 ms interval, plus start/end samples,
from before model loading through generation (excluding final unload). It takes
the maximum of each sample's Whallm + Python inference process total, not the sum
of their independent peaks. This is an absolute footprint, not growth since start.
A standalone server counts only its inference process, not its shell or API client.
Other helper processes and system-wide memory are outside this measurement.

The endpoint and JSON export use `peak_app_memory_bytes` and `memory_scope`
(`app`, `process`, or `simulated` for App Dry run); Throughput no longer exports
`peak_memory_bytes`. Failed reads make the metric unavailable: the endpoint sends
`null`, the App displays `—`, and its JSON export omits the unavailable value.
There is no RSS or MLX fallback. Sampling can miss brief peaks and is not guaranteed
to match Activity Monitor. No new full-model performance result is claimed here;
the historical MLX measurements below retain their original meaning.

**Status uses the same physical-footprint metric and process scope**, not RSS.
Its server-owned monitor samples nominally every **10 ms during model work**
(including loading, generation, warmup, queued work and unload) and every
**1 second while idle**. Status/health polling does not enable fast sampling.
The UI still refreshes roughly once per second, but receives the retained peak,
so a sampled peak is not lost between UI updates.

Status Maximum covers the time since server start or **Clear metric history**,
including idle time and model changes; it is not Throughput's independent trial
window. Clear also resets the server peak. A failed process read makes the
current sample unavailable and invalidates that window's maximum until Clear;
there is no RSS, MLX or zero fallback. Memory P95 retains the older, slower UI
history cadence (generation/completion snapshots, cleared on model changes),
not a percentile of the 10 ms stream. Different windows and sampling times mean
Status and Throughput maxima need not be identical.

## Qwen optimization correctness pilot (2026-09-17)

The following are the standalone runtime defaults and the historical pilot's
opt-in configuration, not the v1.1.8 App defaults. New or reset Qwen App settings
enable pooled index caching, optimized N-gram lookup, compiled tensor operations,
and phase memory; explicitly saved choices are preserved. MTP and its custom
strategy remain off by default. This default change is not evidence of a speedup.

| RuntimeConfig field | Default | Candidate scope |
| --- | --- | --- |
| `qwen_pooled_index_cache` | `false` | Incremental pooling/normalization/RoPE of complete QSA index blocks; raw history still retained |
| `qwen_ngram_lookup_optimized` | `false` | Deduplicate/sort requested rows, FP8 lookup-table decoding, avoid a redundant host copy |
| `qwen_compile_tensor_ops` | `false` | Compile pure grouped RMS normalization; no RNG or mutable-cache capture |
| `qwen_phase_memory` | `false` | Shrink the main expert cache for Prefill; restore capacity for Decode and request cleanup |
| `qwen_mtp_draft_tokens` | `5` | Draft depth from 1 through 5 |
| `qwen_mtp_zero_acceptance_limit` | `1` | Consecutive zero-acceptance rounds before request-local fallback, from 1 through 32 |

These fields are optional in runtime catalogs. At the time of the pilot, Qwen
model settings exposed independent experimental switches, all off by default,
applied on next load. v1.1.8 changes the App defaults as described above.
The custom MTP switch reveals depth and retry choices (initially 2/2); switching
it off uses 5/1 without erasing the choices. MTP must be installed and enabled
to edit that strategy. Model locking and two-confirmation reset still apply.

Phase memory is now connected to generation and warmup. For configured capacity
`S`, Prefill uses `min(S, max(512, S // 2))` slots for the installed Qwen model.
Small budgets therefore remain unchanged. Decode, completion, cancellation and
error cleanup restore `S`, without increasing the expert budget. Readers drain
and GPU work synchronizes before resizing; shrinking drops only slots outside
the smaller range, while growing preserves existing buffers. The released space
is available for temporary work, not a guarantee about total process memory or
speed. The main expert budget changes neither KV nor MTP/DSpark budgets.

On M5 Pro / 64 GiB, an 89-token prompt and 32-token greedy output matched the
non-MTP baseline exactly for each of the first three candidates and their
combination. The expert budget was 7.5 GiB, MLX memory limit 40 GiB, prompt cache
off, LFU eviction, 32 MTP slots, and App-style prefill controls off. These are
fixed pilot settings, not a copy of every App default. OS page cache was not purged.
This short correctness check establishes **no speedup** or long-context guarantee.

Default MTP diverged from non-MTP at output token 11. Both outputs reproduced
with the unchanged `879fc68` source, so this divergence was not introduced by
these candidates; its cause remains unresolved. Further full-model MTP-depth
and retry runs stopped at that gate. Unit tests are not a substitute for resolving it.

Reproduce a single fresh-process pilot using the tracked fixture:

```sh
PYTHONPATH=runtime:. .venv/bin/python Scripts/research_qwen_optimizations.py \
  --model /path/to/qwen3.8-flash-next.dsv4 \
  --prompt runtime/tests/fixtures/qwen_optimization_prompt.txt \
  --variant pooled --output scratch/qwen-pooled-pilot.json
```

A subsequent phase-memory pilot compared `baseline`, `phase`, and `phase-combined`
(the latter also enables the other three non-MTP candidates). The same short
32-token output matched exactly. At 7.5 GiB, capacity changed 3084 → 1542 → 3084
slots. With `--lifecycle`, baseline and combined runs also matched continuation,
cancellation-after-first-token, warmup and the next request, both with normal
Prefill and `--layer-major` batched Prefill. Every checked request boundary
restored 3084 slots. This is short greedy state validation, not a long-context,
stochastic, MTP-combination or performance result. The MTP stop above remains.

Run `baseline` separately and compare `generated_token_ids`. The runner records
source hashes, configuration, workload, environment, token hashes and sampled
process physical footprint; it does not change saved settings or model files.

## v1.1.7

These results were recorded with the built-in **Throughput** benchmark using
**Code** context, input lengths of **1,024, 4,096, 8,192, and 16,384 tokens**,
and an output limit of **128 tokens**. The same rows appear in the
[README benchmark tables](README.md#benchmarks).

Each row is an individual measurement, not a P95 statistic. Build revisions
and cache state were not recorded alongside these rows, so they are reference
results rather than a controlled comparison of acceleration settings.
The workload and output limit differ from the older API benchmarks below;
no cross-version improvement percentages are calculated for these runs.

TTFT is the time to the first token, in milliseconds. Prefill measures input
processing and Decode measures output generation, both in tokens per second.
Peak MLX is MLX allocation in **GiB**, not process RSS or total Mac memory.
That version's app export labels this value GB but divides bytes by 1024³.

### M5 Pro

| Model | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4 | 1152 | 1024 | 19112.3 | 53.6 | 7.7 | 22.77 |
| DeepSeek V4 | 1152 | 4096 | 24498.6 | 167.2 | 5.9 | 22.80 |
| DeepSeek V4 | 1152 | 8192 | 42137.5 | 194.4 | 6.8 | 22.83 |
| DeepSeek V4 | 1152 | 16384 | 81517.9 | 201.0 | 6.3 | 22.90 |
| Qwen3.8 | 3072 | 1024 | 10336.3 | 99.1 | 10.6 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 28831.0 | 142.1 | 9.2 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 53303.2 | 153.7 | 10.1 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 112353.1 | 145.8 | 8.5 | 17.18 |

### M2 Max

| Model | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

## v1.1.4

Run on a MacBook Pro with Apple M5 Pro, 64 GB of unified memory, and 1 TB of
storage.

The benchmark used mixed SPEED-Bench prompts through `/v1/completions`, three
runs per input size, and a 64-token output limit. The prompt cache was not
cleared between runs, but every run used a different input and reused zero
prompt tokens. The filesystem cache was not purged. With three runs,
nearest-rank P95 equals the maximum.

Percentages in parentheses show the improvement over v1.1.0. For total time,
TTFT, and memory, improvement is `(v1.1.0 − v1.1.4) / v1.1.0`. For prefill
and decode throughput, improvement is `(v1.1.4 − v1.1.0) / v1.1.0`. Positive
values are improvements and negative values are regressions. The v1.1.0
baseline used two runs per size, so these percentages are a descriptive release
comparison, not a paired causal measurement.

### DeepSeek V4 Flash 0731

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 27.10 s(+42.0%) | 17.75 s(+52.4%) | 59.9 tok/s(+188.7%) | 7.8 tok/s(+15.6%) | 32.84 GiB(−42.6%) |
| 2,048 | 27.35 s(−1.0%) | 17.60 s(−4.0%) | 117.3 tok/s(+8.5%) | 7.3 tok/s(+11.7%) | 33.26 GiB(−0.2%) |
| 8,192 | 50.12 s(−5.2%) | 40.47 s(−6.6%) | 206.3 tok/s(−1.5%) | 7.4 tok/s(+10.3%) | 34.50 GiB(+0.7%) |
| 16,384 | 88.27 s(−2.6%) | 78.61 s(−3.0%) | 209.0 tok/s(−1.5%) | 7.2 tok/s(+8.0%) | 35.70 GiB(−0.2%) |

### Qwen3.8 Next Flash FP8

v1.1.4 increases the slot count from 1,152 to 4,096 to improve the expert cache
hit rate, at the cost of higher memory use.

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 22.60 s(+7.6%) | 15.65 s(+9.0%) | 69.2 tok/s(+15.7%) | 10.4 tok/s(+11.6%) | 20.92 GiB(−37.7%) |
| 2,048 | 31.42 s(+22.3%) | 24.90 s(+23.7%) | 87.5 tok/s(+34.7%) | 9.8 tok/s(+18.1%) | 21.26 GiB(−29.6%) |
| 8,192 | 84.88 s(+40.4%) | 77.74 s(+42.2%) | 111.9 tok/s(+81.4%) | 10.4 tok/s(+25.0%) | 21.90 GiB(−22.3%) |
| 16,384 | 157.89 s(+42.9%) | 150.33 s(+43.9%) | 113.3 tok/s(+83.4%) | 9.7 tok/s(+23.9%) | 22.75 GiB(−20.3%) |

## v1.1.0 baseline

The v1.1.0 benchmark used the same MacBook Pro, endpoint, dataset files, input sizes, and
64-token output limit, with two runs per input size. With two runs,
nearest-rank P95 also equals the maximum.

### DeepSeek V4 Flash 0731

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 46.72 s | 37.26 s | 20.7 tok/s | 6.7 tok/s | 23.03 GiB |
| 2,048 | 27.08 s | 16.91 s | 108.1 tok/s | 6.6 tok/s | 33.19 GiB |
| 8,192 | 47.66 s | 37.96 s | 209.5 tok/s | 6.7 tok/s | 34.74 GiB |
| 16,384 | 86.01 s | 76.34 s | 212.3 tok/s | 6.7 tok/s | 35.64 GiB |

### Qwen3.8 Next Flash FP8

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 24.45 s | 17.20 s | 59.8 tok/s | 9.3 tok/s | 15.19 GiB |
| 2,048 | 40.45 s | 32.63 s | 65.0 tok/s | 8.3 tok/s | 16.41 GiB |
| 8,192 | 142.39 s | 134.44 s | 61.7 tok/s | 8.3 tok/s | 17.90 GiB |
| 16,384 | 276.41 s | 267.88 s | 61.8 tok/s | 7.8 tok/s | 18.92 GiB |

Performance changes with the prompt, SSD speed, cache state, and runtime
configuration. Memory is peak MLX active memory observed during each request,
not process RSS. These API benchmarks are exploratory measurements and do not
claim model-quality results.
