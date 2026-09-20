# Qwen3.8 Flash Next optimization source audit and experiments

Status: opt-in research implementation. No production default or installed-model
format changed. Host tests and synthetic I/O measurements are not Apple GPU,
model-quality, or token-throughput validation. Qwen3.8-Flash-Next is Whallm's
pinned `qwen4_exp_text` architecture, not Qwen3-8B or the dense Qwen 27B model.

## Reproducible source baseline

All four repositories were fetched by Git at the revisions below. The source
acquisition workflow verifies each detached HEAD and records a SHA-256 for its
source archive. The feature branch starts at the stated Whallm master revision.

| Repository | Branch | Audited commit |
| --- | --- | --- |
| Whallm | master | `6d88d72fafc62d75193a3290cbd14c782af23fd0` |
| npanj/llama.cpp | main | `f507d75f8dfeb90ad96c3e6a08129dbb1999d17a` |
| derekrparris/DynaMoE | main | `b396e115f396790b4269d5118eb1b0bf22581d1e` |
| mihailescu2m/llama.cpp | master | `d1762fce3e88ab78d99639b3cd8fa2e108f74f53` |

## What the reference projects actually implement

### npanj/llama.cpp

Primary sources: [README](https://github.com/npanj/llama.cpp/blob/f507d75f8dfeb90ad96c3e6a08129dbb1999d17a/README.md),
[streaming implementation](https://github.com/npanj/llama.cpp/blob/f507d75f8dfeb90ad96c3e6a08129dbb1999d17a/src/llama-moe-stream.cpp),
[Qwen graph](https://github.com/npanj/llama.cpp/blob/f507d75f8dfeb90ad96c3e6a08129dbb1999d17a/src/models/qwen4exp.cpp),
[Qwen research log](https://github.com/npanj/llama.cpp/blob/f507d75f8dfeb90ad96c3e6a08129dbb1999d17a/docs/Qwen3.8-Flash-Next.md).

- Bounded routed-expert slots, parallel slab reads, hotness/decay eviction,
  demand/prediction queue separation, and shared Metal buffers. This is a fork
  of mihailescu2m, not an independent invention of the underlying streamer.
- Pair-partitioned prefill: a routed `(token, expert)` pair belongs to one wave.
  Its planner additionally balances skewed expert groups with a snake ordering.
  This avoids computing every pair in every masked wave.
- PLE row streaming (`read_ple_rows`): validate, deduplicate, sort rows by file
  offset, schedule reads, then restore duplicate positions. Readers are drained
  before the destination is released, including error paths.
- QSA block-level selection instead of a context-by-query cell table, gathered
  sparse attention, and F16 indexer keys even when attention KV is quantized.
  Indexer precision matters because discrete top-k membership can change.
- Model-shape-specific graph/kernel improvements: four-stream hyper-connection
  kernels, fused routing/reduction, SSM convolution plus SiLU, norm/scale fusion,
  and avoiding an unnecessary F32 rescale in batched expert GEMM.
- Native MTP head support, bounded draft resources, and preserving batch ordering
  so the head receives the intended hidden states.
- Crucial negative evidence: the README distinguishes random-token n-gram table
  page faults from real prompts. Its reported direct-read gain on a real 11k-token
  prompt was within run noise. Upstream PR author numbers are explicitly not all
  measurements reproduced by this fork. None are Whallm results.

### derekrparris/DynaMoE

Primary sources: [README](https://github.com/derekrparris/DynaMoE/blob/b396e115f396790b4269d5118eb1b0bf22581d1e/README.md),
[ExpertIOThreadPool](https://github.com/derekrparris/DynaMoE/blob/b396e115f396790b4269d5118eb1b0bf22581d1e/DynaMoE/DynaMoE/ExpertIOThreadPool.swift),
[working-set integration](https://github.com/derekrparris/DynaMoE/blob/b396e115f396790b4269d5118eb1b0bf22581d1e/DynaMoE/DynaMoE/ContentView.swift),
[JetSpec](https://github.com/derekrparris/DynaMoE/blob/b396e115f396790b4269d5118eb1b0bf22581d1e/core/src/jetspec.rs).

- Packed per-layer expert files and cached read-only descriptors. The inspected
  pool uses `DispatchQueue.concurrentPerform` with POSIX `pread` into supplied
  destinations. Its `initialize(numThreads:)` currently resets descriptors; it
  does not itself create a fixed-size eight-worker scheduler. Do not equate the
  README's nominal thread count with an enforced concurrency limit.
- Direct safetensors-shard access is another path, avoiding mandatory duplicate
  repack storage. Shared Metal buffers and mmap reduce extra host/GPU copies;
  this does not mean storage reads bypass every OS copy or page cache.
- Working-set management at prefill-layer and generation-token boundaries;
  separate accounting for dirty process footprint and clean mapped pages.
- Prefix tracking and delta prefill, plus Qwen architectural support for PLE,
  hyper-connections, recurrent layers, and sparse attention. Architecture support
  is not itself proof of a performance improvement.
- JetSpec contains tree masks, greedy/sampling verification and expert-budget
  pruning of low-confidence leaves. These require a tree-aware target verifier,
  not simply enabling Whallm's existing linear MTP verifier.
- Validation limit: its README says the author lacked disk space to fully repack
  Qwen and test its streaming optimizations. General advertised speedups must
  not be presented as validated Qwen streaming or Whallm results.

### mihailescu2m/llama.cpp

Primary sources: [README](https://github.com/mihailescu2m/llama.cpp/blob/d1762fce3e88ab78d99639b3cd8fa2e108f74f53/README.md),
[streamer](https://github.com/mihailescu2m/llama.cpp/blob/d1762fce3e88ab78d99639b3cd8fa2e108f74f53/src/llama-moe-stream.cpp),
[Qwen graph](https://github.com/mihailescu2m/llama.cpp/blob/d1762fce3e88ab78d99639b3cd8fa2e108f74f53/src/models/qwen4exp.cpp),
[speculative decoding](https://github.com/mihailescu2m/llama.cpp/blob/d1762fce3e88ab78d99639b3cd8fa2e108f74f53/common/speculative.cpp).

- Core expert SSD streaming, hotness eviction, exact pair waves and router-based
  lookahead. A prediction may populate a cache; it must not replace actual routing.
- Phase-aware ubatches/workspace and cache resizing/warming: memory released
  after prefill can become decode expert capacity. This changes the memory budget
  allocation, so comparisons must keep the same overall memory ceiling.
- Used-expert/tile GEMM dispatch, small speculative verification matvec kernels,
  MXFP4 byte lookup improvements, and fast contiguous recurrent-state copies.
- PLE row prefetch advisories, QSA block selection, and opt-in gathered/union QSA
  attention. Preserve each query's membership and causality; changed reduction
  order is not automatically bit-identical.
- Native MTP, optional probability-ratio rejection sampling, history n-gram
  suffixes and bounded drafting. The README warns that the n-gram suffix can
  change greedy text; it is not a free exact acceleration to enable blindly.
- Persistent slots and SSD context checkpoints reduce repeated conversation
  prefill. Its M1 Max benchmark setup and best-of-two figures are external
  measurements, not a Whallm baseline.

## Existing Whallm functionality versus new work

The master baseline already has SSD expert slots, separate prefill I/O, LFU/LRU
and route-aware eviction, whole-layer buffers, grouped expert GEMM, PLE mmap,
opt-in row dedup/FP8 lookup-table decode, block-bounded QSA, pooled index caches,
phase-memory experiments, prompt-state persistence, MTP checkpoint/rollback,
probability-ratio rejection sampling with corrected residuals, and request-local
zero-acceptance fallback. Relevant modules are
`expert_cache.py`, `qwen4_exp.py`, `qwen_ngram_lookup.py`,
`qwen_pooled_cache.py`, `qwen_phase_budget.py`, and `qwen_mtp_policy.py`.
These are not counted as newly implemented features in this branch.

New work is a clean Python/MLX implementation of selected ideas. No Swift/Rust
or ggml/Metal kernel bodies were copied. DynaMoE carries Apache-2.0; llama.cpp
carries MIT. A future code transplant must retain its applicable notices.

### 1. Capacity-bounded exact expert-pair waves

`qwen_expert_waves.py` groups the original route pairs in stable expert order.
`StreamingExperts._waves` loads at most `min(qwen_expert_wave_slots, cache.slots)`
unique experts per wave, computes each complete expert's token group once and
restores the original routing order before weighted reduction. The per-expert
GEMM shapes match the existing individual-expert baseline. It does not prune
routes, renormalize probabilities, or convert quantization.

Every wave is synchronously evaluated before slots may be evicted/reused. A
cancellation/error path drains GPU work. This lifetime fence is required with
MLX lazy evaluation and externally writable shared slots.

The layer-major support path disables full-layer batching when waves are selected;
otherwise it would allocate/read all experts and defeat the experiment. Whole-layer
next-layer prefetch is rejected in combination. Existing single-token ready-decode
behavior remains unchanged. Wave counts/pair counts/peak experts are recorded by
the pilot. The setting bounds experts acquired per wave, **not total process RAM**;
existing cache-slot, activation and returned-output allocations still exist.

This implementation does not claim npanj's snake balancing, GPU-side routing,
lookahead overlap or its measured prefill speed. Smaller waves add synchronization
and CPU scheduling and may be slower when a full expert layer fits comfortably.

### 2. Positioned N-gram reads and a bounded packed-row LRU

`qwen_flash_io.py` validates the file contract, retains one read-only descriptor,
deduplicates/sorts requested rows, coalesces adjacent misses up to 256 KiB,
completes short/interrupted reads and returns rows in the original shape/order.
The LRU retains packed FP8 bytes, not dequantized float rows. A non-multiple budget
is rounded down to complete rows; zero disables retention. Cancellation, EOF,
threaded reads, invalid IDs, and idempotent close are tested. Normal model unload
and failed model load release the reader.

The configured ceiling counts retained packed payload only. Python dictionaries,
row-object metadata, output arrays, transient buffers and OS page cache are extra.
The maximum selectable payload is 512 MiB. Reads use buffered POSIX `pread`, not
`O_DIRECT`, F_NOCACHE, mmap-to-Metal zero-copy, or a new NVMe DMA API. `bytes_read`
counts bytes returned by pread, **not physical SSD traffic**. File contents must
remain immutable while a model is loaded, as with the baseline mmap contract.

### 3. Fused SDPA over the existing QSA selected cells

An opt-in path feeds the **same** gathered K/V rows, per-query causal mask and GQA
head layout to `mx.fast.scaled_dot_product_attention`. It avoids the explicit
attention score/softmax/value sequence in Python/MLX while leaving the indexer and
block selection unchanged. The existing implementation remains the default.

The fused kernel has different reduction/rounding behavior and is not advertised
as bit-identical or greedy-token-identical. Synthetic tests cover head dimensions
32 and 256, FP32/BF16, short/dense and sparse contexts, incomplete block tails,
future-token masking, chunked prefill, decode and cache rollback. Whole-model
quality and stochastic sampling still require validation on the installed model.

## Configuration and reproducible commands

| RuntimeConfig / CLI option | Default | Accepted values |
| --- | --- | --- |
| `qwen_expert_wave_slots` / `--qwen-expert-wave-slots` | `0` | 0 disables; 1..512 |
| `qwen_ngram_io` / `--qwen-ngram-io` | `mmap` | mmap, pread |
| `qwen_ngram_cache_bytes` / `--qwen-ngram-cache-bytes` | `0` | 0..536870912; requires pread |
| `qwen_sparse_sdpa` / `--qwen-sparse-sdpa` | false | boolean; also `--no-qwen-sparse-sdpa` |

Both standalone CLI and server accept the switches. Catalog fields are optional
for backward compatibility. Nondefault settings are rejected for non-Qwen models.
The Qwen model's [App controls](qwen-flash-app-controls.md) expose these experiments.
They are saved per model; packaged defaults remain unchanged.

```sh
# Host tests need only Python and NumPy; no MLX import or mocked GPU.
make test-qwen-flash-portable PYTHON=python3

# Apple Silicon, after installing the existing pinned runtime requirements:
make test-qwen-flash

# Synthetic row I/O only. Do not compare this timing with tokens/second.
make benchmark-qwen-flash-host PYTHON=python3 \
  QWEN_FLASH_OUTPUT=scratch/qwen-flash/host.json

# Each pilot uses a fresh process, refuses to overwrite evidence, and records
# prompt/output hashes, config, source hashes, cache state and memory/IO metrics.
# Set QWEN_MODEL to an existing installed Whallm Qwen model, not raw HF shards.
for variant in baseline waves pread sparse-sdpa flash-combined; do
  make pilot-qwen-flash QWEN_MODEL="$HOME/.dsmodel/YOUR-QWEN-MODEL" \
    PROMPT=your-prompt.txt QWEN_VARIANT="$variant" \
    QWEN_FLASH_OUTPUT="scratch/qwen-flash/$variant.json" \
    QWEN_FLASH_ARGS='--lifecycle --layer-major'
done
```

## Validation evidence and promotion gates

[Host synthetic evidence](validation/qwen-flash-host-20260921.json) records exact
source hashes and a fixed random seed. All six trace/backend combinations produced
identical output bytes. In the reused-row trace, caching reduced pread calls from
3,928 to 193 and returned bytes from 1,031,680 to 40,960. Nevertheless, the mmap
baseline was faster than either pread variant in this warm Linux page-cache test.
The uniform-row trace also exposed cache bookkeeping cost with little reuse.
These are deliberately preserved negative results, not Apple SSD performance claims.

The portable suite has 18 tests with additional subcases and seeded schedules.
GitHub Actions run `35531099283` passed the portable job and the macOS arm64 job
(macOS 15.7.9, MLX 0.32.2, Metal available). The Qwen suite reported 111 tests,
107 passing and 4 skipped because pre-existing local archive evidence was
not present. All 11 new MLX tests and all 18 portable tests passed. This includes
MXFP4 wave parity/lifetime fencing, FP8 row-decode bit parity and QSA numerical,
causality, chunking and rollback checks. It does not load the full model.
The first CI attempt found missing explicit cache arguments in the new QSA tests;
those calls were corrected and the suite rerun, not suppressed. See
[CI evidence](validation/qwen-flash-ci-20260921.json). The source audit itself
successfully acquired all four pinned repositories.

Before promoting an option: run the same installed model/quantization and MTP
settings, representative prose/code/multilingual prompts at short, 8k, 32k and
long context, fresh/warm conversation states and multiple repetitions. Record
commit, chip, RAM, MLX version, input/output token hashes, TTFT, prefill/decode
rates, memory peak, cache hits, pread-returned bytes, and independent physical
storage counters where available. Report p50/p95 and variance, not a best run.

Stop on wave/row bit mismatches, invalid or nonfinite logits, stale state after
rollback/cancellation, memory-budget violations, or failed model-quality gates.
For SDPA, tolerance-based tensor tests are only an initial check, not permission
to change quality. No throughput improvement is claimed for this branch yet.

## Deferred rather than mislabeled as completed

- JetSpec tree speculation: needs branch-isolated recurrent/conv/PLE states,
  tree-causal QSA, acceptance/rollback proofs and a compatible drafter. A tree
  mask on the present linear MTP path is insufficient.
- History n-gram suffixes and changes to MTP proposal construction: require
  correct proposal probabilities, sampling transforms and checkpoint alignment.
  Whallm already uses probability-ratio acceptance and corrected residual rejection
  sampling through `dspark._verify`; it is not newly added or replaced here.
  Existing MTP verification and fallback remain untouched.
- ggml-specific fused HC/MXFP4/used-tile kernels: cannot be pasted into MLX. New
  custom kernels need numerical, dispatch, concurrency and M-series profiling.
- Router lookahead and more prefetch depth: need separate bounded prediction
  resources, demand priority and measured precision/recall. Whole-layer prefetch
  is not equivalent to expert-level lookahead.
- Checkpoint-format or quantization changes, indexer quantization, and automatic
  large-memory defaults: change quality/memory contracts and are not silently
  enabled from external 64 GB benchmark configurations.
