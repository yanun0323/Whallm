# Qwen whole-model throughput review and SSD handoff experiments

Baseline: `c71231229f29c79160f51aba2b02d4aa1380b910`, on
`feat/qwen-flash-optimize`. The reported lack of visible end-to-end improvement
is a user observation, not a reproduced measurement here. This review follows
the benchmark entry, effective catalog, generation, layer-major prefill, expert
cache/reader, Qwen routed/shared experts, attention, MTP and App settings paths.
It is not a claim that every repository file or the full checkpoint was tested.

## Review findings

| Finding in baseline code | Consequence | Action |
| --- | --- | --- |
| `throughput.run_trial` disables ordinary, persistent and DSpark prompt cache | A conversation-cache win cannot improve this cold-prompt benchmark. Other runtime settings are retained. | Keep the benchmark definition, expose effective settings. |
| The App starts the benchmark without waiting for pending model configuration requests | An immediate click after editing can race configuration. This has not been established as the cause of the user's result. | Await configuration completion and propagate failures instead of silently using a stale configuration. |
| Whole-layer prefill calls `release_prefill_slots` and reads all routed expert layers into recyclable layer buffers | Existing decode residents are cleared, and the newly read hot experts are not adopted into decode slots. | Opt-in prompt-tail warm handoff with fixed capacity and no demand-resident eviction. |
| `_read_expert_ids` issues one `_pread_views` per expert despite contiguous whole-layer ranges | Per-call/alignment overhead remains even when requested data is adjacent. The existing 2.49 MiB blob is already large, so benefit is not guaranteed. | Merge consecutive experts without reading gaps; keep batch one as default. |
| Shared expert tensors are lazy and normally submitted with the final routed reduction | Waiting for routed expert reads can leave independent resident work unsubmitted. | Opt-in singleton shared/router async submission. |
| Previous singleton dispatch optimization bypasses sorting only on the non-ready path | It does not improve the existing default ready-expert path. | Do not credit its microbenchmark percentage to normal Decode. |
| Existing QSA optimizations touch attention, not expert SSD transfers or all layers | A large isolated attention speedup may have small total impact. | Keep those results component-scoped; collect request evidence. |
| `read_seconds` is a service/queue interval and can overlap GPU work | It cannot be subtracted from total time as if it were pure blocking. | Add a separate consumer `wait_seconds` counter. |

Existing layer-major scheduling, whole-layer next-layer prefetch, LFU/LRU/route
policies, ready-expert reads, MXFP4 kernels, prompt cache and MTP are not counted
as new features. Routing predictions, lower top-k, approximate expert selection,
new quantization and different sampling are not used to manufacture higher scores.

The default whole-layer Qwen path requests 48 * 512 * 2,611,200 bytes, about
59.77 GiB of logical expert payload per complete layer-major prompt pass.
Increasing a model token chunk does not divide these bytes again: Whallm already
loads a layer once across its chunks. This is different from runtimes that reload
weights for each prompt chunk. Physical SSD traffic depends on the reader and OS.

For example, if attention accounts for only 10% of runtime, doubling attention
speed gives 1 / (0.9 + 0.1 / 2), about 1.053x overall. This is an illustrative
Amdahl calculation, not the user's measured breakdown.

## External research and selected ideas

Only primary sources are used. Source inspection is not a reproduction of their
benchmarks. No third-party kernel bodies or model weights are copied.

- [streamlx at 146507b59293058b3a1cecc43f47947b25012297](https://github.com/srcterm/streamlx/tree/146507b59293058b3a1cecc43f47947b25012297):
  `streamlx/integrate.py` expert sweep adopts hot experts and evaluates layer and
  pool tensors before recycling temporary storage; `reader.py` merges spans and
  resident work uses `async_eval` while misses arrive. Whallm adopts these concepts,
  not its model wrapper or its blanket bit-identity/performance claims. Its tested
  models and memory layout differ from Whallm's pinned Qwen3.8 installation.
- [LLM in a flash, arXiv:2312.11514](https://arxiv.org/abs/2312.11514):
  reducing flash transfer volume and larger contiguous transfers motivate the
  I/O experiment. This implementation is adjacent **expert** bundling, not the
  paper's neuron predictor, sparsification or row-column checkpoint transformation.
- [MoE-Infinity, arXiv:2401.14361v3](https://arxiv.org/abs/2401.14361v3):
  request-level activation locality motivates warm handoff. Whallm's small
  prompt-tail policy is not the paper's trace-selection/prefetch algorithm.
- [DuoServe-MoE, arXiv:2509.07379v2](https://arxiv.org/abs/2509.07379v2):
  phase-specialized scheduling and compute/transfer overlap motivate treating
  prefill and decode separately. No CUDA two-stream pipeline or offline-trained
  predictor has been transplanted.
- [FlashMoE, arXiv:2601.17063](https://arxiv.org/abs/2601.17063):
  SSD-oriented recency/frequency policy motivates measuring reuse rather than
  assuming LRU alone is optimal. Its trained ML replacement policy is deferred;
  Whallm already has a route-aware cache with separate observations/cost signals.

The previously audited oMLX/Rapid-MLX/llama.cpp sparse attention and expert-wave
ideas remain relevant; they do not justify a full-model speed claim. Learned
lookahead is a future candidate only after its predicted-read accuracy, wasted
bytes, demand priority and cancellation/lifetime rules can be measured.

## New experimental controls

App: **Qwen model > Advanced Settings > Qwen Flash experiments**.

| Control | Runtime/CLI | Default | Range |
| --- | --- | --- | --- |
| Experts per prefill read | `qwen_prefill_read_experts` / `--qwen-prefill-read-experts` | 1 | 1..32 |
| Warm decode experts per layer | `qwen_prefill_seed_experts` / `--qwen-prefill-seed-experts` | 0 | 0..128 |
| Overlap shared expert compute with reads | `qwen_shared_expert_overlap` / `--qwen-shared-expert-overlap` | false | boolean |

Read merging and warm handoff require layer-major whole-layer prefill with expert
waves disabled. The App preserves inactive saved choices but sends neutral values
when those dependencies are off. CLI/catalog misuse fails validation. Other model
kinds remain neutral. Settings persist per model, reset with Restore defaults and
lock during load/loaded/unload. All labels and help cover en/zh-Hant/zh-Hans.

### Contiguous reads

Each read contains up to the configured number of adjacent expert blobs. Sparse
prefetch IDs are split at gaps, and region order is preserved in the destination.
The existing aligned cache-bypass reader gets a bounded per-worker bounce buffer
large enough for the merged payload. Buffers remain until unload. For batch 4,
one payload is approximately 9.96 MiB; the staging allocation includes page
alignment slack. There are at most `read_workers` such worker buffers. This is
not zero-copy SSD DMA; copying into region-major buffers still exists. The App's
structural memory estimate includes additional staging in both phases.

### Warm handoff

Successful whole-layer scopes retain at most the last 128 **token positions** of
native routes across chunks, rank observed experts by frequency then recency,
and copy selected packed bytes into free decode slots after a GPU fence.
Per-layer quota is capped by `floor(active_slots / layer_count)`. No demand
resident is evicted for warm admission. This uses the existing slot budget;
there is no unbounded cache or file format change. Final slots own their bytes
and do not alias a layer buffer. Copy cancellation rolls back partial admissions;
failed layer scopes do not admit experts. First demand reuse is counted once.

This can improve early decode only when prompt-tail experts recur. It adds host
ranking, memory copies and prefill time; unrelated continuations may get no gain.
At 32 experts/layer, the maximum retained Qwen payload is about 3.735 GiB inside
the configured cache. Those residents coexist with prefill layer buffers, unlike
the default cold handoff. The App estimate includes that coexistence. The retained
payload may therefore reduce prompt headroom even though the slot budget is fixed.

### Resident shared-expert overlap

For a singleton input, `mx.async_eval(indices, shared)` submits the native router
and immutable resident shared expert before routed-expert acquisition. Final
routing, every selected expert, reduction order and sampling are unchanged.
No new thread or speculative model is created. Multi-token prefill is unchanged.
The setting also reaches native MTP layers, not proposal/acceptance logic. It may
be neutral or slower if launch overhead dominates or the GPU has no useful gap.

## Benchmark evidence rather than timing-only results

Throughput now waits for pending App configuration and emits a `diagnostics`
object. The App displays per-phase expert payload, consumer wait, hit/miss,
read-batch and warm-use counts, plus effective scalar configuration and hashes.
JSON export preserves the raw diagnostics. Old results without it still decode.
The configuration fingerprint omits filesystem paths, trace names and arbitrary
string secrets. Source hash identifies the Python files observed by this server;
it is not a signed build or proof of already imported code. Restart the server
when changing source. Corpus and output token hashes remain the benchmark's own.

Boundaries are **through the first yielded token / after the first yielded token**,
not exact GPU phase barriers. Asynchronous next-token work can cross the boundary;
collecting diagnostics adds no explicit GPU fence. `bytes_read` is logical expert
payload, not physical SSD traffic; `wait_seconds` covers only blocked future
consumption. Neither exposes all GPU stalls. Process disk bytes, when available,
are separate. Prompt cache stays disabled, and expert/OS caches are not flushed.
Cold and warm trials must not be mixed unknowingly. Missing counters are null,
not fictional zeros. Evidence collection failures restore runtime configuration.

## Reproduction and starting experiments

```sh
make test-qwen-flash-portable PYTHON=python3
make test-qwen-flash
WHALLM_QWEN_UI_ARTIFACTS="$PWD/scratch/streaming-ui" make test
PYTHONPATH=runtime .venv/bin/python Scripts/validate_qwen_flash_app_catalog.py scratch/streaming-ui/catalogs
make benchmark-qwen-streaming-io QWEN_FLASH_OUTPUT=scratch/streaming-io/results.json
```

The I/O microbenchmark uses actual Qwen-layout byte payloads and real cache/read
code, but only two synthetic layers and eight experts. It hashes payloads instead
of running an LLM, and includes hash/check cost in its consumer timer. It compares
read-1/4 and seed-0/2 with equal eight-slot capacity under reused and unrelated
traces, interleaving three repetitions. Local file warmness is not controlled;
cache bypass is not a claim that physical storage was cold. It proves contracts
and counts, not model throughput. It refuses to overwrite evidence.

For actual App Throughput, keep the same model, expert capacity, context/output
length, sampling, QSA settings, MTP state and thermal conditions. First export a
baseline with read=1 / seed=0 / overlap=off. Then compare **read=4**, **seed=32**,
and **overlap=on** individually; only then compare their combination. Repeat and
report median and tail latency, completed output token counts, output hash,
TTFT, decode rate, total time, expert waits and memory/swap. Start with 4K/512 and
32K/512, not different input/output lengths for the two versions. Restart and
reload for a separate cold-state series; perform an explicit warm-state series.
Do not increase all memory/worker settings simultaneously. Batch 8 and seed 64
are follow-up sweeps, not recommended maxima. No production defaults are promoted.

No complete checkpoint, physical user Mac, full-model quality or end-to-end speed
was evaluated here. Native-byte and tensor parity are necessary, not sufficient
for deploying an accelerated full model. Stop on numerical/quality differences,
failed continuation/cancellation, nonfinite output or sustained swap. All prior
QSA/MTP experimental caveats remain in force.

## Native validation findings

The expanded I/O suite found a baseline Darwin compatibility issue: CPython
3.12 on the hosted Mac does not expose `fcntl.F_RDAHEAD`. The bypass helper now
uses Apple's `bsd/sys/fcntl.h` value 45 only on Darwin when that symbol is missing.
Failures from the actual fcntl call still propagate. This fixes initialization;
it is not evidence that the user's throughput plateau had this cause.

Shared-expert overlap resolves routing on the CPU before submitting only the
shared branch, then consumes those already-materialized routes. It must not
submit indices and shared together and then read indices: a joint evaluation
can make the CPU wait for the work that was intended to overlap with I/O.
Submission ordering is tested independently of numerical parity. Actual GPU/I/O
overlap and end-to-end benefit remain workload-dependent, not inferred from a
submission counter.
