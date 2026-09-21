# Qwen throughput review and SSD handoff experiments

Baseline: `c71231229f29c79160f51aba2b02d4aa1380b910`, branch
`feat/qwen-flash-optimize`. The user's unchanged Throughput score has not been
reproduced here. The review follows benchmark entry, catalog/configuration,
generation, prefill, expert cache/reader, Qwen routed/shared experts, QSA, MTP
and App settings. It does not claim every repository file or the checkpoint
was tested. Installed model bytes, quantization, routing and sampling are unchanged.

## Findings

| Baseline finding | Consequence | Implemented response |
| --- | --- | --- |
| Throughput disables ordinary, persistent and DSpark prompt caches | Conversation-cache improvements cannot improve this benchmark | Preserve its definition and export effective configuration |
| App benchmark preparation does not await pending model configuration | An immediate run after editing can race the update; this is not proven to explain the user's score | Await completion and propagate configuration failure |
| Whole-layer prefill releases decode slots and recycles whole-layer buffers | Hot experts just read for the prompt are not adopted into the decode cache | Opt-in, bounded prompt-tail warm handoff |
| One `_pread_views` per expert even for contiguous whole-layer ranges | Per-call and alignment overhead; each existing blob is already large | Optional consecutive-expert read batches, never bridging gaps |
| Shared expert computation remains lazy until routed reduction | Independent resident work can remain unsubmitted during I/O waits | Resolve routing once, then submit shared work before acquisition |
| Previous singleton fast path only optimizes non-ready dispatch | Its percentage does not describe normal ready-expert Decode | Retain correct ready order and test that actual path |
| QSA optimization does not reduce routed-expert SSD traffic | Large attention-only gains can have small overall impact | Measure request evidence rather than extrapolate component gains |
| `read_seconds` may overlap computation | It is not the critical-path blocking fraction | Add consumer-only expert-future `wait_seconds` |

The whole-layer path requests 48 * 512 * 2,611,200 bytes, about **59.77 GiB**
of logical expert payload per complete prompt pass. It already loads each layer
once across token chunks; larger token chunks do not divide these bytes again.
Logical payload is not physical SSD traffic. Existing phase memory, next-layer
prefetch, ready reads, eviction policies, MXFP4 and MTP are not new inventions.

For illustration only, doubling attention that occupies 10% of runtime gives
1 / (0.9 + 0.1 / 2), about 1.053x overall. The 10% is not a measured breakdown.

## Primary research and the port boundary

Source inspection is not reproduction of third-party benchmarks. No external
kernel bodies or weights were copied.

- [streamlx, pinned source](https://github.com/srcterm/streamlx/tree/146507b59293058b3a1cecc43f47947b25012297):
  `streamlx/integrate.py` adopts hot sweep experts and fences pool/layer tensors;
  `reader.py` merges spans. Resident work is submitted while misses arrive.
  These motivate the handoff and scheduling, not adoption of its model wrapper
  or claims about Whallm's particular Qwen3.8 installation.
- [LLM in a flash](https://arxiv.org/abs/2312.11514): flash-volume reduction and
  larger contiguous transfers motivate adjacent expert bundling. This is not
  the paper's neuron predictor, sparsification or checkpoint transformation.
- [MoE-Infinity](https://arxiv.org/abs/2401.14361v3): activation locality motivates
  warm handoff. Our prompt-tail heuristic is not its trace-prefetch algorithm.
- [DuoServe-MoE](https://arxiv.org/abs/2509.07379v2): phase-specialized scheduling
  and compute/transfer overlap motivate separate prefill/decode treatment.
  No CUDA two-stream pipeline or offline-trained predictor was transplanted.
- [FlashMoE](https://arxiv.org/abs/2601.17063): SSD-oriented frequency/recency
  motivates measuring cache reuse. Its ML replacement policy remains deferred;
  useful training and replacement evaluation require actual Whallm traces.

Learned lookahead, top-k pruning, new quantization and different sampling were
not used to manufacture higher scores. Prior oMLX/Rapid-MLX/llama.cpp reviews
remain relevant, but cannot establish an end-to-end Whallm gain.

## Experimental controls

App: **Qwen model > Advanced Settings > Qwen Flash experiments**.

| Control | Runtime field / CLI | Default | Range |
| --- | --- | --- | --- |
| Experts per prefill read | `qwen_prefill_read_experts` / `--qwen-prefill-read-experts` | 1 | 1..32 |
| Warm decode experts per layer | `qwen_prefill_seed_experts` / `--qwen-prefill-seed-experts` | 0 | 0..128 |
| Overlap shared expert compute with reads | `qwen_shared_expert_overlap` / `--qwen-shared-expert-overlap` | false | boolean |

Read merging and warm handoff require layer-major, whole-layer expert prefill
with expert waves disabled. Inactive UI choices are preserved but exported as
neutral values. CLI/catalog misuse fails validation. Other models stay neutral.
Preferences persist per model, reset with Restore defaults and lock during
load/loaded/unload. English, Traditional Chinese and Simplified Chinese cover
labels, warnings and dependencies. No production default is promoted.

### Consecutive reads

Reads combine adjacent requested expert blobs, preserving source-region order
and never reading across unrequested gaps. The aligned cache-bypass reader has
a bounded bounce buffer per worker. Batch four has about 9.96 MiB payload per
read plus alignment slack; buffers remain until unload. This is not zero-copy
SSD DMA. Extra CPU copies, staging and SSD behavior can outweigh fewer calls.
The App's structural estimate includes additional staging in both phases.

### Warm handoff

A successful whole-layer scope keeps at most the last 128 token positions of
native routes across chunks. Frequency, then recency ranks experts; per-layer
quota is capped by floor(active_slots / layer_count). Selected packed bytes are
copied into **free** existing decode slots after GPU work is fenced. No demand
resident is evicted for prediction, and slots do not alias recycled layer buffers.
Failed/cancelled copies roll back admissions. First demand reuse is counted once.

Warm handoff helps early Decode only when those prompt-tail experts recur. It
adds host ranking, copies and Prefill work. Thirty-two experts per layer means
up to about 3.735 GiB of resident payload inside the existing expert budget,
coexisting with prefill layer buffers. Fixed total slot capacity does not mean
unchanged prefill headroom. App estimates account for this coexistence.

### Shared expert overlap

Single-token routing is first materialized on the CPU. Only then does
`mx.async_eval(shared)` submit the independent, resident shared branch. The
routed path consumes those already materialized IDs without another host wait.
Submitting indices and shared together can accidentally wait for both when
reading the indices and defeat overlap. Submission order and MXFP4 output parity
are separately tested for ordinary and ready dispatch. Multi-token Prefill is
unchanged. The flag also reaches native MTP layers without changing acceptance.
A submission counter is not proof of GPU/I/O concurrency or end-to-end speed.

## Throughput diagnostics

Throughput waits for pending model configuration and returns `diagnostics`.
The App displays expert logical payload, consumer wait, hits/misses, read batches,
warm admissions/use, effective scalar settings and configuration/source hashes.
JSON export preserves these values; old results without diagnostics still work.
Paths, trace filenames and arbitrary string credentials are not fingerprinted.
The source hash identifies files observed by the server, not a signed build or
proof of already imported code. **Restart the server after updating source.**

Counter boundaries are **through / after the first yielded token**, not exact
GPU barriers: pipelined work may cross them. No explicit GPU fence is added for
measurement. `wait_seconds` counts expert-future consumer blocking, not all GPU,
N-gram or synchronization waits. It is not interchangeable with `read_seconds`.
Expert tables cover the **main-model** cache, excluding separate MTP/DSpark caches
and N-gram reads. Optional process I/O has broader process scope, not device-wide
physical traffic. Prompt cache is disabled; expert and OS caches are not flushed.

## Native validation findings and recorded outcomes

Expanded native tests found a baseline Darwin compatibility bug: this CPython
3.12 build does not export `fcntl.F_RDAHEAD`. The helper uses Apple's
`bsd/sys/fcntl.h` value 45 only on Darwin when the symbol is missing. Actual
fcntl errors still propagate. Fixing initialization does not establish the
cause of the user's unchanged score.

Run **35576548142** applied the digest-pinned candidate and published source
`181a6cb8b4fe1f0712cc23103a2b35071ed309ef`. Results: **210 Python tests, 203 passed,
7 pre-existing archive-dependent skips; 155 Swift tests, 152 passed, 3 existing
installed-model skips**. The 31 portable tests are included in Python discovery,
not additional to it. Thirteen Swift-to-Python catalogs and twelve localized
settings renders passed. Final read-only CI checks actual committed source.

The I/O fixture is two layers, eight experts/layer, identical eight-slot capacity
and three interleaved repetitions on an Apple Paravirtual device. No model
inference is run; the consumer hashes native bytes. All output hashes agree.
Every variant reads 41,779,200 logical prefill bytes. Batch four reduces calls
**16 -> 4**, but does not improve the measured reuse-trace prefill p50:
**17.63 ms baseline, 18.17 ms read-four, 23.09 ms combined read-four/seeding**.
Seeding removes four demand misses and 10,444,800 logical bytes only in the
intentionally reused-route trace. An unrelated continuation still has four
misses and zero seeded-expert first-demand hits. Negative results are preserved.
Warm/OS-cache state is uncontrolled; these are not physical-SSD or token rates.

See [the machine-readable evidence](validation/qwen-streaming-review-20260921.json)
and the run artifact for samples, source/output hashes, tests and rendered UI.
No complete checkpoint, physical user Mac, model quality or end-to-end speed was
measured. Native byte/tensor parity is necessary, not sufficient for promotion.

## Controlled trials on the user's machine

Keep model bytes, context/output lengths, sampling, worker counts, expert budget,
QSA options and MTP settings constant. First measure with MTP off. Test code and
novel corpora at 4K/512 and 32K/512, with multiple repetitions and a separate warm
series. Compare completed output counts/hashes, TTFT, Decode rate, total time,
expert waits and peak memory/swap. Do not raise every setting simultaneously.

| Variant | Read experts | Seed experts/layer | Shared overlap |
| --- | ---: | ---: | --- |
| Baseline | 1 | 0 | off |
| Read only | 4 | 0 | off |
| Handoff only | 1 | 32 | off |
| Shared only | 1 | 0 | on |
| Combined, only after individual qualification | 4 | 32 | on |

Read eight / seed sixty-four are later sweeps, not recommended maxima. Revert
options that do not reduce total latency or that cause memory pressure. Stop on
quality differences, nonfinite output, failed cancellation/continuation or swap.
All prior QSA and MTP caveats remain. No speed improvement is guaranteed.

```sh
make test-qwen-flash-portable PYTHON=python3
make test-qwen-flash
make test
make benchmark-qwen-streaming-io QWEN_FLASH_OUTPUT=scratch/streaming/io.json
```

The microbenchmark requires pinned Apple Metal dependencies and refuses to
overwrite evidence. It is separate from the App's real-model Throughput run.
