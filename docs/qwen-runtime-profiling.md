# Qwen runtime profiling

`Scripts/profile_qwen_runtime.py` profiles the installed, complete Qwen3.8 model
without changing App preferences, installed model bytes or normal server behavior.
It uses the pinned runtime and requires Apple Silicon. No extra dependencies or
root privileges are needed.

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/profile_qwen_runtime.py \
  --model "$HOME/.dsmodel/qwen3.8-flash-next.dsv4" \
  --output scratch/qwen-profile-baseline --mode baseline --requests 2
# Repeat in separate processes/directories with --mode host and --mode sync.
```

Defaults: bundled Code corpus, 4096 input tokens, 128 output limit, greedy,
seed 42, 3072 expert slots, LRU, 16 readers, ready-expert decode, 1024-token
layer-major non-batched Prefill, 30 GiB MLX limit, and the App-style optimized
N-gram/normalization/index-cache/phase-memory controls. These are an explicit
measurement configuration, not an automatic copy of saved App settings.
Prompt Cache, MTP, DSpark and approximate expert dropping are disabled.
`--config overrides.json` accepts RuntimeConfig overrides; the complete effective
configuration is saved. New output directories are required to preserve evidence.

## Three modes

- **baseline**: no model hooks; a nominal 0.5-second system sampler remains active.
  Use this for normal request time, first-yield latency and decode throughput.
- **host**: temporary, owner-thread-only hooks on all 48 decoder layers and their
  components. No additional GPU evaluation. Durations measure graph construction
  and existing waits, **not GPU execution**; deferred work crosses boundaries.
- **sync**: additionally evaluate returned arrays and synchronize at each compute
  boundary. This breaks normal overlap, may change allocation and is **not normal
  throughput or pure GPU kernel time**. Use it to locate expensive components,
  then validate hypotheses against separate baseline runs.

Hooks cover embedding, PLE/N-gram hashing and row lookup, hyper-connections and
residuals, Gated DeltaNet, QSA/index projection/attention, router projection,
shared expert, routed experts, expert acquisition, LM head and cache evaluation.
Load has common-weight and construction/loading subspans. Uninstrumented outer
work (including generation/sampling/detokenization) remains in the phase residual;
HTTP, queueing and UI are outside this harness.

## Evidence and units

Each run saves `protocol.json`, `result.json`, `samples.jsonl`, input token IDs
and a source archive. Results include commit, source/script/manifest/corpus/token
hashes, environment, full configuration, generated token IDs and output hash.
Compare hashes before interpreting profiler or configuration differences.

- Phase boundaries: load, corpus tokenization, through first **yielded** token,
  and after first yield. Next-token pipelining may cross that boundary.
- Layer/component aggregates (separate for each request): call/error counts,
  inclusive/exclusive wall seconds, process CPU seconds, owner-thread CPU seconds
  from `thread_time`, OS-reported minor/major faults and context switches, page-ins,
  process disk read/write bytes, logical expert
  bytes, hits/misses/evictions, read/pack/routing-sync/consumer-wait counters,
  and boundary RSS/physical-footprint/MLX active/cache gauges. N-gram logical bytes
  count requested rows including duplicates, not pages fetched from storage.
- Inclusive times and counters are nested: **do not sum parents and children**.
  Worker activity is process-wide and may be charged at a later boundary.
  Read time can overlap compute; do not add it to total request time.
- Process CPU seconds use `getrusage`, not an assumed nanosecond conversion of
  Darwin's Mach-tick `proc_pid_rusage` CPU counters. CPU utilization is
  `100 * (user + system seconds) / wall seconds`; 100% means one CPU core.
  Owner CPU excludes other threads, but includes profiling work between snapshots.
  CPU counter windows also include snapshot bookkeeping outside the timed wall
  window. For tiny components, CPU delta can therefore exceed timed wall: do not
  derive component CPU utilization or pure operator CPU cost from these deltas.
  OS-reported zero fault/switch counters are not evidence that every kind of stall
  is absent.
- Memory gauges overlap; RSS, physical footprint and MLX memory cannot be added.
  Boundary maxima and sampled peaks can miss transient allocations. Phase MLX
  allocator peaks are recorded separately and include live resident allocations.
- GPU utilization comes from device-wide IORegistry PerformanceStatistics. It
  includes other applications, uses a driver-defined averaging window, and is
  sampled at phase level. It is **not per-layer/process utilization, bandwidth,
  power, occupancy or per-kernel GPU time**. Unavailable counters are not zero.
- Process disk accounting is not physical SSD-device traffic. Logical reads can
  hit OS file cache. Empty expert slots do not imply a cold OS/SSD cache.
- `vm_stat` and swap snapshots are system-wide, not attributable solely to Qwen.

Repeated requests keep expert slots but not Prompt Cache. OS caches are not
purged and other applications are not terminated. Default safety stops are
900 seconds, sampled 48 GiB process footprint, or 256 MiB system swap growth;
these are stop conditions, not hard allocation limits. A stopped run is not a
completed benchmark. Raw samples and failure information are retained.

A bottleneck conclusion requires normal-run evidence plus a controlled change,
not just a large synchronized span. Report workload, cache state, repeated-run
variation, observer overhead and all unavailable metrics alongside the conclusion.

## Per-layer Metal activity

`--timeline` writes `timeline.jsonl` in host or sync mode. It contains PID, the
Mach absolute clock, request/phase boundaries and decoder intervals. Aggregates
now use `(request, phase, layer, component)`; result/profile schema version is 2.
Normal baseline mode rejects this option. Server behavior and App settings are
unchanged.

For temporal per-layer GPU activity, collect **a separate sync diagnostic** with
Xcode Metal System Trace. Sync timeline mode drains submitted GPU work before each
decoder window; it still cannot identify semantic ownership of lazy dependencies.
Use a matched normal run with the same token limit to measure observer effects.
For example, from the repository root, with fresh output paths:

```sh
export PYTHONPATH="$PWD/runtime"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun xctrace record --template 'Metal System Trace' --time-limit 180s \
  --no-prompt --output scratch/qwen-layers.trace \
  --env "PYTHONPATH=$PYTHONPATH" --launch -- \
  "$PWD/.venv/bin/python" "$PWD/Scripts/profile_qwen_runtime.py" \
  --model "$HOME/.dsmodel/qwen3.8-flash-next.dsv4" \
  --output "$PWD/scratch/qwen-layers-sync" --max-tokens 32 --mode sync --timeline
for schema in time-info metal-gpu-intervals; do
  xcrun xctrace export --input scratch/qwen-layers.trace \
    --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$schema\"]" \
    --output "scratch/qwen-$schema.xml"
done
.venv/bin/python Scripts/analyze_qwen_metal.py \
  --result scratch/qwen-layers-sync/result.json \
  --timeline scratch/qwen-layers-sync/timeline.jsonl \
  --time-info scratch/qwen-time-info.xml \
  --gpu-table scratch/qwen-metal-gpu-intervals.xml \
  --output scratch/qwen-layer-gpu-summary
```

The analyzer validates PID, sync mode, complete non-overlapping decoder windows,
phase containment, capture tail coverage and counts against profiler rows.
Missing GPU evidence is an error, not zero usage. It unions overlapping Active
intervals, including nested GPU records, before calculating activity/wall ratios.
Host timelines are **not** accepted for this attribution because asynchronous GPU
work crosses their boundaries. Decoder and phase rows are nested; do not sum them.
GPU-active fraction is not core occupancy, memory bandwidth or reclaimable idle
time. Whole-process RAM at a layer boundary is not that layer's owned bytes or a
transient layer peak. Trace finalization is outside the recording time limit and
needs its own time/memory budget; traces may contain other processes' metadata.

## Configuration and external-observer follow-up

Vary one setting at a time and verify output IDs, not just the greedy seed.
SDPA/indexed attention can change rounding, generated tokens and subsequent expert
routes. A changed working set can increase Decode disk reads even when an isolated
attention operation gets faster. To separate these effects, replay identical
materialized component inputs, check their hashes before/after, and report that
resident microbenchmark separately from complete-model throughput and quality.

A larger QSA query chunk may improve Prefill while increasing transient MLX memory.
Compare phase peaks as well as the overall footprint: a larger Decode peak can
hide a Prefill regression. Whole-layer batching and shared overlap are hypotheses,
not automatically beneficial configurations.

Optional Xcode Metal System Trace can expose per-process GPU encoder intervals.
Use a separate run, filter by launched PID, align phase clocks through trace
`time-info`, and union overlapping intervals rather than summing nested records.
Report capture coverage, other-process activity, observer slowdown and cache state.
Command-buffer counts are not kernel counts; active intervals are not ALU occupancy
or hardware memory bandwidth. Finalizing a trace may itself take minutes and
substantial memory; bound collection and finalization separately. Keep traces local
because they can include other processes and system metadata.

Validate third-party profilers before interpreting them. In the measured local
CPython 3.14.7 build, `cProfile` captured a worker-only probe despite being enabled
from the main thread, and model profiles contained self times exceeding cumulative
times. Those timing rankings were rejected, not relabeled as CPU-only cost. This
is separate from `QwenProfile`'s explicit owner-thread guard. Independently checked
call counts can still motivate an experiment, but are not proof of its speedup.

## Bulk eviction follow-up

Large LRU/LFU slot reservations now build a temporary rank-ordered list per layer
rather than repeatedly popping and reinserting the same protected/reserved entries
for each victim. Eligibility is rechecked after every insertion/eviction, so the
incoming layer can cross its reserve boundary. Hard expert pins remain excluded;
soft layer pins remain the last fallback, after unpinned reserved entries.

The fast path requires more pending evictions than `max(32, layer_count)` and all
incoming expert keys to be protected. Small Decode reservations and the `route`
policy keep their existing selector. Rank, slot assignment, frequency decay,
I/O scheduling and GPU buffer lifetime are unchanged. Temporary metadata is
bounded by resident entries and is not retained across reservations. The existing
global heap discards stale victim records through normal lazy cleanup/rebuilding.

`runtime/tests/test_batched_eviction.py` checks scalar equivalence, reserve/pinning,
resize/decay, failed reads, ready/staged consumers and a deterministic heap-work
regression. The tiny fixtures need no installed checkpoint or local research archive.
