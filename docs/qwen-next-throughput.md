# Qwen Next throughput: indexed decode and bounded prefill scheduling

Status: opt-in research implementation. Full-model throughput and quality have
not been measured. The previous feature baseline is
`0b1cb7a46021b446195716ceb64d9fc1cced4cf4`. Installed model bytes, expert top-k,
quantization, sampling and master are unchanged.

## Audited external implementations

These source snapshots were acquired by GitHub Actions run 35562693244, with
Git revisions in the source artifact. Only source was read; third-party code
was not executed or copied into Whallm.

| Repository | Audited revision | Applicable ideas and limits |
| --- | --- | --- |
| [oMLX](https://github.com/jundot/omlx) | `d94d4cec989019344c2c4df592975e133f18255b` | Direct sparse-cache attention, narrow-query storage-axis gathers, larger bounded query batches, reduced Python dispatch. Resident-model and aggregate concurrency benchmarks are not SSD Whallm benchmarks. |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | `18cac511f573e127c0feb81b7f0f1e44693e320f` | Fused sparse prefill/verify, cache correctness and context-dependent crossover. Different quantization packs and M5-only NAX code are not interchangeable with Whallm's installed model. |
| [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) | `66e8938550c348c9b37a6d576163d2ca1e1bf8df` | Narrow direct-index split-K attention, explicit device/shape qualification, GDN fusion. Its measured chip/workload thresholds do not establish thresholds on M2 Max or M5. |
| [ds4](https://github.com/antirez/ds4) | `0aaea5a238fb41a35106a551e73c8409dfb751ac` | Dedicated C/Metal/CUDA Qwen graph, n-gram storage and native MTP. The GGUF/quantization contract differs; kernels cannot be transplanted unchanged. |

Primary code paths:

- oMLX: `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py`.
- Rapid-MLX: `rapid_mlx/kernels/qsa_indexed_splitk.py`,
  `rapid_mlx/kernels/qwen4_fused_gdn_decode.py` and
  `docs/engineering/performance/2026-09-01-qwen4-fused-gdn-decode.md`.
- mlx-serve: `src/qwen4_exp.zig` and `src/kernels/qsa_nax.metal`.
- ds4: `docs/QWEN38_FLASH_NEXT.md`.

SGLang/Triton/CUDA optimizations on DGX Spark are architectural references, not
Apple Metal implementations. Raw advertised tok/s across different weights,
contexts, quantization, MTP acceptance or batching must not be ranked as an
apples-to-apples engine comparison.

## Implementation

### Bounded QSA query batches

`qwen_qsa_query_chunk` accepts 1..128, retaining the existing default 4. Larger
chunks amortize Python iteration, index selection and Metal graph submission
in sparse prefill. The original indexer, discrete top-k and causal selection
remain in use; this is not expert pruning or an approximate attention budget.

`qwen_qsa_schedule.py` estimates per-query index-score, gathered K/V and score
workspace, and reduces the requested chunk to a 256 MiB estimated workspace
ceiling. This is not a hard GPU allocation or process RSS limit: resident
weights/cache, retained layer outputs, allocator overhead and scratch outside
this formula remain additional. Different batched kernels can round differently.

### Indexed QSA Decode / narrow MTP verification

`qwen_qsa_indexed.py` is an original two-stage Metal implementation. One SIMD
threadgroup per query/head/split reads only selected K/V rows, maintaining FP32
online-softmax maxima, sums and weighted values. A second kernel merges 32
splits. There is no gathered K/V temporary or full token-major cache conversion.
Tensor strides permit read-only cache slices without copying the full prefix.

The existing indexer supplies row IDs; the kernel also checks range, causality
and validity before dereferencing. Empty splits and all-masked queries return
finite zero contributions. Duplicate row IDs preserve their original multiplicity.
No KV, recurrent state, index history or shared expert slot is mutated.

The path is opt-in through `qwen_qsa_indexed`, requires `qwen_sparse_sdpa`, and
only admits batch-one GPU inputs with at most 8 queries and supported head
shapes/dtypes. The original gathered path handles prefill and unsupported
layouts. The existing dense no-gather path remains for KV <= indexer budget.
Validation/compilation errors after a GPU dispatch are not silently caught and
retried; this avoids hiding a failed command buffer.

FP32 partials change the floating-point reduction order versus gathered BF16
SDPA. This is numerically checked, not advertised as bit-identical, output-token
identical or full-model quality validated. Main-model and native MTP attention
receive the new settings; existing MTP attention defaults are retained when
both new controls are at their defaults. No MTP acceptance rule is modified.

### Storage-axis gather

The fallback for narrow query chunks gathers from the stored head-major cache
axis instead of indexing a transposed full-cache view. Selected row values and
order are byte-identical; the downstream computation remains unchanged. This
is not new SSD prefetch or a change to the expert cache.

## App and configuration

The model's Advanced Settings > Qwen Flash experiments includes:

| UI | Preference | Runtime / CLI | Default |
| --- | --- | --- | --- |
| QSA queries per chunk | qwenQSAQueryChunk | qwen_qsa_query_chunk / --qwen-qsa-query-chunk | 4 |
| Use indexed QSA decode | qwenQSAIndexed | qwen_qsa_indexed / --qwen-qsa-indexed | false |

English, Traditional Chinese and Simplified Chinese labels, range checks and
accessibility identifiers are included. Choices persist per model, reset with
Restore defaults, and are locked while loading, loaded or unloading. Indexed
attention is effectively off when SDPA is off, without erasing the saved choice.
Old preferences and catalogs omit the new fields safely. Non-Qwen model catalog
fields are neutralized; standalone non-Qwen runtime opt-ins are rejected.
The memory overview reflects the requested query size and indexed scratch;
its structural estimate is not a measured peak.

## Reproduction

```sh
make test-qwen-flash-portable PYTHON=python3
make test-qwen-flash
WHALLM_QWEN_UI_ARTIFACTS="$PWD/scratch/qsa-ui" make test
PYTHONPATH=runtime .venv/bin/python Scripts/validate_qwen_flash_app_catalog.py scratch/qsa-ui/catalogs
make benchmark-qwen-qsa QWEN_FLASH_OUTPUT=scratch/qsa/components.json
```

The microbenchmark requires Apple Metal, pinned runtime dependencies and Git
history containing the baseline commit. It uses actual QSA dimensions, seeded
synthetic BF16 tensors, 3 warmups and 9 alternating A/B-order rounds. Each timing
includes Python graph creation and `mx.eval`. It records all samples, source and
output hashes, numerical errors, device details and extra MLX peak allocation.
It does not load a full checkpoint, read experts from SSD or measure tok/s.
No evidence file is overwritten.

Installed-model pilots add `qsa-chunk16`, `qsa-chunk32`, `qsa-indexed` and
`qsa-throughput` variants to `pilot-qwen-flash`. That pilot deliberately uses
its own fixed experimental settings, not the App speed profile. Compare old
and new Git worktrees to isolate unconditional gather changes. Keep identical
model bytes, cache budgets, prompt history, sampling and MTP settings.

## Deferred decisions

GDN fusion was not blindly copied. Rapid-MLX's Qwen4 specialization uses a
normalization and gate-rounding contract different from the pinned Qwen3.5 GDN
implementation currently composed by Whallm. Its Qwen3.5 gate specialization
also does not implement Whallm's patched sigmoid gate. A correct port needs a
separate recurrent-state and full-logit validation effort, not a class rename.

NAX-only kernels, lower-bit model/PLE repacking, top-k pruning and speculative
tree restructuring were not enabled. No full-model accuracy or end-to-end
throughput claims follow from this component work. Physical-device A/B, real
code/prose/multilingual prompts, long-context continuation, cancellation and
model-quality checks are required before adopting a production profile.
