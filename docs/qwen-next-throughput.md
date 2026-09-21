# Qwen Next throughput: indexed decode and bounded prefill scheduling

Status: opt-in research implementation, not a measured full-model speed profile.
The previous feature baseline is `0b1cb7a46021b446195716ceb64d9fc1cced4cf4`.
Installed model bytes, expert top-k, quantization, sampling and master are unchanged.

## Audited external implementations

Source snapshots were acquired by GitHub Actions run 35562693244. Only source
was inspected; third-party implementations were not executed or copied into Whallm.

| Repository | Audited revision | Applicable ideas and limits |
| --- | --- | --- |
| [oMLX](https://github.com/jundot/omlx) | `d94d4cec989019344c2c4df592975e133f18255b` | Direct sparse-cache attention, narrow-query storage-axis gathers, larger bounded query batches, reduced Python dispatch. Resident-model and aggregate concurrency benchmarks are not SSD Whallm benchmarks. |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | `18cac511f573e127c0feb81b7f0f1e44693e320f` | Fused sparse prefill/verify, cache correctness and context-dependent crossover. Different quantization packs and M5-only NAX code are not interchangeable with Whallm's installed model. |
| [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) | `66e8938550c348c9b37a6d576163d2ca1e1bf8df` | Narrow direct-index split-K attention, explicit device/shape qualification, GDN fusion. Its measured chip/workload thresholds do not establish thresholds on M2 Max or M5. |
| [ds4](https://github.com/antirez/ds4) | `0aaea5a238fb41a35106a551e73c8409dfb751ac` | Dedicated C/Metal/CUDA Qwen graph, n-gram storage and native MTP. The GGUF/quantization contract differs; kernels cannot be transplanted unchanged. |

Primary code paths: oMLX's
`omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py`;
Rapid-MLX's `rapid_mlx/kernels/qsa_indexed_splitk.py`,
`rapid_mlx/kernels/qwen4_fused_gdn_decode.py` and
`docs/engineering/performance/2026-09-01-qwen4-fused-gdn-decode.md`;
mlx-serve's `src/qwen4_exp.zig` and `src/kernels/qsa_nax.metal`;
ds4's `docs/QWEN38_FLASH_NEXT.md`.
CUDA/Triton results are not Apple Metal measurements. Different weights,
contexts, MTP acceptance and concurrency prevent direct tok/s comparisons.

## Implemented and rejected experiments

### Bounded QSA query batches

`qwen_qsa_query_chunk` accepts 1..128 and retains the existing default 4.
Larger chunks amortize Python iteration, index selection and graph submission.
The original indexer, discrete top-k and causal selection remain in use.
`qwen_qsa_schedule.py` estimates per-query index-score, gathered K/V and score
workspace, reducing the request to a 256 MiB estimated per-chunk ceiling.
This is not a total allocation or RSS bound: pending lazy graph chunks, retained
outputs, SDPA scratch and resident tensors are additional. Different batched
kernels can round differently. The unchanged default bypasses extra planner work.

### Indexed QSA Decode / narrow MTP verification

`qwen_qsa_indexed.py` is an original two-stage Metal implementation. Each
query/head uses 32 splits that read only selected K/V rows, maintain FP32
online-softmax maxima/sums/weighted values, and merge their partials. There is
no gathered K/V temporary or token-major full-cache conversion. Actual tensor
strides support read-only cache slices without copying the full prefix.

The existing indexer supplies row IDs. Range, causality and validity are checked
before dereferencing. Empty splits and all-masked queries produce zero; duplicate
IDs preserve their multiplicity. KV, recurrent state and expert slots are not mutated.

Admission requires opt-in `qwen_qsa_indexed`, `qwen_sparse_sdpa`, batch-one GPU
inputs, 1..8 queries and supported head dimensions/dtypes. Unsupported layouts
use the existing gathered path. The dense no-gather route remains for KV lengths
at or below the 2048-token indexer budget. GPU dispatch errors are not silently
caught and retried. FP32 reductions differ from gathered BF16 SDPA: numerical
checks do not guarantee bit-identical logits, generated text or model quality.

Main-model and native MTP attention receive the new settings. Existing MTP
attention defaults remain when both controls are unchanged. No proposal,
acceptance, sampling or rollback algorithm is altered.

### Rejected storage-axis gather

An alternative narrow gather along the stored head-major axis was implemented
and checked for byte-identical selected rows. It showed no repeatable speed or
allocation benefit, so the final refinement restores the original gather.
It is retained as negative evidence, not advertised as an additional optimization.

## App and configuration

Advanced Settings > Qwen Flash experiments adds the following controls:

| UI | Preference | Runtime / CLI | Default |
| --- | --- | --- | --- |
| QSA queries per chunk | qwenQSAQueryChunk | qwen_qsa_query_chunk / --qwen-qsa-query-chunk | 4 |
| Use indexed QSA decode | qwenQSAIndexed | qwen_qsa_indexed / --qwen-qsa-indexed | false |

English, Traditional Chinese and Simplified Chinese copy, range checks and
accessibility identifiers are included. Settings persist per model, reset with
Restore defaults, and lock while loading, loaded or unloading. Indexed attention
is effectively off with SDPA off without erasing its saved choice. Old catalogs
and preferences safely omit the fields; non-Qwen effective fields are neutralized.
The memory overview is a structural estimate, not measured peak allocation.

[Recommended starting values](qwen-recommended-settings.md) and
[RuntimeConfig fragments](qwen-recommended-settings.json) provide theoretical
64/96 GiB profiles. They do not automatically change defaults or claim model fit.

## Measurements and evidence boundaries

Source recheck run [35564306040](https://github.com/yanun0323/Whallm/actions/runs/35564306040)
tested published source `7e3ff70e1ba653fa07b106547868607e17fb4404`.
Refinement run [35564665325](https://github.com/yanun0323/Whallm/actions/runs/35564665325)
validated the removal of unhelpful gathering and added narrow 128K fixtures;
its validation job passed, but publication failed on workflow-write permission.
That publication failure is not reported as a successful run. Subsequent source
publication and read-only CI are independently checked.

The environment reports Apple M1 (Virtual), Apple Paravirtual device, 7 GiB RAM,
macOS 15.7.9 and MLX 0.32.2. These are synthetic BF16 QSA component measurements
with 24 query heads, 2 KV heads and dimension 256, not physical-chip or complete
model benchmarks. There are 3 warmups and 9 interleaved A/B-order rounds; timing
includes Python graph construction and synchronous mx.eval. Nine-call p95 is
noisy and is not an end-user request percentile. No expert SSD I/O is timed.

The following unfiltered refinement p50 values compare the previous fused path
with indexed decode or larger query chunks, in milliseconds:

| KV tokens / queries | Previous fused, chunk 4 | Indexed decode | Chunk 32 |
| --- | ---: | ---: | ---: |
| 8192 / 1 | 3.7954 | 3.2838 | not applicable |
| 8192 / 3 | 6.8268 | 5.8238 | not applicable |
| 8192 / 128 | 141.9900 | not admitted | 142.4273 |
| 32768 / 1 | 7.3900 | 5.5604 | not applicable |
| 32768 / 3 | 9.4785 | 6.2640 | not applicable |
| 32768 / 128 | 208.0747 | not admitted | 201.9915 |
| 131072 / 1 | 26.2385 | 15.0913 | not applicable |
| 131072 / 3 | 28.3290 | 16.7276 | not applicable |

Prefill improvement varies substantially across runs. The earlier source recheck
measured 203.719 -> 173.372 ms at 128 queries / 32K, but extra peak MLX allocation
rose from about 798 to 1100 MiB. At 8K larger chunks did not help consistently;
chunk 16's 32K p95 also regressed in the source recheck. Unchanged-path timing
noise is not credited as optimization. All trials, raw samples and output/source
hashes remain in the workflow artifacts. No full-model speed multiplier follows.

The refinement validation ran 137 Qwen results (133 passed, 4 pre-existing
archive-dependent skips), 150 Swift results (147 passed, 3 installed-model skips),
25 portable tests and 10 Swift-to-Python catalog cases. Portable tests are also
included in Qwen discovery: do not add the counts together. Additional final
qualification covers head dimensions 64/128, large logits across all splits,
empty-head admission and the recommended-profile parameter contract. Refer to
the final commit's CI artifact for those expanded counts and their results.

## Reproduction and promotion gates

```sh
make test-qwen-flash-portable PYTHON=python3
make test-qwen-flash
WHALLM_QWEN_UI_ARTIFACTS="$PWD/scratch/qsa-ui" make test
PYTHONPATH=runtime .venv/bin/python Scripts/validate_qwen_flash_app_catalog.py scratch/qsa-ui/catalogs
make benchmark-qwen-qsa QWEN_FLASH_OUTPUT=scratch/qsa/components.json
```

The microbenchmark requires Apple Metal, pinned dependencies and Git history
containing the baseline. It refuses to overwrite evidence. Installed-model
pilots include qsa-chunk16, qsa-chunk32, qsa-indexed and qsa-throughput. Compare
separate worktrees and identical model bytes, cache budgets, sampling, prompt
history and MTP settings to isolate changes.

GDN fusion was not blindly copied: Rapid-MLX's Qwen4 normalization/gate rounding
contract differs from the pinned Qwen3.5 GDN currently composed by Whallm,
including Whallm's sigmoid-gate patch. NAX-only kernels, lower-bit repacking,
expert pruning and speculative-tree restructuring also remain deferred.
Require physical-device A/B, real code/prose/Traditional Chinese prompts,
continuation, cancellation, peak-memory and full-model quality checks before
using these experiments as a production profile.
