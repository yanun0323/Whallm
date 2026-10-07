# Qwen Flash App controls

The Qwen model's Advanced Settings page now contains a **Qwen Flash experiments**
section. It exposes opt-in runtime experiments without changing existing defaults
or the installed-model format. It is not shown for DeepSeek models. The Packed
GDN control appears only for **Swift1.5-Qwen3.8-Flash-Next**.

| App control | Saved preference | Runtime catalog field | Default |
| --- | --- | --- | --- |
| Use packed GDN prefill (Swift only) | `qwenPackedGDNPrefill` | `qwen_packed_gdn_prefill` | `false` |
| Use sorted expert prefill | `qwenSortedExpertPrefill` | `qwen_sorted_expert_prefill` | `false` |
| Experts per wave | `qwenExpertWaveSlots` | `qwen_expert_wave_slots` | `0` (off) |
| N-gram read backend | `qwenNgramIO` | `qwen_ngram_io` | `mmap` |
| N-gram row cache MiB | `qwenNgramCacheMiB` | `qwen_ngram_cache_bytes` | `0` (off) |
| QSA queries per chunk | `qwenQSAQueryChunk` | `qwen_qsa_query_chunk` | `16` |
| Use indexed QSA decode | `qwenQSAIndexed` | `qwen_qsa_indexed` | `false` |
| Use fused QSA SDPA | `qwenSparseSDPA` | `qwen_sparse_sdpa` | `false` |
| Skip text | `qwen_qsa_skip_complete_gather` | same name | `true` |

Wave size accepts every integer from 0 through 512. It bounds experts acquired
per wave, not the total expert cache or process memory. The runtime also caps the
wave by its available expert slots. MiB uses 1,048,576 bytes and accepts every
integer from 0 through 512. The cache budget counts retained packed FP8 payload;
Python metadata, transient buffers and the OS page cache are additional.

## Editing and lifecycle

Open **Model > Qwen3.8-Flash-Next > Advanced Settings**, then scroll to
**Qwen Flash experiments** below Runtime. Settings use the existing per-model
UserDefaults persistence and configure-model catalog update. Changes take effect
on the next model load. A loaded, loading or unloading model has locked controls;
unload it first. Starting the server with a catalog, configuring an unloaded
model, and loading it all use the same runtime JSON fields.

The existing two-confirmation **Restore defaults** action resets these controls
with the rest of that model's advanced settings. It does not remove model files
or Alias values. Existing preferences and runtime catalogs without the new
fields decode to the disabled/default behavior. Other model kinds normalize the
new fields to their neutral values before sending them to Python.

## Dependencies without losing saved choices

- With waves enabled, the effective runtime disables whole-layer expert batching,
  whole-layer next-layer prefetch and grouped prefill. The corresponding existing
  controls display their effective disabled state and explain why. Their saved
  choices are not overwritten; setting wave size back to zero restores them.
- Sorted expert prefill is active only with Prefill acceleration on and expert
  waves off. Otherwise the control shows as off with an explanation, Python
  receives `false`, and the saved choice is kept. It works with MTP on.
- With mmap selected, the row-cache control is disabled and Python receives a zero
  cache budget. Its saved MiB value is retained and becomes active again with pread.
- Enabling MTP temporarily disables Packed GDN and sends `false` to the runtime,
  without erasing its saved choice. Turning MTP off restores that choice. Direct
  runtime configuration rejects Packed GDN with MTP or a non-Swift model.
  No MTP proposal, acceptance, checkpoint or rollback changes are included.

The memory overview counts the active packed-row budget in prefill/decode and
uses the non-whole-layer allocation path for waves. It does not reduce the total
expert-slot budget to the wave size, or claim that SDPA reduces peak memory.
These remain structural estimates, not measured peaks or hard limits.

## Packed GDN prefill experiment

This is a working, **default-off experiment**, not an adopted speed improvement.
The earlier prototype's two-pair 4K Code/Novel pilot did not meet the 5% first-token
speed threshold; its component speedups must not be described as App speedups.
There is no new performance claim from the production integration.

The control uses a per-model GDN implementation, not a global mlx-lm patch or a
research-file import. It is limited to M5 (`applegpu_g17*`), batch-one text Prefill
calls of 2–1024 tokens, the Swift 16/48-head, 128-dimensional geometry, scalar
gates and FP32 recurrent state. Decode, image embeddings/positions, verification
capture/rewind, masked or ragged input, training, sharding and other unsupported
calls keep the original path. Admitted dispatch errors propagate; they do not
silently retry different math. Mean-RMS q/k normalization and the local sigmoid
output gate are unchanged.

`--qwen-packed-gdn-prefill` enables it for direct runtime use;
`--no-qwen-packed-gdn-prefill` disables it. Old catalogs default to `false`.
Disabled mode preserves the existing Prompt Cache contract. Enabled mode adds a
separate `qwenPackedGDNPrefill: m5-v1` contract, preventing cross-mode reuse.

On the M5 Pro/64 GiB test machine with MLX/MLX Metal 0.32.2, off/on integration
checks matched every output token, complete logits row and captured logical
state for Code128, Code4096 and Traditional Chinese2153 (128 greedy output tokens
each), both with `MLX_ENABLE_TF32=0` and with the App-default environment. Separate
short checks covered Memory/Disk reuse, cross-mode cache rejection, injected
Prefill error/cancellation ownership and recovery, and forced first-token EOS.
These are bounded regression checks, not broad quality, sampled-decoding,
maximum-context or new full-model image qualification.

## Warnings and accessibility

English, Traditional Chinese and Simplified Chinese cover all new labels, help,
validation messages, dependency explanations and lifecycle messages. Number
controls include a text field and bounded stepper; controls have accessibility
labels and stable identifiers. The section labels the experiments explicitly:
pread may be slower than mmap, and fused SDPA can change rounding and generated
text. Neither speed nor full-model quality improvements are claimed.

## Tests

`QwenFlashSettingsTests` covers defaults, old preference/catalog decoding,
per-model save/load, isolation, range/overflow checks, inactive cache values,
wave dependencies, MTP independence, reset/locking, localized copy, rendering and
Swift catalog output. `SortedExpertSettingsTests` covers sorted expert prefill defaults, persistence,
inactive states, catalog output and three-language rendering for both Qwen models.
`PackedGDNSettingsTests` adds Swift-only defaults,
persistence, MTP inactivity, reset/locking and three-language rendering;
`runtime/tests/test_qwen_packed_gdn.py` covers configuration, layer byte parity,
instance isolation, fallback, image/capture routing and interrupted state ownership.
`MemoryPlanningTests` also covers row-cache accounting and
wave capacity semantics. The render test works without an installed model.

```sh
make test
# Capture screenshots and Swift-generated JSON for a cross-language check:
WHALLM_QWEN_UI_ARTIFACTS="$PWD/scratch/qwen-flash-ui" make test
PYTHONPATH=runtime .venv/bin/python Scripts/validate_qwen_flash_app_catalog.py \
  scratch/qwen-flash-ui/catalogs
```

The App CI runs the full Swift test suite on macOS, validates all localization
files and consumes twenty-three Swift-generated catalogs with the actual pinned Python
runtime parser. Rendered previews cover default, enabled and locked states in
three languages. Rendering checks do not replace interactive assistive-technology
or end-to-end user acceptance testing. No full model weights are loaded.

The two newer QSA throughput controls and their limits are documented in
[Qwen Next throughput](qwen-next-throughput.md).
