# Qwen Flash App controls

The Qwen model's Advanced Settings page now contains a **Qwen Flash experiments**
section. It exposes the existing opt-in runtime experiments without changing their
defaults or the installed-model format. It is not shown for DeepSeek models.

| App control | Saved preference | Runtime catalog field | Default |
| --- | --- | --- | --- |
| Experts per wave | `qwenExpertWaveSlots` | `qwen_expert_wave_slots` | `0` (off) |
| N-gram read backend | `qwenNgramIO` | `qwen_ngram_io` | `mmap` |
| N-gram row cache MiB | `qwenNgramCacheMiB` | `qwen_ngram_cache_bytes` | `0` (off) |
| QSA queries per chunk | `qwenQSAQueryChunk` | `qwen_qsa_query_chunk` | `4` |
| Use indexed QSA decode | `qwenQSAIndexed` | `qwen_qsa_indexed` | `false` |
| Use fused QSA SDPA | `qwenSparseSDPA` | `qwen_sparse_sdpa` | `false` |

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
- With mmap selected, the row-cache control is disabled and Python receives a zero
  cache budget. Its saved MiB value is retained and becomes active again with pread.
- Switching MTP on or off does not reset these experiments. No MTP proposal,
  acceptance, checkpoint or rollback algorithm changes are included here.

The memory overview counts the active packed-row budget in prefill/decode and
uses the non-whole-layer allocation path for waves. It does not reduce the total
expert-slot budget to the wave size, or claim that SDPA reduces peak memory.
These remain structural estimates, not measured peaks or hard limits.

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
Swift catalog output. `MemoryPlanningTests` also covers row-cache accounting and
wave capacity semantics. The render test works without an installed model.

```sh
make test
# Capture screenshots and Swift-generated JSON for a cross-language check:
WHALLM_QWEN_UI_ARTIFACTS="$PWD/scratch/qwen-flash-ui" make test
PYTHONPATH=runtime .venv/bin/python Scripts/validate_qwen_flash_app_catalog.py \
  scratch/qwen-flash-ui/catalogs
```

The App CI runs the full Swift test suite on macOS, validates all localization
files and consumes ten Swift-generated catalogs with the actual pinned Python
runtime parser. Rendered previews cover default, enabled and locked states in
three languages. Rendering checks do not replace interactive assistive-technology
or end-to-end user acceptance testing. No full model weights are loaded.

The two newer QSA throughput controls and their limits are documented in
[Qwen Next throughput](qwen-next-throughput.md).
