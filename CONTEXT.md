# Project context

This project builds a memory-bounded Apple Silicon runtime for
`DeepSeek-V4-Flash-0731`, `DeepSeek-V4.1-Flash`, and
`Qwen3.8-Flash-Next`.

## Ubiquitous language

- **checkpoint**: The pinned Hugging Face model revision and its safetensors shards.
- **common tensor**: A tensor that the runtime keeps resident. It is not a routed expert tensor.
- **routed expert**: One model-specific expert. DeepSeek uses `w1`, `w2`, and `w3`.
  Qwen uses fused `gate_up` and `down`.
- **expert blob**: The canonical packed bytes for one routed expert.
- **repack plan**: The complete mapping from checkpoint byte ranges to installed model byte ranges.
- **installed model**: A verified local directory produced from a repack plan.
- **API model ID**: The fixed, case-sensitive API name for one model kind.
- **Alias**: An optional, case-sensitive request name for an API model ID. An Alias does not
  change the API model ID.
- **manifest**: The JSON file that defines an installed model and its integrity metadata.
- **slot**: A fixed Metal-visible memory region that can hold one expert blob.
- **main model**: The target model that produces or verifies output tokens. It excludes
  auxiliary speculative decoding modules such as DSpark.
- **DSpark**: The optional speculative decoding module stored under `mtp.*`.
- **N-gram store**: Qwen FP8 N-gram rows stored in `ngram.bin` for read-only row lookup.
- **Engram store**: DeepSeek V4.1 FP8 embedding rows and E8M0 scales stored in
  layer-specific files for read-only row lookup.
- **model kind**: The stable identity that selects a model support package.
- **model support package**: The installation rules, model loading, conversation format,
  and state operations needed to support one model kind.

## Prompt cache compatibility and memory

Ordinary disk prompt caches use compatibility contract 2. Earlier caches are
ignored and rebuilt on demand because their saved token lists could omit an EOS
that the main model had already consumed. The installed model is unchanged;
DSpark uses a separate cache contract and is unaffected by this invalidation.

The in-memory entry limit counts requests. An ordinary request can retain up to
three target states; a V4.1 DSpark request retains one target-and-draft snapshot.
The App estimates these retained states within the configured cache budget,
except that the runtime always keeps the newest state even if it exceeds that
budget. It also allows for two in-flight ordinary snapshots before eviction.
A reused prefix can make the first snapshot nearly full-sized, even when
layer-major prefill is enabled. These are conservative capacity estimates, not
measured peaks or a process-wide memory limit.
