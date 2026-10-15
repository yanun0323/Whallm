# Project context

This project builds a memory-bounded Apple Silicon runtime for
`DeepSeek-V4-Flash-0731`, `DeepSeek-V4.1-Flash`, `Qwen3.8-Flash-Next`,
`MiMo-V2.6-Flash-RL`, and `Swift1.5-Qwen3.8-Flash-Next`. Qwen3.8 and Swift1.5 use
the native engine under their original model identities.

## Ubiquitous language

- **checkpoint**: The pinned Hugging Face model revision and its safetensors shards.
- **common tensor**: A tensor that the runtime keeps resident. It is not a routed expert tensor.
- **routed expert**: One model-specific expert. DeepSeek uses `w1`, `w2`, and `w3`.
  Qwen uses fused `gate_up` and `down`.
- **expert blob**: The canonical packed bytes for one routed expert.
- **repack plan**: The complete mapping from checkpoint byte ranges to installed model byte ranges.
- **installed model**: A verified local directory produced from a repack plan.
- **display name**: The model name shown on the Model page, shared by Chat, Status,
  Throughput and human-readable benchmark tables. It is not a request identifier.
- **API model ID**: The fixed, case-sensitive API name for one model kind. Requests
  and benchmark JSON retain this ID even when the display name changes.
- **Alias**: An optional, case-sensitive request name for an API model ID. An Alias does not
  change the API model ID or the App's display name.
- **manifest**: The JSON file that defines an installed model and its integrity metadata.
- **slot**: A fixed Metal-visible memory region that can hold one expert blob.
- **main model**: The target model that produces or verifies output tokens. It excludes
  auxiliary speculative decoding modules such as DSpark.
- **DSpark**: The optional speculative decoding module stored under `mtp.*`.
- **N-gram store**: FP8 N-gram rows stored in `ngram.bin` for read-only row lookup by the Qwen inference engine.
- **Engram store**: DeepSeek V4.1 FP8 embedding rows and E8M0 scales stored in
  layer-specific files for read-only row lookup.
- **model kind**: The stable identity that selects a model support package.
- **model support package**: The installation rules, model loading, conversation format,
  and state operations needed to support one model kind.
- **native engine**: The C++ and Metal inference engine in `Native/QwenNative`, built
  as `libWhallmQwenNative.dylib`. It does not use MLX. The MLX runtime is the other
  engine.
- **automatic expert cache**: The native engine's expert cache sized at load from the
  memory free at that moment, at most 8,192 slots.

## Shared Qwen engine, separate model kinds

`swift1.5-qwen3.8-flash-next` and `qwen3.8-flash-next` use the native engine.
Swift's API model ID is `swift1.5-qwen3.8-flash-next-mxfp4`.
Installation files come only from the pinned
`Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4` artifact. The upstream checkpoint
identity remains `ukisai/Swift1.5-Qwen3.8-Flash-Next` at
`0bd4fe22431372cdad1979267d3ab45aa7e6150a`.

The published manifest retains `qwen3.8-flash-next` as its schema label. Swift
and Python resolve that exact source identity to the new model kind without
rewriting the artifact. Installation folders, settings, Aliases and prompt
caches stay separate from the FP8 model. The new artifact includes its own MTP
weights; Swift never downloads the FP8 model's MTP. Both Qwen model kinds show
Native settings, with automatic expert cache, MTP and 8-bit resident weights on
for new or reset settings. Saved generation, read-worker, MTP and 8-bit choices
are retained; MLX-only tuning no longer applies.

Both Qwen model kinds accept still images through App Chat and the Chat/Responses
APIs. Each installation includes its own pinned `vision/common.bin` (897,864,704
bytes). The original published text manifest stays unchanged; the bundled
`QwenVision.json` defines the supplemental file, source ranges and tensor layout.
Verification includes this file. Existing installations repair only missing or
damaged files rather than redownloading intact text weights. Total installed
payloads are 126,189,355,659 bytes for Qwen FP8 and 127,714,522,478 bytes for Swift.

Image requests use request-owned vision weights, embeddings and T/H/W positions,
including QSA pooled-key positions. They support MTP on or off, bypass prompt
caching, and reset the engine after the request.
PNG/JPEG/WebP inputs retain the shared 8-image, 8-MiB/file, 32-MiB/request,
1,048,576-pixel/image and 2,048-media-token/request limits. Video, audio and
documents are not enabled for Qwen. Text-only inference does not load vision
weights. Short image checks are not broad quality or performance validation.

## Native engine model kinds

The original Qwen3.8 and Swift1.5 model kinds use the native engine. Their API
model IDs remain `qwen3.8-flash-next-fp8` and
`swift1.5-qwen3.8-flash-next-mxfp4`. The two `-native` preview model kinds are
removed. Each model keeps its existing installation, settings and Alias;
native prompt cache files are not compatible with MLX prompt caches.

The Python server keeps one interface for both engines. `NativeQwenRuntime`
(`runtime/deepseek_v4_ssd/native_qwen.py`) samples tokens, runs MTP rounds and keeps
the prompt cache; the engine returns logits. The App sends MLX-only runtime options at
their defaults for these kinds, and the runtime rejects any other value. New or reset
settings use the automatic expert cache, MTP and 8-bit resident weights.

## Prompt cache compatibility and memory

MLX disk prompt caches use compatibility contract 2. Earlier caches are
ignored and rebuilt on demand because their saved token lists could omit an EOS
that the main model had already consumed. The installed model is unchanged;
DSpark uses a separate cache contract and is unaffected by this invalidation.

For the MLX runtime, the in-memory entry limit counts requests. An ordinary
request can retain up to three target states; a V4.1 DSpark request retains one target-and-draft snapshot.
The App estimates these retained states within the configured cache budget,
except that the runtime always keeps the newest state even if it exceeds that
budget. It also allows for two in-flight ordinary snapshots before eviction.
A reused prefix can make the first snapshot nearly full-sized, even when
layer-major prefill is enabled. These are conservative capacity estimates, not
measured peaks or a process-wide memory limit.
