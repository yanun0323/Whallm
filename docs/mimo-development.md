# MiMo-V2.6-Flash-RL development status

**Development support, not complete P0–P5 acceptance.** A complete, losslessly
preserved installed artifact, text backbone and native still-image path now exist.
Full-checkpoint English/Traditional Chinese generation, red/blue image questions,
streaming tool-call parsing and a tool-result round trip passed locally.
Bounded PDF/DOCX/PPTX/XLSX and native PCM WAV input are connected to API/App
attachments. Two spoken-number audio fixtures passed full-checkpoint HTTP checks.
Video, audio/video synchronization, Agent UI, full reference parity, general document-format acceptance
and App end-to-end acceptance remain unfinished. The complete public artifact is verified and pinned.
The current source is locally packaged; App/ZIP signatures and isolated three-language startup passed.
The existing three model packages and their defaults are unchanged.

## Phase status

| Phase | Status | Remaining acceptance work |
| --- | --- | --- |
| P0: contract/reference audit | Partial | Image/audio preprocessing and independent F32 encoder equations checked; BF16/CUDA, routing and full-model reference validation remain |
| P1: target backbone | Partial | Complete artifact, text/image smoke and tool-result round trip passed; full reference and App/API acceptance remain |
| P2: multimodal inputs | Partial | Ordered image/audio/document parts, bounded assets, prepared embeddings and cache isolation implemented; video/synchronization remain |
| P3: multimodal encoders | Partial | Native ViT and AudioTokenizer/RVQ/patch encoder implemented; video/synchronization and broader reference acceptance remain |
| MCP parallel work | Partial | Official-SDK stdio/Streamable HTTP client and single-use approvals tested; broker, Agent loop, credentials/OAuth and App integration remain |
| P4: product integration | Partial | App attachments and bounded PDF/Office semantic extraction implemented; broader document acceptance, other media, Agent interface and end-to-end acceptance remain |
| P5: acceptance/package | Partial | Local App/ZIP signatures and isolated startup passed; full model, modality/security/resource and App end-to-end acceptance remain |

The descriptor advertises text, still-image input, bounded PCM WAV/document input and ready-expert decoding.
Audio generation and draft execution remain disabled. Neither small fixtures nor
short complete-model smoke tests establish model quality, speed or a usable 1M context.

## Pinned sources

- Model: `XiaomiMiMo/MiMo-V2.6-Flash-RL`
- Upstream checkpoint revision: `5711b268169967567844e1e560e8a3966da959b1`
- Public installed-artifact revision: `a25b1711c5c9f97e0562f39c9cd9f043f6be3f09`
- Original audit/reference revision: `3b38d063180c3e4aed9691fdc735f3d10b266ee4`
- HF metadata comparison found unchanged weights, main configs, model code and
  tokenizer; differences are the README, technical report and DFlash config.
- Weight-conversion reference: SGLang
  `0c53fec4768a46a003a7afe27c938469db956368`,
  [`python/sglang/srt/models/mimo_v2.py`](https://github.com/sgl-project/sglang/blob/0c53fec4768a46a003a7afe27c938469db956368/python/sglang/srt/models/mimo_v2.py)
- [Model config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/3b38d063180c3e4aed9691fdc735f3d10b266ee4/config.json)
- [Model implementation](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/3b38d063180c3e4aed9691fdc735f3d10b266ee4/modeling_mimo_v2.py)

## Checkpoint audit

`Scripts/audit_mimo_checkpoint.py` uses Hugging Face Hub metadata/range requests.
It does not download complete weight shards or execute checkpoint Python code.
It checks all indexed tensors, duplicate names, tensor byte sizes and offsets,
index/header coverage, the 47 expert-bearing layers, and the native MXFP4 layout.
It also reads the separate AudioTokenizer and DFlash headers.

```sh
PYTHONPATH=runtime:. .venv/bin/python Scripts/audit_mimo_checkpoint.py \
  --output scratch/mimo-checkpoint-audit

# Repeat validation without network access:
PYTHONPATH=runtime:. .venv/bin/python Scripts/audit_mimo_checkpoint.py \
  --output scratch/mimo-checkpoint-audit --offline
```

Verified metadata: 65 indexed shards, 67 including the two companion files,
73,081 indexed tensors, and 12,032 routed experts. Each expert occupies
13,369,344 native bytes including scales. This is not a RAM or throughput result.

### QKV installation trap

The QKV weights are TP=4 interleaved, not a simple concatenation of all Q, K, V.
The FP8 scale grid is separately padded in each source shard:

```text
checkpoint: [Q0 K0 V0] [Q1 K1 V1] [Q2 K2 V2] [Q3 K3 V3]
                  | decode FP8 using shard-local 128x128 scales
canonical:  [Q0 Q1 Q2 Q3] [K0 K1 K2 K3] [V0 V1 V2 V3]
```

For global attention, the checkpoint has 13,568 QKV rows and 108 scale rows.
A single global block grid would expect only 106 scale rows and is incorrect.
`mimo/weights.py` decodes source shards separately before deinterleaving. Common
QKV is kept as BF16 rather than requantized; routed experts remain packed MXFP4.

MLX's FP8 byte decoder also maps E4M3FN reserved NaN bytes to finite numbers.
The converter rejects these codes and invalid scales before conversion.

### Unresolved/reference-specific issues

- Main `processor_config` and `preprocessor_config.json` disagree on pixel limits.
  The pinned SGLang processor explicitly uses main `processor_config`; actual
  still-image preprocessing now follows and is checked against that processor.
  This does not validate its audio/video pipeline.
- HF routing code uses FP32 inputs while SGLang CUDA uses BF16 operands with FP32
  output. The prototype implements the latter arithmetic, checked against an
  independent small fixture, not a full CUDA reference run.
- Main weights contain three MTP layers; the separate DFlash companion has five
  layers. Neither is enabled. DFlash config in the original audit revision had a
  strict-JSON trailing comma; the installed revision contains the upstream fix.

## Native weight parity

`Scripts/validate_mimo_weights.py` fetches about 66 MiB: one expert's six tensors
and one global-attention QKV tensor plus scales. It rejects ignored or incorrect
HTTP byte ranges before reading a response body. The rest of the checkpoint is
not downloaded.

The check uses an isolated PyTorch CPU environment for the reference, not an App
dependency. It executes only three reviewed, SHA-256-pinned SGLang conversion
functions, not the remote model module. The captured reference is the canonical
BF16 tensor **before SGLang's subsequent FP8 requantization**; matching it is not
an assertion of end-to-end CUDA parity.

Example setup (use the project's existing Python environment for the audit):

```sh
mkdir -p scratch/mimo-reference
.venv/bin/python -m venv scratch/mimo-reference/venv
scratch/mimo-reference/venv/bin/python -m pip install \
  --only-binary=:all: torch==2.10.0 numpy==2.3.5
curl -fL \
  https://raw.githubusercontent.com/sgl-project/sglang/0c53fec4768a46a003a7afe27c938469db956368/python/sglang/srt/models/mimo_v2.py \
  -o scratch/mimo-reference/mimo_v2.py

MLX_ENABLE_TF32=0 PYTHONPATH=runtime:. .venv/bin/python Scripts/validate_mimo_weights.py \
  --audit scratch/mimo-checkpoint-audit \
  --output scratch/mimo-weight-parity \
  --reference-source scratch/mimo-reference/mimo_v2.py \
  --reference-python scratch/mimo-reference/venv/bin/python
```

Use `--offline` with the same output/reference paths to reuse the recorded slices.
Outputs include source-slice hashes, the reference hash and a validation report.

Validated on the local Apple Silicon development machine:

- The real GA QKV tensor (13,568×4096) matches the captured BF16 reference bitwise.
- Three actual expert projections pass `rtol=2e-4, atol=2e-5` against independent
  MXFP4 decoding and FP32 matrix multiplication; observed maximum absolute errors
  are approximately 1.68e-7, 1.42e-7 and 8.57e-8.
- This establishes those weight paths only, not complete checkpoint inference.

## Complete installed artifact

`Scripts/convert_mimo_model.py` reads the already downloaded, pinned HF snapshot.
It never overwrites an existing destination or changes the source. `.partial`
conversion can resume at the completed raw-copy phase; an interrupted raw-copy
phase is recopied. It processes source shards with four bounded-copy workers and
converts common FP8 tensors individually, without materializing complete BF16 experts.

```sh
PYTHONPATH=runtime:. .venv/bin/python Scripts/convert_mimo_model.py \
  --source "$HOME/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/5711b268169967567844e1e560e8a3966da959b1" \
  --output "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4"

# Independent source-byte reconstruction verification, without the HF snapshot:
PYTHONPATH=runtime:. .venv/bin/python Scripts/convert_mimo_model.py \
  --output "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" --verify

PYTHONPATH=runtime:. .venv/bin/python Scripts/validate_mimo_install.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --report scratch/mimo-install/text-smoke.json
```

The local artifact has 151 payload files totaling **193,730,445,975 bytes**
(180.426 GiB), excluding `manifest.json`. Format 4 distinguishes 48 backbone/KV
layers from 47 expert-cache layers. Every one of the 90 upstream files is retained
whole or exactly reconstructible; all 64 repacked source shards passed original
SHA-256 reconstruction. Original Git blob IDs and LFS SHA-256 values are checked.

- `experts/`: all 12,032 native MXFP4 experts in the existing slot layout.
- `common.bin`: 331 prepared target-backbone tensors, BF16/F32.
- `vision/common.bin`, `audio/common.bin`, `mtp/common.bin`: all 364 vision,
  95 audio-patch and 36 prepared MTP tensors.
- `checkpoint/common.bin` and `checkpoint/headers/`: exact original common
  payload bytes, including FP8 weights/scales, and original shard headers.
- `checkpoint/`: all 26 remaining upstream files, including original MTP,
  AudioTokenizer, DFlash, mask embedding, configs, code, tokenizers, model card,
  report and image. Preserved Python and pickle files are not executed by conversion.
- `checkpoint-map.json`: source provenance and byte reconstruction map.
- `preservation.json`: completed source-preservation checks.

The converter hashes every artifact file before finalizing it. The full-model
smoke produced `Hello.` and `你好。` and recovered after closing a generation
iterator. It also exposed an F32-sink/BF16-attention dtype mismatch, now fixed and
covered by a real-attention-shape BF16 regression test. This is not full reference
parity, a controlled performance benchmark, or multimodal acceptance.

Swift reuses the installed-artifact downloader and repair path, with explicit model
identity checks. The public repository
[`Yanun/MiMo-V2.6-Flash-RL-MXFP4`](https://huggingface.co/Yanun/MiMo-V2.6-Flash-RL-MXFP4)
contains the complete artifact. Whallm pins immutable artifact revision
`a25b1711c5c9f97e0562f39c9cd9f043f6be3f09`; installation requires no conversion.
Anonymous verification checked inventory, all sizes/digests (LFS metadata plus
Git-file downloads), and the immutable manifest. This was **not** another full
180.426 GiB download or a fresh App installation/repair acceptance run.

## Target backbone

`runtime/deepseek_v4_ssd/mimo/model.py` implements dense first-layer FFN, hybrid
SWA/global attention, partial RoPE, sink logits, V scaling, routing and the
existing packed expert compute path. Expert-cache indices exclude the dense
first layer; model KV state includes it.

The model accepts mlx-lm's `input_embeddings`; prepared still-image features now
use this path under the existing generation owner/MLX stream. Media requests disable
speculative decoding, layer-major Prefill, and Prompt Cache acquire/store.

Tests cover independent attention/router references, asymmetric QK/V dimensions,
chunked prefill, sliding-window rollover, cache continuation/cloning, cancellation,
real on-disk expert slots/eviction and mlx-lm's final-prompt-token embedding path.
Small fixtures are intentionally separate from complete model acceptance.

### Layer metrics and final-layer work reduction (2026-09-23)

An opt-in MiMo profiler now records all48 layers with host/sync subcomponent,
expert I/O/wait, owner CPU and memory diagnostics. The bounded Prefill path also
skips the last layer's discarded attention output and MoE, preserving only KV
state; ordinary forward/Decode and media remain unchanged. Code1K/128 and
Novel1K/32 retain exact outputs and save2.179/1.693GiB of logical Prefill reads.
Severe timing drift prevents a speedup claim; reader/cache-policy candidates
were not adopted. App defaults remain unchanged. Python634 tests pass. These
runtime changes were locally packaged on2026-09-23; Swift172 tests completed with
three skips and no failures, including the local MiMo manifest check. Both the
App and extracted ZIP passed signatures, SDK27.0 and three-language isolated
startup checks; relocated Python/MLX also ran with project/Homebrew access denied.
The build is ad-hoc signed, not notarized or publicly released. See
[metrics, reproducible commands and limits](mimo-runtime-profiling.md).

### Bounded layer-major text Prefill (2026-09-23, initial implementation)

The measurements below precede the final-layer work reduction above.

MiMo now supports **Use layer-major prefill** without changing the attention or
expert-compute chunk sizes. Each layer processes the chunks in a bounded block
before moving to the next layer, reusing the existing expert slots. Hidden-state
blocks are at most 4096 tokens, rounded down to a multiple of the selected chunk
size; an explicitly larger chunk remains one block. Each chunk's output and KV
state are evaluated before slots can be reused. No extra whole-layer expert
allocation, quantization, or change to Decode is introduced.

The App default remains **off**, and saved settings are preserved. Enable the
existing switch in MiMo's advanced settings and reload the model to try it.
The existing `layer_major_prefill` runtime flag and threshold (default 1024)
control the path. Standalone CLI/`RuntimeConfig` defaults already set that flag
to true; it now takes effect for MiMo. Use `--no-layer-major-prefill` in the CLI,
or `layer_major_prefill=false` in a runtime configuration, for the old path.
Media requests still use chunked Prefill. MiMo does not gain batched expert
prefill, next-layer prefetch, Prompt Cache, or a calibrated memory estimate.

On M2 Max / 64 GiB, with 771 slots and four readers, Code 1K/128 retained the
exact output token hash and reduced logical Prefill expert reads from 430.11 to
140.20 GiB (67.4%). Novel 1K/16 also retained its output hash and reduced reads
from 307.58 to 111.10 GiB. **These are I/O results, not a stable speed guarantee:**
Code first-token measurements ranged from 77.89–130.60s on the old path and
32.59–48.75s on the candidate; the Novel candidate was slower overall
(113.28s vs 67.14s). Timing varied substantially even with identical Decode
counters; its cause was not isolated. OS caches were not flushed, no swap was
observed, and longer contexts were checked only with small fixtures. A separate
larger-chunk candidate changed the output and was not adopted.

Python 628 tests pass; Swift 172 tests complete with four skips and no failures.
That validation preceded packaging. The Prefill path is now included in the
2026-09-23 local build described above; it has not been publicly released.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH=runtime:. .venv/bin/python -m unittest discover \
  -s runtime/tests -p 'test_mimo*.py' -v
make test-python
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer make test
```

The full Python regression includes existing, uncommitted Qwen work. A process-local
`DEVELOPER_DIR` is needed on this development Mac because the selected Command Line
Tools installation cannot resolve XCTest; the system Xcode selection is not changed.

## Still-image contract

- Authenticated `POST /api/assets` accepts a raw PNG/JPEG/WebP body and returns
  `id`, `sha256`, `bytes`, `mime_type`, and expiry. `DELETE /api/assets/{id}` removes it.
  `GET /api/capabilities` reports model input types and current media limits.
- Chat/Responses content preserves text/image order, including tool-result images:
  `[{"type":"text","text":"Describe"},{"type":"image","file_id":"file-…"}]`.
  Inline image data URLs are accepted within the existing **1 MiB JSON-body limit**;
  use asset uploads for larger images. Remote and local file URLs are rejected.
- Limits: 8 MiB/asset, 32 MiB/request, eight images, 1,048,576 decoded pixels/image,
  2048 media tokens. Animated images are rejected. Assets expire after 900 seconds;
  the store permits 64 files/64 MiB and four simultaneous uploads.
- Endpoints follow the server's existing optional API-key policy. Asset ownership
  is a **single server principal**, not multi-tenant session isolation. Server
  shutdown removes its temporary store.
- App attachments are private local files, hash-checked on reuse; history stores
  references rather than image bytes or expiring server IDs. Clear chat removes
  referenced images; unfinished selections are removed when leaving the view.
  Uploads are renewed each turn and deleted best-effort afterward (TTL fallback).
  Crash-orphan cleanup, visual/VoiceOver testing and full App acceptance remain open.
- Vision follows the pinned SGLang equations, including column-window ordering,
  **denominator sinks and an RMSNorm merger**. The HF sketch is not interchangeable.
  All 364 prepared vision tensors load strictly. No CUDA parity is claimed.

Reproduce the processor/encoder checks with an isolated reference environment:

```sh
mkdir -p scratch/mimo-vision
curl -fL https://raw.githubusercontent.com/sgl-project/sglang/0c53fec4768a46a003a7afe27c938469db956368/python/sglang/srt/multimodal/processors/mimo_v2.py \
  -o scratch/mimo-vision/processor.py
# Add pillow==12.3.0 to the isolated PyTorch environment described above.
MLX_ENABLE_TF32=0 PYTHONPATH=runtime:. .venv/bin/python Scripts/validate_mimo_vision.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --output scratch/mimo-vision/results --processor-source scratch/mimo-vision/processor.py \
  --reference-python scratch/mimo-reference/venv/bin/python
PYTHONPATH=runtime:. .venv/bin/python Scripts/validate_mimo_media.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --report scratch/mimo-vision/integration.json
```

The reference fixture gave patch maximum error 7.15e-7 and F32 vision relative L2
9.37e-6. Runtime BF16 versus the F32 reference had relative L2 0.03914; this is a
precision-budget check, **not bitwise BF16 parity**. The full model answered `Red`
and `Blue` for identical placeholder IDs with different image identities and stored
no media Prompt Cache. A forced `lookup_weather` call parsed in both streaming and
non-streaming modes; after a fixture result, the model answered `23°C`. These are
short functional checks, not quality or performance benchmarks.

## App attachment UI

In **Playground → Chat**, start the server and select MiMo. **Attach files** offers
separate image, document and WAV pickers; **Choose files…** (`⇧⌘A`) accepts all
implemented types. Local-file drag-and-drop uses the same bounded, atomic import
path. Web links are rejected, and a failed batch removes only its newly imported
copies. Changing to a text-only model shows an incompatibility warning rather
than silently dropping attachments.

Pending files appear in a horizontally scrollable strip with image thumbnails or
document/audio icons, filename, format, size and a labeled remove action. Full
filenames are available in tooltips/accessibility names; missing files are marked
unavailable. The strip has bounded height so multiple attachments do not displace
the editor. Keyboard focus scrolls the selected remove control into view.

The usage indicator includes **chat history**, not just pending files. **Attachment
limits** shows per-type counts, file/duration/page limits, shared media-token limits
and where files are sent. The server still validates actual decoded content.
**Clear Chat** also works for an unsent, attachment-only draft. The native picker
is cancelled on navigation, avoiding late imports into an abandoned composer.

Requests show separate **Uploading attachments**, **Preparing response** and
**Generating** stages. **Cancel request** works during uploads or generation;
completed upload counts advance only after returned size/SHA-256 validation.
The request state belongs to the chat session and survives navigation. Sending
rechecks file integrity and budgets before appending the user's message.

English, Traditional Chinese and Simplified Chinese native composer previews
cover narrow/wide, empty, populated, full, text-only and missing-file states in the
App's dark appearance. Swift tests cover atomic imports, model gating, history
budgets, attachment-only sends, stage transitions and cancellation. These are
component/render tests, **not full App/VoiceOver or packaged end-to-end acceptance**.
No audio playback/recording, document rendering, video UI or MCP approval UI is
added by this increment.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  WHALLM_ATTACHMENT_PREVIEWS="$PWD/scratch/mimo-ui/previews" \
  make test ARGS='--filter Chat'
```

## Audio input contract

Native audio input uses the preserved AudioTokenizer and prepared audio-patch
weights, **not external ASR or a transcript substituted for audio**. The runtime
adds no Torch, torchaudio, FFmpeg or codec-service dependency.

- Upload a raw `audio/wav` body to `/api/assets`, then send
  `{"type":"input_audio","file_id":"file-…"}` in Chat/Responses content.
  Inline Chat-style `{"type":"input_audio","input_audio":{"format":"wav","data":"…"}}`
  is also accepted, subject to the existing **1 MiB JSON-body limit**. No URLs.
- Only **24 kHz, 16-bit PCM WAV**, mono or stereo, longer than 20 ms and at most
  **30 seconds/file; two audio parts/request**. No resampling, compressed formats,
  recording, audio output or automatic truncation. Malformed/truncated or duplicate
  WAV format/data chunks are rejected. Stereo channels are averaged.
- Existing 8 MiB/file, 32 MiB/request, 2048 combined image/audio-token budget,
  asset authentication/ownership/TTL and cancellation apply. Each clip is encoded
  independently. Oversized audio is rejected before audio-encoder GPU work/SSE starts.
- Text/image/audio/document ordering, including user and tool-result history, is
  retained. Native placeholders are `151673`, repeated `151669`, then `151674`.
  Audio digests and modality participate in input identity. Media bypasses Prompt
  Cache acquire/store and unsupported speculative/layer-major paths.
- App selection/import/history/reupload uses the same private attachment files.
  Audio gets a filename/icon preview with no autoplay or in-process audio decoder.
  The picker explains the format limits; the server performs authoritative parsing.
  App selection allows eight attachments total, including two audio clips and four
  documents. Visual/VoiceOver and full App end-to-end validation remain open.

Equations follow pinned SGLang
[`models/mimo_audio.py`](https://github.com/sgl-project/sglang/blob/0c53fec4768a46a003a7afe27c938469db956368/python/sglang/srt/models/mimo_audio.py)
and [`processors/mimo_audio.py`](https://github.com/sgl-project/sglang/blob/0c53fec4768a46a003a7afe27c938469db956368/python/sglang/srt/multimodal/processors/mimo_audio.py):
960-point periodic Hann, hop 240, reflect-centered magnitude STFT, 128 HTK bands,
log/clamp at 1e-7; 24 encoder layers with alternating `[128,0]` local and **noncausal
global** attention, the layer-3 skip connection, learned stride-2 pooling, 20 RVQ
codebooks, last-code group padding, six group-local Qwen2 layers and projection.
The HF sketch's all-causal mask is not interchangeable. Loading strictly matches
389 encoder inference tensors and all 95 patch tensors; decoder/training-only
weights remain preserved but are not loaded. Encoders are request-owned, not cached.
Runtime rounds the F32 codebooks to BF16 before F32 RVQ, matching SGLang's cast order.

Independent PyTorch CPU F32 equations on a deterministic noise/chirp fixture gave
mel max error **9.03e-5**, encoder relative L2 **5.72e-6**, patch relative L2
**3.46e-5**, and identical F32 RVQ codes. Runtime BF16 versus this F32 reference had
encoder/patch relative L2 **0.03182 / 0.10103** and **66.18% code agreement**; patch
comparison uses the same reference codes to isolate patch arithmetic. These BF16
observations are **not a parity pass**; CUDA/BF16 and general audio quality remain
unvalidated. A 30-second silence boundary produced 751×20 codes and 188×4096 finite
features; that is an execution-limit check, not audio-quality evidence.

```sh
# Use the isolated reference environment from the weight-validation section.
scratch/mimo-reference/venv/bin/python -m pip install torchaudio==2.10.0 safetensors==0.7.0
MLX_ENABLE_TF32=0 PYTHONPATH=runtime:. .venv/bin/python Scripts/validate_mimo_audio.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --output scratch/mimo-audio/reference-results \
  --reference-python scratch/mimo-reference/venv/bin/python

mkdir -p scratch/mimo-audio
say -v Samantha -r 145 -o scratch/mimo-audio/42.aiff 'The secret number is forty two.'
say -v Samantha -r 145 -o scratch/mimo-audio/73.aiff 'The secret number is seventy three.'
for n in 42 73; do
  afconvert -f WAVE -d LEI16@24000 -c 1 scratch/mimo-audio/$n.aiff scratch/mimo-audio/$n.wav
done
PYTHONPATH=runtime:. .venv/bin/python Scripts/validate_mimo_audio_http.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" --report scratch/mimo-audio/http.json \
  --wav42 scratch/mimo-audio/42.wav --wav73 scratch/mimo-audio/73.wav
```

The complete checkpoint answered **`42` via Chat and `73` via Responses**, storing
zero media Prompt Caches despite two entries being available. The input is synthetic
macOS speech, not a general ASR/quality benchmark. Both scripts refuse to overwrite
existing evidence paths. Checkpoint/artifact bytes were not modified.

## Document input contract

Documents use the same `/api/assets` upload, ownership, TTL and deletion path.
Send `{"type":"input_file","file_id":"file-…"}` in Chat/Responses content;
`{"type":"file","file_id":"file-…"}` and nested `file.file_id` also work.
User messages and tool-result history can contain documents. No document URL or
inline file download is performed. Only MiMo advertises `documentInput`.

- **PDF:** page text plus a PNG of every page, including pages without a text layer.
  pypdfium2 5.10.1 renders at up to 1.5× page coordinates, capped at 768 pixels on the
  longest side. Page labels and the source SHA-256 accompany the content. Interactive
  forms and encrypted files are rejected; no PDF actions/scripts are invoked.
- **DOCX:** body text, table row/cell labels, referenced raster images, headers,
  footers, comments and notes. This is semantic extraction, **not Word pagination**.
- **PPTX:** declared slide order, text/tables, raster images and speaker notes.
- **XLSX:** sheet/cell coordinates, stored strings/numbers and drawing anchors.
  Values are **unformatted**, including hidden sheets/cells; date serials are not
  reformatted. Formulas are shown with their cached value (possibly absent/stale),
  never evaluated or refreshed.
- Office layout, fonts and master/layout inheritance are not rendered. Known
  unsupported charts, SmartArt, alternate drawings, equations and embedded media
  are rejected rather than silently dropped. Macros, ActiveX and embedded objects
  are rejected. External hyperlinks are not fetched; external content references
  are rejected. This is a deliberately limited subset, not full Office fidelity.

Limits: **4 documents/request**, 4 PDF pages / PPTX slides / XLSX sheets per file,
64 KiB total extracted text including source labels, 512 ZIP entries and 32 MiB
expanded ZIP bytes per file. The existing 8 MiB/file, 32 MiB/request, eight-image,
pixel and 2048-media-token limits also apply after expansion. A four-page PDF can
still exceed the aggregate media-token budget. Oversized input is rejected, not
truncated. DOCX has no inferred page count. App selection allows eight total
attachments, including at most four documents; the server revalidates all limits.

Parsing runs in at most two child processes, with 30-second wall/10-second CPU
limits, 16 MiB disk-backed output, disabled core dumps and a polled 768 MiB RSS
budget (250 ms sampling plus a final peak check). Cancellation kills and reaps the
worker. **RSS monitoring can overshoot and is not a hard memory/OS sandbox.**
The decoder is not a general hostile-file sandbox. Native parser hardening and
full packaged-App document workflows remain separate acceptance gates. ZIP traversal, duplicates,
symlinks, encrypted entries, DTDs and XML entities are rejected; no archive is
extracted into the filesystem.

The App stores documents as private, hash-checked attachments and previews them
with a document icon, not an in-process PDF/Office renderer. Its import check is
only a size/signature envelope check; parsing is authoritative on the server.

```sh
PYTHONPATH=runtime:. .venv/bin/python -m unittest runtime.tests.test_documents runtime.tests.test_media_api
PYTHONPATH=runtime:. .venv/bin/python Scripts/validate_mimo_documents.py \
  --model "$HOME/.dsmodel/mimo-v2.6-flash-rl.dsv4" \
  --report scratch/mimo-documents/smoke.json
```

The complete checkpoint answered `42` from a synthetic DOCX over Chat and `73`
from a different **raster-only invoice PDF** over Responses. The latter has no extractable text;
its answer uses the existing native ViT path. Synthetic fixtures cover all four
formats, unsafe packages, budgets and worker cleanup; they do not establish
arbitrary Office fidelity or general OCR quality. Latest regression including audio: **624 Python
tests; 171 Swift tests, three skipped, zero failures**. Local packaging is now verified as described below.

## Opt-in MCP client (not yet an App Agent)

`runtime/deepseek_v4_ssd/mcp_client.py` uses the official Python SDK 1.30.0 for
initialize/discovery, stdio and Streamable HTTP. Each call requires an explicit,
expiring, single-use approval; even `readOnlyHint` tools are not auto-approved.
Arguments are validated in bounded-time subprocesses; external schema references
are rejected. Declarations are refreshed before execution and changed declarations
invalidate approval. Cancellation/timeout consumes approval and never retries a
side effect; cancellation cannot undo a remote effect.

Commands use argv without a shell; configured subprocesses are **not sandboxed**.
Inference credentials are not forwarded. Remote HTTP requires HTTPS; redirects and
environment proxies are disabled. Names are deterministically server-namespaced.
Typed results are retained, not silently flattened into text. Inventory, proposal
and deserialized-result limits exist; the SDK transport does **not** yet provide a
hardened per-frame memory boundary. Resources/prompts are not fetched automatically.

Real local SDK servers test both transports, denial, schema rejection, credential
non-forwarding, cancellation and replay prevention. No Agent loop, broker endpoint,
OAuth/Keychain integration or approval UI is connected. Ordinary compatible APIs
continue to return tool calls without executing them.

## Local App packaging validation

The local build produces `dist/Whallm.app` and `dist/Whallm-macOS-arm64.zip`.
Both the App and a temporary ZIP extraction passed `codesign --verify --deep --strict`,
SDK 27.0 checks, and English/Simplified Chinese/Traditional Chinese startup while
access to the project's `.build` and the Swift resource bundle was denied.
Each launch stayed alive for the verifier's three-second observation and completed
L10n initialization from `Contents/Resources` before `Bundle.module`.

The relocated bundled Python passed MLX GPU, native audio preprocessing, MCP imports,
PDF decoding/rasterization and document child-worker checks with project and Homebrew
reads denied. The full document-worker coordinator also passed in a clean relocated
environment outside that sandbox: macOS refuses execution of the setuid `/bin/ps`
inside `sandbox-exec`, even with an allow-default profile. The RSS monitor was not
removed or weakened to bypass that OS restriction. This is not a hostile-file sandbox
or full App document-workflow acceptance.

This is a **local ad-hoc-signed build**, not a notarized release. No tag or artifact
upload was performed. The model artifact and runtime defaults are unchanged.

## Next acceptance gates

1. Implement video/AV synchronization; broaden audio formats, real-world quality and BF16/CUDA references.
2. Broaden document-format acceptance and complete the MCP broker/Agent loop, credentials and UI.
3. Add full-model reference, richer image fixtures, mixed-media/tool-result workflows,
   resource/cancellation acceptance and App accessibility/end-to-end checks.
4. Validate fresh pinned installation/repair and full packaged-App workflows; basic packaging checks above have passed.
   No tag, notarization or release is authorized by this development work.
