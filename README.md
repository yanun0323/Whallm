# Whallm

<p align="center">
  <a href="."><img height="160" src="Packaging/AppIcon.png" alt="Whallm"></a>
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-blue" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

Whallm runs large language models on your Apple Silicon Mac. It keeps the shared weights in memory and reads only the experts each token needs from your SSD, so it can run models much larger than your memory. It comes with a chat window and an OpenAI-compatible API.

Supported models: DeepSeek V4, DeepSeek V4.1, Qwen3.8, Swift1.5 Qwen3.8 and MiMo (preview).

## Benchmark summary

User-provided v1.1.13 measurements for **Swift1.5-Qwen3.8-Flash-Next**, using the Native engine. The summary below uses the separate **16K input / 512 output** tables; the full length sweep is listed later. These are different reported values, not averages.

| Chip | Cache mode | Slots | Prefill (tok/s) | Decode (tok/s) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | Automatic | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | Fixed | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | Automatic | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | Fixed | 2089 | 536.0 | 21.7 | 16.25 |

The source’s quick summary and 16K tables show different peak-memory values; this table uses the more precise values from the 16K tables. Older Qwen and DeepSeek results remain below, with their original versions.

> See the [full benchmark](#benchmarks) for details.

## Quick start

1. Download `Whallm-macOS-arm64.zip` from [GitHub Releases](https://github.com/yanun0323/Whallm/releases), unzip it and open `Whallm.app`.
2. Open **Model**, pick a model and select **Download Model**. The app checks free space first. If a download stops, you can resume it.
3. Open **Server** and select **Start Server**.
4. Open **Chat** and pick your model, or connect an API client as shown below.

The server listens on `http://127.0.0.1:11434`. A model loads the first time you use it. The server keeps one model loaded and runs one request at a time. If a model you just installed is missing from the Chat list, restart the server.

The [1.1.13 release notes](Packaging/ReleaseNotes/1.1.13.md) list the changes, upgrade steps and known limits.

## Requirements

| Item | Requirement |
| --- | --- |
| Mac | Apple Silicon, macOS 15 or later |
| Memory | 64 GiB recommended; actual use depends on the model and settings |
| Storage | A fast internal, Thunderbolt or USB4 SSD |
| Free space | The app shows how much each model needs, including partial downloads |
| Network | Needed to download models and app updates |

Model weights are not included in the app. DeepSeek V4.1 alone needs about **458 GiB** for its expert and Engram files, plus its other weights.

## Qwen models

Both Qwen models now use the Native engine (C++ and Metal, without MLX). Existing downloads, model names, API IDs and Aliases stay unchanged. The two `-Native` preview entries have been removed; clients using those IDs must switch to the original IDs below. MLX-only settings no longer apply, and MLX prompt caches are not reused.

### Swift1.5-Qwen3.8-Flash-Next

Added in 1.1.10.

- API model ID: `swift1.5-qwen3.8-flash-next-mxfp4`
- Source: `Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4`, revision `257cb509d72ecc07519be35a8b7558fde3b31b21`
- Download size: **127,714,522,478 bytes** (about **118.94 GiB**). This covers 59 text and MTP files plus **897,864,704 bytes** of image weights. Audit files are not downloaded.
- It uses the same engine as Qwen3.8 FP8 but keeps its own folder, settings, Alias and prompt cache.

### Qwen3.8-Flash-Next

- API model ID: `qwen3.8-flash-next-fp8`
- Qwen3.8 FP8 uses the same Native engine and settings as Swift1.5, with its own installation, Alias and prompt cache.

### Settings

- Both models expose generation, expert cache size, read workers, Prompt Cache, warmup, MTP and 8-bit resident weights. MLX memory guidelines, prefetch workers, cache eviction and the separate MTP cache size are no longer shown.
- **Advanced Settings** has **Size the expert cache automatically**, on by default. The engine then uses memory that is free when the model loads, up to about 20 GiB, and keeps at least 7 GiB or 15% of this Mac's memory for macOS and other apps. Turn it off to choose a fixed size. The other settings are generation, read workers, Prompt Cache, warmup, MTP and 8-bit resident weights.
- MTP is on by default. It usually speeds up output, but not for every prompt. Output can differ slightly from output with MTP off.
- **Use 8-bit resident weights** is on by default for new or reset settings; saved settings keep their choice. It keeps the weights that stay in memory, except experts, in 8 bits instead of BF16. This uses less memory and usually speeds up output, most with MTP on. Output can differ slightly from output with the setting off.

### Images

- Both Qwen models accept still PNG, JPEG and WebP images in Chat, `/v1/chat/completions` and `/v1/responses`.
- Images work with MTP on or off. Image requests do not use the prompt cache.
- Limits per request: 8 images, 8 MiB per file, 32 MiB in total, 1,048,576 pixels per image after resizing and 2,048 image tokens.
- New installs include the image weights. On an older install, select **Model → Verify and Repair**; it downloads only missing or damaged files. A full Qwen3.8 FP8 install is **126,189,355,659 bytes** (about **117.52 GiB**).
- Video, audio and document input are not available for Qwen.
- Native checks on M2 Max covered short text, Chat, Responses, streaming and images for both models. User-provided v1.1.13 Swift1.5 speed measurements on M2 Max and M5 Pro are listed below. Outputs can differ from MLX; these measurements do not establish broad quality or performance validation.

## Features

Whallm keeps shared weights in memory and reads the chosen experts from SSD. It also reads DeepSeek V4.1 Engram rows and Qwen N-gram rows only when needed.

| Model | Speed-ups |
| --- | --- |
| DeepSeek V4 | Layer-by-layer input processing, batched expert math, FP8 KV cache, optional ANE projection and DSpark |
| DeepSeek V4.1 | Layer-by-layer input processing, batched experts, packed KV and index caches, scoring only candidate indexes, CED input processing, ANE projection and DSpark |
| Qwen3.8 | Grouped experts during input processing, expert math as soon as each read finishes, QSA cache compression, next-layer prefetch and MTP |

## Benchmarks

### v1.1.13: Swift1.5 · Native engine

All rows below have a 512-token output limit and report 512 output tokens. Automatic cache sizing used 8,192 slots on these machines; fixed sizing used 2,089. Automatic sizing can choose a different size on another Mac or when less memory is free. The source does not record the prompt type, MTP or 8-bit settings, temperature, seed, cache state, complete hardware/software environment, source commit, raw runs or output token hashes. These are reference observations, not a controlled comparison with earlier versions.

**TTFT** is time to first token; **TPOT** is time per output token. **Prefill** and **Decode** are input and output tokens per second. **Total** is elapsed time; **Throughput** counts input plus output tokens per second. **Peak memory** is the reported process-memory peak, not Peak MLX or total Mac memory. The App samples physical footprint about every 10 ms; brief peaks may be missed. The source does not specify whether the measurement includes the App or only a standalone inference process.

#### M2 Max · Automatic · Full length sweep

| Slots | Input / Output | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | Total | Throughput (tok/s) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 4096 / 512 | 13.6s | 48ms | 302.0 | 20.9 | 38s | 121.1 | 29.94 |
| 8192 | 8192 / 512 | 27.4s | 48ms | 298.8 | 20.8 | 52s | 167.5 | 30.39 |
| 8192 | 16384 / 512 | 56.8s | 51ms | 288.6 | 19.5 | 1m23s | 203.6 | 31.27 |
| 8192 | 32768 / 512 | 2m3s | 55ms | 266.1 | 18.3 | 2m31s | 220.2 | 32.30 |
| 8192 | 65536 / 512 | 4m45s | 60ms | 230.3 | 16.8 | 5m15s | 209.7 | 34.45 |

#### M2 Max · Fixed · Full length sweep

| Slots | Input / Output | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | Total | Throughput (tok/s) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2089 | 4096 / 512 | 13.7s | 57ms | 299.3 | 17.5 | 42.8s | 107.6 | 15.17 |
| 2089 | 8192 / 512 | 27.3s | 57ms | 300.0 | 17.5 | 56.5s | 154.0 | 15.63 |
| 2089 | 16384 / 512 | 56.5s | 60ms | 290.2 | 16.6 | 1m27s | 193.6 | 16.50 |
| 2089 | 32768 / 512 | 2m2s | 62ms | 269.0 | 16.2 | 2m33s | 217.0 | 17.61 |

#### M5 Pro · Automatic · Full length sweep

| Slots | Input / Output | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | Total | Throughput (tok/s) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 1024 / 512 | 5.2s | 35ms | 197.6 | 28.6 | 23s | 66.7 | 29.44 |
| 8192 | 4096 / 512 | 8.4s | 38ms | 489.6 | 26.6 | 27.6s | 166.9 | 29.84 |
| 8192 | 8192 / 512 | 15.9s | 37ms | 515.9 | 26.9 | 34.9s | 249.6 | 30.23 |
| 8192 | 16384 / 512 | 31.2s | 39ms | 525.7 | 25.6 | 51.1s | 330.6 | 31.06 |
| 8192 | 32768 / 512 | 1m5s | 39ms | 505.8 | 25.5 | 1m25s | 392.2 | 32.06 |

#### M5 Pro · Fixed · Full length sweep

| Slots | Input / Output | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | Total | Throughput (tok/s) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2089 | 1024 / 512 | 5.4s | 41ms | 188.9 | 24.3 | 26.5s | 58.1 | 14.60 |
| 2089 | 4096 / 512 | 8.6s | 48ms | 475.9 | 20.7 | 33.3s | 138.3 | 15.01 |
| 2089 | 8192 / 512 | 16.4s | 45ms | 499.7 | 22.0 | 39.6s | 219.8 | 15.41 |
| 2089 | 16384 / 512 | 31.8s | 51ms | 514.7 | 19.5 | 58.1s | 290.8 | 16.22 |
| 2089 | 32768 / 512 | 1m6s | 50ms | 492.8 | 19.9 | 1m32s | 361.1 | 17.26 |

### Historical results: v1.1.7–v1.1.12

The following Qwen rows used the previous MLX engine, not the Native engine. Their Peak MLX values are not directly comparable with the process-memory peaks above.

Except for the image table, rows use **Code** input. The output limit is **512 tokens** for the Swift1.5 rows and **128 tokens** for the other rows. For those rows the build revision and cache state were not recorded, so treat these numbers as a reference, not as a controlled comparison between versions or speed-ups.

- **TTFT**: time until the first token appears.
- **Prefill**: input processing speed. **Decode**: output speed. Both are in tokens per second.
- **Peak MLX**: the most memory MLX used, in **GiB**. It is not the Mac's total memory use. Older app exports label this value GB, but it is computed in GiB.

#### M5 Pro

| Model | Version | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Swift1.5 Qwen3.8 | 1.1.12 | 2089 | 4096 | 6976.5 | 587.1 | 19.4 | 13.13 |
| Swift1.5 Qwen3.8 | 1.1.12 | 2089 | 8192 | 13911.4 | 588.9 | 19.8 | 13.86 |
| Swift1.5 Qwen3.8 | 1.1.12 | 2089 | 16384 | 29654.2 | 552.5 | 18.6 | 14.29 |
| Qwen3.8 | 1.1.11 | 2089 | 4096 | 8456.6 | 484.4 | 15.0 | 16.58 |
| Qwen3.8 | 1.1.11 | 2089 | 8192 | 14968.3 | 547.3 | 17.2 | 17.41 |
| Qwen3.8 | 1.1.11 | 2089 | 16384 | 32572.7 | 503.0 | 15.0 | 18.38 |
| DeepSeek V4 | 1.1.7 | 1152 | 1024 | 19112.3 | 53.6 | 7.7 | 22.77 |
| DeepSeek V4 | 1.1.7 | 1152 | 4096 | 24498.6 | 167.2 | 5.9 | 22.80 |
| DeepSeek V4 | 1.1.7 | 1152 | 8192 | 42137.5 | 194.4 | 6.8 | 22.83 |
| DeepSeek V4 | 1.1.7 | 1152 | 16384 | 81517.9 | 201.0 | 6.3 | 22.90 |

The 1.1.11 rows ran with MTP on, temperature 0.0 and seed 42. The 1.1.12 rows ran with MTP and **Use 8-bit resident weights** on, temperature 0.0 and seed 42. Older measurements are kept in [BENCHMARK.md](BENCHMARK.md).

#### M5 Pro: Swift1.5 image prompts with MTP

Measured on the 1.1.11 source with 160 output tokens, temperature 0, Prompt Cache off and 3,084 expert slots (7.5 GiB). Each number is the median of 3 rounds. Draft acceptance is the share of MTP's guesses that the model kept.

| Image | Image tokens | Input tokens | MTP | Decode (tok/s) | First token (s) | Draft acceptance |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| House drawing | 384 | 404 | Off | 12.31 | 3.78 | — |
| House drawing | 384 | 404 | On | 17.13 | 3.56 | 67% |
| Bar chart | 256 | 281 | Off | 11.92 | 3.19 | — |
| Bar chart | 256 | 281 | On | 17.07 | 3.09 | 74% |

With MTP on, output was 1.38× and 1.43× as fast, and the first token arrived slightly sooner.

#### M2 Max

| Model | Version | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Swift1.5 Qwen3.8 | 1.1.12 | 2089 | 4096 | 15151.2 | 270.3 | 15.0 | 13.15 |
| Swift1.5 Qwen3.8 | 1.1.12 | 2089 | 8192 | 30693.5 | 266.9 | 15.8 | 13.84 |
| Swift1.5 Qwen3.8 | 1.1.12 | 2089 | 16384 | 65538.0 | 250.0 | 14.7 | 14.42 |
| Qwen3.8 | 1.1.12 | 2089 | 4096 | 15019.1 | 272.7 | 15.1 | 12.58 |
| Qwen3.8 | 1.1.12 | 2089 | 8192 | 30346.1 | 270.0 | 17.2 | 13.15 |
| Qwen3.8 | 1.1.12 | 2089 | 16384 | 64771.4 | 253.0 | 14.4 | 14.05 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

The 1.1.12 rows ran with MTP and **Use 8-bit resident weights** on, temperature 0.0 and seed 42. Older measurements are kept in [BENCHMARK.md](BENCHMARK.md).

To test your own Mac, open **Throughput**, then pick:

- an installed model
- **Code** or **Novel** input
- input length from **1K to 200K**
- output limit of **128, 512, 1024 or 4096**

You can copy the results as plain text, JSON or Markdown. The app unloads the test model when the run finishes or is cancelled. Local builds also offer **Dry run**, which shows simulated results.

Results depend on SSD speed, prompt length, cache state and settings.

## Connect Codex

Start Whallm's server, then add this to `~/.codex/config.toml`:

```toml
model = "deepseek-v4-flash-0731"
model_provider = "deepseek-v4-ssd"
model_reasoning_effort = "high"

[model_providers.deepseek-v4-ssd]
name = "Whallm"
base_url = "http://127.0.0.1:11434/v1"
wire_api = "responses"
requires_openai_auth = false
```

Set `model` to an API model ID, or to an Alias you set in **Model → Advanced Settings**. Then restart Codex.

- The names shown in Chat, Status and Throughput are display names, not API model IDs. Requests and benchmark JSON still use the API model IDs.
- This example uses the default address and no API key. If you set a key in Whallm, set the same key in your client.

See the [Codex configuration reference](https://developers.openai.com/codex/config-reference/).

## API and privacy

The API streams text and supports tool calls through:

- `GET /healthz` and `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` and `POST /v1/completions`
- `POST /api/models/load` and `POST /api/models/unload`

**Seed** (since v1.1.8): all three generation endpoints accept an optional `seed` from `0` to `4294967295`. Leave it out or send `null` to get a new random seed each time. In Chat, Seed applies only to the next message. A fixed seed repeats a result only with the same prompt, model, settings and runtime; other versions, cache states or speed-up settings can still change the text. `temperature=0` always picks the most likely token.

**Tools**: your client runs the tools and sends the results back. Whallm never runs MCP tools by itself. Since v1.1.9, `/v1/responses` accepts `function_call_output.output` as a string or as a list of `input_text` parts for every model. Parts are joined in order with nothing added between them. Include the matching `function_call` with the same `call_id` in the request history.

**MiMo (preview)** accepts still images and short PCM WAV audio through Chat, Responses and app attachments.

- JSON request bodies: up to **1 MiB**.
- `/api/assets` uploads (with sign-in): PNG, JPEG, WebP, PCM WAV and small PDF, DOCX, PPTX or XLSX files, up to **8 MiB** each.
- Audio: 24 kHz, 16-bit PCM WAV, mono or stereo, up to 30 seconds per clip and two clips per request.
- Not supported: audio output, video, `logprobs`, `response_format` and `stop`.

See the [MiMo contract and limits](docs/mimo-development.md).

**Privacy**: models run on your Mac. Whallm uses the network only for downloads, updates and API connections. Clients you connect may send data elsewhere. **Debug** records settings and cache details, not full prompts, tool results, images or API keys. Optional file logging saves to `~/.whallm/server.log` (10 MB maximum; oldest content is removed first). Logs can still contain local paths and network addresses; review them before sharing.

## Validation and limits

- Tests: **705 Python tests passed**; **193 Swift tests ran, 4 skipped, none failed**.
- Packaging checks cover signatures, bundled files and startup in English, Simplified Chinese and Traditional Chinese, without access to the build folder. They run on both the app and the unzipped ZIP. Releases repeat them on the signed, notarized download.
- Tool-call tests use fixed model output. They are not a full Pi or Codex session.
- MiMo is a preview. It supports text, images, short WAV audio and some documents from a fixed prepared artifact. Video, audio-video sync and the MCP Agent interface are not finished. See the [development status](docs/mimo-development.md).
- Older MTP measurements predate the current runtime. MTP is not faster for every prompt.
- Some speed-ups are tested only on small models or single parts.
- Very long prompts need more cache memory.

## License

Whallm uses the [MIT License](LICENSE). Model weights have their own terms. Whallm is not affiliated with DeepSeek or Qwen.
