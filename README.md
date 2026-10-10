# Whallm

<p align="center">
  <a href="."><img height="160" src="Packaging/AppIcon.png" alt="Whallm"></a>
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-青" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

Whallm runs large language models on your Apple Silicon Mac. It keeps the shared weights in memory and reads only the experts each token needs from your SSD, so it can run models much larger than your memory. It comes with a chat window and an OpenAI-compatible API.

Supported models: DeepSeek V4, DeepSeek V4.1, Qwen3.8, Swift1.5 Qwen3.8 and MiMo (preview).

## Benchmark summary

| Model | Chip | Prefill | Decode | Peak memory | Expert cache slots |
| --- | --- | ---: | ---: | ---: | ---: |
| `Swift1.5-Qwen3.8-Flash-Next` | M5 Pro | 552.5–588.9 tok/s | 18.6–19.8 tok/s | 14 GiB | 2089 |
| `Swift1.5-Qwen3.8-Flash-Next` | M2 Max | 250.0–270.3 tok/s | 14.7–15.8 tok/s | 14 GiB | 2089 |
| `Qwen3.8-Flash-Next-FP8` | M5 Pro | 484.4–547.3 tok/s | 15.0–17.2 tok/s | 18 GiB | 2089 |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> Measured with the built-in Throughput test. Swift1.5 ran on a build from source after v1.1.11 with MTP and **Use 8-bit resident weights** on, 4,096 to 16,384 input tokens and a 512-token output limit. Qwen3.8 FP8 ran on v1.1.11 with MTP on and the same input sizes. The DeepSeek models ran on v1.1.7 with 1,024 to 16,384 input tokens. Because the versions and settings differ, don't compare those rows directly.
>
> See the [full benchmark](#benchmarks) for details.

## Quick start

1. Download `Whallm-macOS-arm64.zip` from [GitHub Releases](https://github.com/yanun0323/Whallm/releases), unzip it and open `Whallm.app`.
2. Open **Model**, pick a model and select **Download Model**. The app checks free space first. If a download stops, you can resume it.
3. Open **Server** and select **Start Server**.
4. Open **Chat** and pick your model, or connect an API client as shown below.

The server listens on `http://127.0.0.1:11434`. A model loads the first time you use it. The server keeps one model loaded and runs one request at a time. If a model you just installed is missing from the Chat list, restart the server.

The [1.1.12 release notes](Packaging/ReleaseNotes/1.1.12.md) list the changes, upgrade steps and known limits.

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

### Swift1.5-Qwen3.8-Flash-Next

Added in 1.1.10.

- API model ID: `swift1.5-qwen3.8-flash-next-mxfp4`
- Source: `Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4`, revision `257cb509d72ecc07519be35a8b7558fde3b31b21`
- Download size: **127,714,522,478 bytes** (about **118.94 GiB**). This covers 59 text and MTP files plus **897,864,704 bytes** of image weights. Audit files are not downloaded.
- It uses the same engine as Qwen3.8 FP8 but keeps its own folder, settings, Alias and prompt cache.

### Settings

- Both Qwen models show the same short **Advanced Settings**: Alias, generation, memory, read workers, Prompt Cache, warmup, MTP and 8-bit resident weights. Everything else uses fixed values.
- MTP is on by default. It usually speeds up output, but not for every prompt. Output can differ slightly from output with MTP off.
- **Use 8-bit resident weights** is off by default. It keeps the weights that stay in memory, except experts, in 8 bits instead of BF16. This uses less memory and usually speeds up output, most with MTP on. Output can differ slightly from output with the setting off.

### Images

- Both Qwen models accept still PNG, JPEG and WebP images in Chat, `/v1/chat/completions` and `/v1/responses`.
- Images work with MTP on or off. On the full Swift1.5 model, every token MTP produced was the one the model itself would pick, and output was about 1.4× faster for two test images. Image requests don't use the prompt cache.
- Limits per request: 8 images, 8 MiB per file, 32 MiB in total, 1,048,576 pixels per image after resizing and 2,048 image tokens.
- New installs include the image weights. On an older install, select **Model → Verify and Repair**; it downloads only missing or damaged files. A full Qwen3.8 FP8 install is **126,189,355,659 bytes** (about **117.52 GiB**).
- Video, audio and document input are not available for Qwen.
- What's tested: the full Swift1.5 model recognized red and blue images, and short text answers stayed the same before and after image requests and cancellation. Image generation on the full Qwen3.8 FP8 model, overall image quality and image speed are not tested yet.

## Features

Whallm keeps shared weights in memory and reads the chosen experts from SSD. It also reads DeepSeek V4.1 Engram rows and Qwen N-gram rows only when needed.

| Model | Speed-ups |
| --- | --- |
| DeepSeek V4 | Layer-by-layer input processing, batched expert math, FP8 KV cache, optional ANE projection and DSpark |
| DeepSeek V4.1 | Layer-by-layer input processing, batched experts, packed KV and index caches, scoring only candidate indexes, CED input processing, ANE projection and DSpark |
| Qwen3.8 | Grouped experts during input processing, expert math as soon as each read finishes, QSA cache compression, next-layer prefetch and MTP |

## Benchmarks

Except for the image table, rows use **Code** input. The output limit is **512 tokens** for the Swift1.5 rows and **128 tokens** for the other rows. For those rows the build revision and cache state were not recorded, so treat these numbers as a reference, not as a controlled comparison between versions or speed-ups.

- **TTFT**: time until the first token appears.
- **Prefill**: input processing speed. **Decode**: output speed. Both are in tokens per second.
- **Peak MLX**: the most memory MLX used, in **GiB**. It is not the Mac's total memory use. Older app exports label this value GB, but it is computed in GiB.

### M5 Pro

| Model | Version | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Swift1.5 Qwen3.8 | 1.1.11+ | 2089 | 4096 | 6976.5 | 587.1 | 19.4 | 13.13 |
| Swift1.5 Qwen3.8 | 1.1.11+ | 2089 | 8192 | 13911.4 | 588.9 | 19.8 | 13.86 |
| Swift1.5 Qwen3.8 | 1.1.11+ | 2089 | 16384 | 29654.2 | 552.5 | 18.6 | 14.29 |
| Qwen3.8 | 1.1.11 | 2089 | 4096 | 8456.6 | 484.4 | 15.0 | 16.58 |
| Qwen3.8 | 1.1.11 | 2089 | 8192 | 14968.3 | 547.3 | 17.2 | 17.41 |
| Qwen3.8 | 1.1.11 | 2089 | 16384 | 32572.7 | 503.0 | 15.0 | 18.38 |
| DeepSeek V4 | 1.1.7 | 1152 | 1024 | 19112.3 | 53.6 | 7.7 | 22.77 |
| DeepSeek V4 | 1.1.7 | 1152 | 4096 | 24498.6 | 167.2 | 5.9 | 22.80 |
| DeepSeek V4 | 1.1.7 | 1152 | 8192 | 42137.5 | 194.4 | 6.8 | 22.83 |
| DeepSeek V4 | 1.1.7 | 1152 | 16384 | 81517.9 | 201.0 | 6.3 | 22.90 |

The 1.1.11 rows ran with MTP on, temperature 0.0 and seed 42. The 1.1.11+ rows come from a build from source after 1.1.11, with MTP and **Use 8-bit resident weights** on, temperature 0.0 and seed 42. Older measurements are kept in [BENCHMARK.md](BENCHMARK.md).

### M5 Pro: Swift1.5 image prompts with MTP

Measured on the 1.1.11 source with 160 output tokens, temperature 0, Prompt Cache off and 3,084 expert slots (7.5 GiB). Each number is the median of 3 rounds. Draft acceptance is the share of MTP's guesses that the model kept.

| Image | Image tokens | Input tokens | MTP | Decode (tok/s) | First token (s) | Draft acceptance |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| House drawing | 384 | 404 | Off | 12.31 | 3.78 | — |
| House drawing | 384 | 404 | On | 17.13 | 3.56 | 67% |
| Bar chart | 256 | 281 | Off | 11.92 | 3.19 | — |
| Bar chart | 256 | 281 | On | 17.07 | 3.09 | 74% |

With MTP on, output was 1.38× and 1.43× as fast, and the first token arrived slightly sooner.

### M2 Max

| Model | Version | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Swift1.5 Qwen3.8 | 1.1.11+ | 2089 | 4096 | 15151.2 | 270.3 | 15.0 | 13.15 |
| Swift1.5 Qwen3.8 | 1.1.11+ | 2089 | 8192 | 30693.5 | 266.9 | 15.8 | 13.84 |
| Swift1.5 Qwen3.8 | 1.1.11+ | 2089 | 16384 | 65538.0 | 250.0 | 14.7 | 14.42 |
| Qwen3.8 | 1.1.7 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 1.1.7 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 1.1.7 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 1.1.7 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

The 1.1.11+ rows use the same build and settings as on M5 Pro.

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

**Privacy**: models run on your Mac. Whallm uses the network only for downloads, updates and API connections. Clients you connect may send data elsewhere. **Debug** logs can contain full prompts and tool results.

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
