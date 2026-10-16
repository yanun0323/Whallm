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

Whallm runs large language models on your Apple Silicon Mac. These models are split into many small parts called experts, and each word uses only a few of them. Whallm keeps the parts every word needs in memory and reads the experts from your SSD only when they are needed. That is how it can run models much larger than your Mac's memory. It comes with a chat window and an OpenAI-compatible API.

Supported models: DeepSeek V4, DeepSeek V4.1, Qwen3.8, Swift1.5 Qwen3.8 and MiMo (preview).

## Benchmark summary

A user measured **Swift1.5-Qwen3.8-Flash-Next** on v1.1.13 with a 16K-token input and a 512-token answer. **Prefill** is how fast Whallm reads your input; **Decode** is how fast it writes the answer. Both are in tokens per second.

| Chip | Cache mode | Slots | Prefill (tok/s) | Decode (tok/s) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | Automatic | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | Fixed | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | Automatic | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | Fixed | 2089 | 536.0 | 21.7 | 16.25 |

Automatic mode lets Whallm choose the cache size; Fixed uses a smaller set size, which takes less memory but writes more slowly.

> See the [full benchmark](#benchmarks) for details.

## Quick start

1. Download `Whallm-macOS-arm64.zip` from [GitHub Releases](https://github.com/yanun0323/Whallm/releases), unzip it and open `Whallm.app`.
2. Open **Model**, pick a model and select **Download Model**. The app checks free space first. If a download stops, you can resume it.
3. Open **Server** and select **Start Server**.
4. Open **Chat** and pick your model, or connect an API client as shown below.

By default the server runs at `http://127.0.0.1:11434`. A model loads the first time you use it. The server keeps one model loaded and handles one request at a time. If a model you just installed is missing from the Chat list, restart the server.

The [1.1.13 release notes](Packaging/ReleaseNotes/1.1.13.md) list the changes, upgrade steps and known limits.

## Requirements

| Item | Requirement |
| --- | --- |
| Mac | Apple Silicon, macOS 15 or later |
| Memory | 64 GiB recommended; actual use depends on the model and settings |
| Storage | A fast internal, Thunderbolt or USB4 SSD |
| Free space | The app shows how much each model needs, including partial downloads |
| Network | Needed to download models and app updates |

The app does not include model weights. DeepSeek V4.1 alone needs about **458 GiB** for its expert and Engram files, plus its other weights.

## Qwen models

Both Qwen models use Whallm's own engine, written in C++ and Metal (the **Native engine**), instead of MLX. Your downloads, model names, API IDs and Aliases stay the same. If a client still uses an old `-Native` test ID, switch it to the ID below. Prompt caches saved by the old MLX engine are not reused.

### Swift1.5-Qwen3.8-Flash-Next

Added in 1.1.10.

- API model ID: `swift1.5-qwen3.8-flash-next-mxfp4`
- Source: `Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4`, revision `257cb509d72ecc07519be35a8b7558fde3b31b21`
- Download size: **127,714,522,478 bytes** (about **118.94 GiB**), including **897,864,704 bytes** of image weights.
- It uses the same engine as Qwen3.8 FP8 but has its own folder, settings, Alias and prompt cache.

### Qwen3.8-Flash-Next

- API model ID: `qwen3.8-flash-next-fp8`
- It uses the same engine and settings as Swift1.5, with its own install, Alias and prompt cache.

### Settings

- **Advanced Settings** covers generation, expert cache size, read workers, Prompt Cache, warmup, MTP and 8-bit resident weights.
- **Size the expert cache automatically** is on by default. Whallm then uses the memory that is free when the model loads, up to about 20 GiB, and leaves at least 7 GiB or 15% of your Mac's memory for macOS and other apps. Turn it off to pick a fixed size.
- **MTP** guesses the next few words ahead, then checks them. It is on by default and usually makes answers faster, but not for every prompt. The text can differ slightly from the text with MTP off.
- **Use 8-bit resident weights** stores the parts that stay in memory in a smaller format. It uses less memory and usually speeds up answers, most of all with MTP on. The text can differ slightly from the text with the setting off. It is on for new or reset settings; saved settings keep your choice.

### Images

- Both Qwen models accept still PNG, JPEG and WebP images in Chat, `/v1/chat/completions` and `/v1/responses`.
- Images work with MTP on or off. Requests with images do not use the prompt cache.
- Each request allows up to 8 images, 8 MiB per file and 32 MiB in total. Each image is shrunk to at most 1,048,576 pixels, and all images together can use up to 2,048 image tokens.
- New installs include the image weights. For an older install, select **Model → Verify and Repair**; it downloads only missing or damaged files. A full Qwen3.8 FP8 install is **126,189,355,659 bytes** (about **117.52 GiB**).
- Qwen does not accept video, audio or documents.
- On M2 Max, short text, Chat, Responses, streaming and images worked with both models. The text can differ from the old MLX engine. These checks do not prove quality or speed across all uses.

## DeepSeek V4.1

DeepSeek V4.1 also runs on its own Native engine instead of MLX. Your download, model name, API ID and Alias stay the same. Prompt caches saved by the old MLX engine are not reused.

- **Advanced Settings** covers generation, expert cache size, read workers, Prompt Cache, warmup and DSpark.
- **Size the expert cache automatically** is on by default. Whallm then uses the memory that is free when the model loads, up to about 72 GiB, and leaves at least 7 GiB or 15% of your Mac's memory for macOS and other apps. A fixed size must hold at least 96 experts.
- Settings that only worked with MLX are gone, such as approximate mode, ANE projection and the MLX memory guideline. Requests that ask for approximate mode are rejected.
- **DSpark** guesses several words ahead, then checks them. Its experts share the main expert cache, so it no longer has its own cache size. With greedy sampling, DSpark does not change the text. It helps most with predictable text such as code; on other text it can be a little slower than DSpark off.
- On M5 and later chips, long inputs use the GPU's Neural Accelerators. This has not been measured on an M5 yet.
- Measured on M2 Max with 64 GiB and the file cache emptied first: Whallm read a 3,000-token Code prompt at about 70 tok/s and a 16,000-token one at about 140 tok/s. Without DSpark, it wrote about 3.4 tok/s. DSpark's speed has not been measured again since the latest speed-ups. With greedy sampling, the output matched the old MLX engine. Quality, inputs longer than 16,000 tokens and M5 have not been tested.

## Features

Whallm keeps the parts every word needs in memory and reads only the experts it needs from your SSD. It also reads DeepSeek V4.1 Engram rows and Qwen N-gram rows only when they are needed.

| Model | How it speeds things up |
| --- | --- |
| DeepSeek V4 | Reads the input layer by layer, does expert math in batches, keeps a smaller (FP8) attention cache, and offers ANE projection and DSpark |
| DeepSeek V4.1 | Native engine: for long inputs, reads each expert once while the GPU works; when writing, starts with experts already in memory; DSpark checks guesses without redoing kept words |
| Qwen3.8 | Groups experts while reading the input, starts expert math as soon as each read finishes, compresses the QSA cache, reads the next layer ahead and uses MTP |

## Benchmarks

### v1.1.13: Swift1.5 · Native engine

Every row below asked for a 512-token answer and got 512 tokens. Automatic mode chose 8,192 slots on these Macs; Fixed used 2,089. Automatic mode can choose a different size on another Mac or when less memory is free. The person measuring did not record the prompt type, settings, cache state or full setup, so use these numbers as a reference, not as a fair comparison with older versions.

- **TTFT**: time until the first token appears. **TPOT**: time per output token.
- **Prefill** and **Decode**: input and output tokens per second.
- **Total**: time from start to finish. **Throughput**: input plus output tokens per second.
- **Peak memory**: the highest memory use reported for the process, not your Mac's total. The app checks memory about every 10 ms, so very short peaks may be missed. The source does not say whether the app itself was included.

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

These rows come from the old MLX engine. Their **Peak MLX** is the most memory MLX used, in **GiB**, which is not the same as the process memory above. Older app exports label this value GB, but it is in GiB.

Except for the image table, rows use **Code** input. Swift1.5 rows asked for a 512-token answer; the others asked for 128 tokens. The build and cache state were not recorded, so use these numbers as a reference, not as a fair comparison between versions.

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

The 1.1.11 rows had MTP on, temperature 0.0 and seed 42. The 1.1.12 rows also had **Use 8-bit resident weights** on. Older measurements are in [BENCHMARK.md](BENCHMARK.md).

#### M5 Pro: Swift1.5 image prompts with MTP

Measured on 1.1.11 with a 160-token answer, temperature 0, Prompt Cache off and 3,084 expert slots (7.5 GiB). Each number is the middle result of 3 runs. Draft acceptance is the share of MTP's guesses that the model kept.

| Image | Image tokens | Input tokens | MTP | Decode (tok/s) | First token (s) | Draft acceptance |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| House drawing | 384 | 404 | Off | 12.31 | 3.78 | — |
| House drawing | 384 | 404 | On | 17.13 | 3.56 | 67% |
| Bar chart | 256 | 281 | Off | 11.92 | 3.19 | — |
| Bar chart | 256 | 281 | On | 17.07 | 3.09 | 74% |

With MTP on, answers came 1.38× and 1.43× as fast, and the first token came slightly sooner.

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

The 1.1.12 rows had MTP and **Use 8-bit resident weights** on, temperature 0.0 and seed 42. Older measurements are in [BENCHMARK.md](BENCHMARK.md).

To test your own Mac, open **Throughput** and pick:

- an installed model
- **Code** or **Novel** input
- an input length from **1K to 200K**
- an output limit of **128, 512, 1024 or 4096**

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

- Chat, Status and Throughput show display names, not API model IDs. Requests and benchmark JSON still use the API model IDs.
- This example uses the default address and no API key. If you set a key in Whallm, set the same key in your client.

See the [Codex configuration reference](https://developers.openai.com/codex/config-reference/).

## API and privacy

The API streams text and supports tool calls through:

- `GET /healthz` and `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` and `POST /v1/completions`
- `POST /api/models/load` and `POST /api/models/unload`

**Seed** (since v1.1.8): all three generation endpoints accept an optional `seed` from `0` to `4294967295`. Leave it out or send `null` to get a new random seed each time. In Chat, Seed applies only to the next message. The same seed repeats a result only with the same prompt, model, settings and version; a different cache state or speed-up setting can still change the text. `temperature=0` always picks the most likely token.

**Tools**: your client runs the tools and sends back the results. Whallm never runs MCP tools by itself. Since v1.1.9, for every model, `/v1/responses` accepts `function_call_output.output` as a string or as a list of `input_text` parts. The parts are joined in order with nothing between them. Include the matching `function_call` with the same `call_id` in the request history.

**MiMo (preview)** accepts still images and short PCM WAV audio in Chat, Responses and app attachments.

- JSON request bodies: up to **1 MiB**.
- `/api/assets` uploads (sign-in required): PNG, JPEG, WebP, PCM WAV and small PDF, DOCX, PPTX or XLSX files, up to **8 MiB** each.
- Audio: 24 kHz, 16-bit PCM WAV, mono or stereo, up to 30 seconds per clip and two clips per request.
- Not supported: audio output, video, `logprobs`, `response_format` and `stop`.

See the [MiMo contract and limits](docs/mimo-development.md).

**Privacy**: models run on your Mac. Whallm uses the network only for downloads, updates and API connections. Clients you connect may send data elsewhere. **Debug** records settings and cache details, but not full prompts, tool results, images or API keys. You can also save the log to `~/.whallm/server.log` (10 MB at most; the oldest lines are removed first). The log can still contain local paths and network addresses, so check it before you share it.

## Validation and limits

- Tests: **786 Python tests passed, 4 skipped**; **209 Swift tests passed, 4 skipped**. None failed.
- Packaging checks the app's signature and bundled files, and that it starts in English, Simplified Chinese and Traditional Chinese without access to the build folder. It checks both the app and the unzipped ZIP. Releases repeat these checks on the signed, notarized download.
- Tool-call tests use fixed model output. They are not a full Pi or Codex session.
- MiMo is a preview. It supports text, images, short WAV audio and some documents, from one fixed prepared copy of the model. Video, audio-video sync and the MCP Agent interface are not finished. See the [development status](docs/mimo-development.md).
- Older MTP measurements were made before the current engine. MTP is not faster for every prompt.
- Some speed-ups have been tested only on small models or on single parts.
- Very long prompts need more cache memory.

## License

Whallm uses the [MIT License](LICENSE). Model weights have their own terms. Whallm is not affiliated with DeepSeek or Qwen.
