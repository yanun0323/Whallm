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

Whallm runs large language models on Apple Silicon Macs by reading the experts it needs from SSD. It supports DeepSeek V4, DeepSeek V4.1, and Qwen3.8, with built-in chat and an OpenAI-compatible API.

## Benchmark summary

| Model | Chipset | Prefill | Decode | Peak memory | Expert cache slots |
| --- | --- | ---: | ---: | ---: | ---: |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `Qwen3.8-Flash-Next-FP8` | M5 Pro | 99.1–153.7 tok/s | 8.5–10.6 tok/s | 18 GiB | 3072 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> Tested in v1.1.7 using the built-in Throughput benchmark with 1,024 to 16,384 input tokens.
>
> See the [full benchmark](#benchmarks) for details.

## Quick start

1. Download `Whallm-macOS-arm64.zip` from [GitHub Releases](https://github.com/yanun0323/Whallm/releases), extract it, and open `Whallm.app`.
2. Open **Model**, choose a model, and select **Download Model**. The app checks storage space; interrupted downloads can resume.
3. Open **Server** and select **Start Server**.
4. Open **Chat** and choose your model, or connect an API client using the example below.

The default address is `http://127.0.0.1:11434`. Models load when first used. The server keeps one model loaded and handles one generation request at a time. Restart the server if a newly installed model is missing from the chat picker.

Public downloads may contain fewer features than the source described here.

## Requirements

| Item | Requirement |
| --- | --- |
| Mac | Apple Silicon, macOS 15 or later |
| Unified memory | 64 GiB recommended; usage depends on the model and settings |
| Storage | A fast internal, Thunderbolt, or USB4 SSD |
| Free space | The app calculates the requirement for each model, including partial downloads |
| Network | Needed for model downloads and app updates |

Model weights are not included with the app. DeepSeek V4.1's expert and Engram files alone need about **458 GiB**, plus common weights and metadata.

## Features

Whallm keeps common weights in memory and reads selected experts from SSD. DeepSeek V4.1 Engram rows and Qwen N-gram rows are also read as needed.

| Model | Acceleration options |
| --- | --- |
| DeepSeek V4 | Layer-by-layer input processing, batched expert calculations, FP8 KV cache, optional ANE projection and DSpark |
| DeepSeek V4.1 | Layer-by-layer input processing, batched experts, packed KV/index caches, candidate-only index scoring, CED input processing, ANE projection and DSpark |
| Qwen3.8 | Grouped experts during input processing, expert calculations as reads finish, QSA cache compression, next-layer prefetch, ANE projection and MTP |

## Benchmarks

These recorded v1.1.7 runs use **Code** context and an output limit of **128 tokens**. The build revisions and cache state were not recorded alongside these rows, so they are reference results, not a controlled comparison of the new acceleration options.

TTFT is the wait for the first token. Prefill measures input processing; Decode measures output generation, both in tokens per second. Peak MLX is MLX allocation in **GiB**, not total Mac memory. The app export labels this value GB but divides bytes by 1024³.

### M5 Pro

| Model | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4 | 1152 | 1024 | 19112.3 | 53.6 | 7.7 | 22.77 |
| DeepSeek V4 | 1152 | 4096 | 24498.6 | 167.2 | 5.9 | 22.80 |
| DeepSeek V4 | 1152 | 8192 | 42137.5 | 194.4 | 6.8 | 22.83 |
| DeepSeek V4 | 1152 | 16384 | 81517.9 | 201.0 | 6.3 | 22.90 |
| Qwen3.8 | 3072 | 1024 | 10336.3 | 99.1 | 10.6 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 28831.0 | 142.1 | 9.2 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 53303.2 | 153.7 | 10.1 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 112353.1 | 145.8 | 8.5 | 17.18 |

### M2 Max

| Model | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

To test your Mac, open **Throughput**, choose an installed model, select **Code** or **Novel**, input lengths from **1K to 200K**, and an output limit of **128, 512, 1024, or 4096**. Results can be copied as plain text, JSON, or Markdown. The app unloads the test model when the run finishes or is cancelled. Local builds also offer **Dry run**, which produces simulated results.

SSD speed, prompt length, cache state, and settings affect results.

## Connect Codex

Start Whallm's server, then add this to your user-level `~/.codex/config.toml`:

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

Set `model` to an API model ID or Alias shown in Whallm, then restart Codex. This example assumes the default local address and no API key. If you configure a key in Whallm, configure the same key in your client. See the [Codex configuration reference](https://developers.openai.com/codex/config-reference/).

## API and privacy

The API supports text streaming and tool calls through:

- `GET /healthz` and `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` and `POST /v1/completions`
- `POST /api/models/load` and `POST /api/models/unload`

Current source supports an optional `seed` integer from `0` to `4294967295` on all three generation endpoints. Omit it or send `null` for a fresh random seed on every request. In Playground Chat, Seed applies only to the next message and clears after sending. A fixed seed helps reproduce a result with the same prompt, model, settings, and runtime; it does not guarantee identical text across versions, cache states, or acceleration settings. `temperature=0` remains greedy. Available in v1.1.8.

The client executes tools and sends their results back; inference APIs never run MCP tools automatically. Current development source adds still-image and bounded PCM WAV input for MiMo through Chat/Responses and App attachments. JSON bodies remain limited to **1 MiB**; authenticated `/api/assets` uploads accept PNG/JPEG/WebP, PCM WAV and bounded PDF/DOCX/PPTX/XLSX documents up to **8 MiB** per file. Audio input is limited to 24 kHz, 16-bit PCM WAV, mono/stereo, 30 seconds per clip and two clips per request. Audio output, video, `logprobs`, `response_format`, and `stop` remain unsupported. See the [MiMo contract and limits](docs/mimo-development.md).

Inference runs on your Mac. Network access is used for downloads, updates, and API connections. Connected clients may send data elsewhere; **Debug** logs can contain complete prompts and tool results.

## Validation and limits

The v1.1.8 source passed **464 Python tests** and **135 Swift tests**. Both the app and extracted ZIP passed signature and isolated startup checks in English, Simplified Chinese, and Traditional Chinese.

Alongside the three pinned text checkpoints, current development source adds MiMo text/image, bounded WAV audio and limited document support, plus a complete, immutable prepared artifact. Python 624 tests and Swift 171 tests (three skipped) pass; this MiMo work is locally packaged, but full acceptance remains incomplete. Chat attachments include categorized pickers, drag-and-drop, quota indicators and upload status. Video/AV synchronization and the MCP Agent interface remain unfinished; audio and document inputs have format and resource limits. See [development status](docs/mimo-development.md). New acceleration paths have small-model and component tests; full-model speed and quality comparisons are still pending. Very long prompts need more cache memory.

## License

Whallm is released under the [MIT License](LICENSE). Model weights have their own terms. Whallm is not affiliated with DeepSeek or Qwen.
