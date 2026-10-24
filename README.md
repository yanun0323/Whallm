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

Whallm runs language models larger than your Mac’s memory on Apple Silicon. It keeps common weights in memory and reads experts from SSD as needed. Includes a chat interface and an OpenAI-compatible API for clients such as Codex.

## Requirements

- **Apple Silicon Mac, macOS 15 or later.**
- **64 GiB memory recommended**; actual use depends on the model and settings.
- **Fast SSD**: internal, Thunderbolt or USB4. Model weights are downloaded separately; the app checks required free space, including partial downloads.

## Supported models

| Model | Input | Model files |
| --- | --- | --- |
| Swift1.5 Qwen3.8 | Text, images | ~119 GiB |
| Qwen3.8 | Text, images | ~118 GiB |
| DeepSeek V4.1 | Text | >458 GiB |
| DeepSeek V4 | Text | See app |
| MiMo (preview) | Text, images, short WAV audio, some documents | See app |

DeepSeek V4.1 needs about **458 GiB for experts and Engram alone**, plus other weights. Qwen sizes include image weights. See [model settings and input limits](docs/usage.md) for details.

## Quick start

1. Download `Whallm-macOS-arm64.zip` from [GitHub Releases](https://github.com/yanun0323/Whallm/releases), unzip it and open `Whallm.app`.
2. Open **Model**, pick a model and select **Download Model**. The app checks free space first. If a download stops, you can resume it.
3. Open **Server** and select **Start Server**.
4. Open **Chat** and pick your model, or connect an API client as shown below.

By default the server runs at `http://127.0.0.1:11434`. A model loads the first time you use it. The server keeps one model loaded and handles one request at a time. If a model you just installed is missing from the Chat list, restart the server.

## Benchmark summary

A user measured **Swift1.5-Qwen3.8-Flash-Next** on v1.1.13 with a 16K-token input and a 512-token answer. **Prefill** is how fast Whallm reads your input; **Decode** is how fast it writes the answer. Both are in tokens per second.

| Chip | Cache mode | Slots | Prefill (tok/s) | Decode (tok/s) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | Automatic | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | Fixed | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | Automatic | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | Fixed | 2089 | 536.0 | 21.7 | 16.25 |

Automatic mode chooses the expert cache size; the smaller fixed cache used here trades speed for lower memory use. Slots are the number of expert cache entries.

**Reference only:** user-provided measurements, without the prompt type, full settings, cache state or complete environment. Not a controlled comparison between versions. Peak memory is reported process memory, not total Mac memory.

[Full results and test instructions](BENCHMARK.md). Use **Throughput** in the app to test your own Mac.

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

Find API model IDs with `GET /v1/models` or in the [model ID table](docs/usage.md#api-model-ids). Display names are not request IDs.

This example uses the default address and no API key. If you set a key in Whallm, set the same key in your client.

See the [Codex configuration reference](https://developers.openai.com/codex/config-reference/).

## Limits and privacy

- **One loaded model and one request at a time.** Long prompts need more cache memory.
- **MiMo is a preview**; video and audio output are not supported. Qwen accepts still images, not audio, video or documents.
- Tool calls are executed by your client, not Whallm. Full Pi/Codex sessions and broad model quality are not fully validated.
- Speed depends on SSD, prompt and settings; acceleration is not faster for every prompt.

**Privacy:** inference runs on your Mac. Downloads, updates and API connections use the network; connected clients may send data elsewhere. Debug logs omit full prompts, tool results, images and API keys, but can contain local paths and network addresses. Check logs before sharing.

## Documentation

- [Model settings, API IDs, image limits and API reference](docs/usage.md)
- [Complete benchmarks and historical results](BENCHMARK.md)
- [Release notes and upgrade steps](Packaging/ReleaseNotes/1.1.13.md)
- [MiMo preview status](docs/mimo-development.md)

## License

Whallm uses the [MIT License](LICENSE). Model weights have their own terms. Whallm is not affiliated with DeepSeek or Qwen.
