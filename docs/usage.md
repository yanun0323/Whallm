# Model settings and API reference

[Quick start](../README.md) · [Benchmarks](../BENCHMARK.md) · [Release notes](../Packaging/ReleaseNotes/1.1.13.md)

## API model IDs

Use these case-sensitive IDs in requests, or set an Alias in **Model → Advanced Settings**. Chat, Status and Throughput show display names; requests and benchmark JSON use API model IDs.

| Model | API model ID |
| --- | --- |
| DeepSeek V4 | `deepseek-v4-flash-0731` |
| DeepSeek V4.1 | `deepseek-v4.1-flash` |
| Qwen3.8 | `qwen3.8-flash-next-fp8` |
| Swift1.5 Qwen3.8 | `swift1.5-qwen3.8-flash-next-mxfp4` |
| MiMo (preview) | `mimo-v2.6-flash-rl` |

## Qwen models

Both Qwen models use Whallm's own engine, written in C++ and Metal (the **Native engine**), instead of MLX. Your downloads, model names, API IDs and Aliases stay the same. If a client still uses an old `-Native` test ID, switch it to the ID below. Prompt caches saved by the old MLX engine are not reused.

### Swift1.5-Qwen3.8-Flash-Next

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

### Thinking and total output limits (1.1.14-dev.2)

Both Qwen models support an optional `think_budget` in Chat, Responses and Completions requests. It counts generated thinking tokens, not input tokens. At the limit, Whallm emits and feeds the model's `</think>` token back into the engine, then continues answering or making tool calls within the same request. Thinking that ends earlier is left alone. A prompt with no open thinking block is unchanged.

- `max_tokens` (or `max_completion_tokens` / `max_output_tokens`) still limits **all output**: thinking, its closing marker, answers and tool calls. Thinking is not a separate free allowance.
- `think_budget` is a nonnegative integer. `0` ends an open thinking block immediately; `null` disables the budget for that request. Omitting it uses the App/catalog default, initially unset. Other models reject a non-null budget.
- Leave at least two tokens beyond the budget for the closing marker and one answer token; allow much more for useful answers. A budget that does not fit the requested total is rejected. If the input leaves too little context space, the request fails with `context_length_exceeded` rather than silently spending the answer space on thinking.
- In **Model → Advanced Settings**, set **Total output limit**, enable **Limit thinking**, and set **Thinking limit**. Enabling initially chooses half the total, capped at 8,192; it does not increase the total. Old settings remain unchanged. API values override these defaults, so a client sending `max_tokens: 8192` must raise that value to use an 8,192-token thinking budget.
- MTP remains available; verification rounds stop at the thinking boundary. The closing marker counts in usage and is included in the engine's prompt-cache history. Forcing the boundary does not guarantee a correct or complete answer.

For example, these fields allow up to 8,192 thinking tokens, leaving up to 8,191 tokens for the answer after the closing marker:

```json
{
  "model": "swift1.5-qwen3.8-flash-next-mxfp4",
  "messages": [{"role": "user", "content": "Explain your solution."}],
  "reasoning_effort": "xhigh",
  "think_budget": 8192,
  "max_tokens": 16384
}
```

This requires `1.1.14-dev.2` or later; it is not included in `1.1.14-dev.1`. Short Swift-model checks cover forced transitions, MTP on/off, repeat generation and state-file round trips; they are not broad quality validation or an 8,192-token remote Mho session test.

Starting with `1.1.14-dev.2`, the shared Qwen engine also overlaps shared-expert computation with routing and required expert reads during 2–6-row MTP verification. No setting is needed. The change preserves the arithmetic and does not prefetch guessed experts. A short Swift-only M2 Max experiment measured about 2.15% faster decoding overall; other models, Macs and long contexts have not been validated for this release.

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
- See the [DeepSeek V4.1 observations](../BENCHMARK.md#deepseek-v41-native-observations) for measured performance and remaining validation limits.

## Features

Whallm keeps the parts every word needs in memory and reads only the experts it needs from your SSD. It also reads DeepSeek V4.1 Engram rows and Qwen N-gram rows only when they are needed.

| Model | How it speeds things up |
| --- | --- |
| DeepSeek V4 | Reads the input layer by layer, does expert math in batches, keeps a smaller (FP8) attention cache, and offers ANE projection and DSpark |
| DeepSeek V4.1 | Native engine: for long inputs, reads each expert once while the GPU works; when writing, starts with experts already in memory; DSpark checks guesses without redoing kept words |
| Qwen3.8 | Groups experts while reading the input, starts expert math as soon as each read finishes, compresses the QSA cache, reads the next layer ahead and uses MTP |

## API and privacy

The API streams text and supports tool calls through:

- `GET /healthz` and `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` and `POST /v1/completions`
- `POST /api/models/load` and `POST /api/models/unload`

**Seed** (since v1.1.8): all three generation endpoints accept an optional `seed` from `0` to `4294967295`. Leave it out or send `null` to get a new random seed each time. In Chat, Seed applies only to the next message. The same seed repeats a result only with the same prompt, model, settings and version; a different cache state or speed-up setting can still change the text. `temperature=0` always picks the most likely token.

**Tools**: your client runs the tools and sends back the results. Whallm never runs MCP tools by itself. Since v1.1.9, for every model, `/v1/responses` accepts `function_call_output.output` as a string or as a list of `input_text` parts. The parts are joined in order with nothing between them. Include the matching `function_call` with the same `call_id` in the request history.

**Output limits (1.1.14-dev.1)**: reaching the output-token limit keeps Chat's `finish_reason: "length"` and token usage, including tool-enabled requests. Responses returns `status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"`; streams end with `response.incomplete`. Unfinished thinking or tool markup alone does not turn a length-limited response into a parsing error. Required-tool requests do not retry after exhausting the output budget; malformed completed output still fails validation.

**MiMo (preview)** accepts still images and short PCM WAV audio in Chat, Responses and app attachments.

- JSON request bodies: up to **1 MiB**.
- `/api/assets` uploads (sign-in required): PNG, JPEG, WebP, PCM WAV and small PDF, DOCX, PPTX or XLSX files, up to **8 MiB** each.
- Audio: 24 kHz, 16-bit PCM WAV, mono or stereo, up to 30 seconds per clip and two clips per request.
- Not supported: audio output, video, `logprobs`, `response_format` and `stop`.

See the [MiMo contract and limits](mimo-development.md).

**Privacy**: models run on your Mac. Whallm uses the network only for downloads, updates and API connections. Clients you connect may send data elsewhere. **Debug** records settings and cache details, but not full prompts, tool results, images or API keys. You can also save the log to `~/.whallm/server.log` (10 MB at most; the oldest lines are removed first). The log can still contain local paths and network addresses, so check it before you share it.

## Validation and limits

- Packaging checks the app's signature and bundled files, and that it starts in English, Simplified Chinese and Traditional Chinese without access to the build folder. It checks both the app and the unzipped ZIP. Releases repeat these checks on the signed, notarized download.
- Tool-call tests use fixed model output. They are not a full Pi or Codex session.
- MiMo is a preview. It supports text, images, short WAV audio and some documents, from one fixed prepared copy of the model. Video, audio-video sync and the MCP Agent interface are not finished. See the [development status](mimo-development.md).
- Older MTP measurements were made before the current engine. MTP is not faster for every prompt.
- Some speed-ups have been tested only on small models or on single parts.
- Very long prompts need more cache memory.
