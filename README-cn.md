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

Whallm 让你在 Apple Silicon Mac 上运行大语言模型。共享权重放在内存里，每个 token 需要的专家才从 SSD 读取，所以能运行比内存大得多的模型。App 内置聊天窗口，也提供 OpenAI 兼容 API。

支持的模型：DeepSeek V4、DeepSeek V4.1、Qwen3.8、Swift1.5 Qwen3.8，以及 MiMo（预览版）。

## 性能摘要

| 模型 | 芯片 | Prefill | Decode | 峰值内存 | 专家缓存 Slots |
| --- | --- | ---: | ---: | ---: | ---: |
| `Swift1.5-Qwen3.8-Flash-Next` | M5 Pro | 552.5–588.9 tok/s | 18.6–19.8 tok/s | 14 GiB | 2089 |
| `Swift1.5-Qwen3.8-Flash-Next` | M2 Max | 250.0–270.3 tok/s | 14.7–15.8 tok/s | 14 GiB | 2089 |
| `Qwen3.8-Flash-Next-FP8` | M2 Max | 253.0–272.7 tok/s | 14.4–17.2 tok/s | 14 GiB | 2089 |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> 使用内置 Throughput 测试测量。Swift1.5 的数据行和 M2 Max 的 Qwen3.8 FP8 数据行使用 v1.1.12，开启 MTP 和 **使用 8-bit 常驻权重**，输入 4,096 至 16,384 tokens；输出上限 Swift1.5 是 512 tokens，Qwen3.8 FP8 是 128 tokens。DeepSeek 使用 v1.1.7，输入 1,024 至 16,384 tokens。版本和设置不同，请勿直接比较这些行。
>
> 详细数据见[完整性能测试](#性能测试)。

## 快速开始

1. 从 [GitHub Releases](https://github.com/yanun0323/Whallm/releases) 下载 `Whallm-macOS-arm64.zip`，解压后打开 `Whallm.app`。
2. 打开 **Model**，选择模型并点击 **Download Model**。App 会先检查剩余空间；下载中断后可以继续。
3. 打开 **Server**，点击 **Start Server**。
4. 打开 **Chat** 选择模型，或按下方示例连接 API 客户端。

服务器地址是 `http://127.0.0.1:11434`。模型在第一次使用时加载。服务器同一时间只加载一个模型、处理一个请求。如果刚安装的模型没有出现在 Chat 列表中，请重启服务器。

[1.1.12 版本说明](Packaging/ReleaseNotes/1.1.12.md)列出了更改、升级步骤和已知限制。

## 使用要求

| 项目 | 要求 |
| --- | --- |
| Mac | Apple Silicon，macOS 15 或更高版本 |
| 内存 | 建议 64 GiB；实际用量取决于模型和设置 |
| 存储 | 高速内置、Thunderbolt 或 USB4 SSD |
| 剩余空间 | App 会显示每个模型需要多少空间，包括下载到一半的文件 |
| 网络 | 下载模型和 App 更新时需要 |

App 不含模型权重。仅 DeepSeek V4.1 的专家和 Engram 文件就需要约 **458 GiB**，另外还有其他权重。

## Qwen 模型

### Swift1.5-Qwen3.8-Flash-Next

1.1.10 版新增。

- API model ID：`swift1.5-qwen3.8-flash-next-mxfp4`
- 来源：`Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4`，版本 `257cb509d72ecc07519be35a8b7558fde3b31b21`
- 下载大小：**127,714,522,478 bytes**（约 **118.94 GiB**），包含 59 个文本和 MTP 文件，以及 **897,864,704 bytes** 的图片权重。不下载审计文件。
- 与 Qwen3.8 FP8 使用同一套推理引擎，但文件夹、设置、Alias 和提示词缓存各自独立。

### 设置

- 两款 Qwen 的 **Advanced Settings** 相同，只保留 Alias、生成、内存、读取线程数、Prompt Cache、预热、MTP 和 8-bit 常驻权重。其余选项都使用固定值。
- MTP 默认开启。它通常能加快输出，但不是每个提示词都会变快。输出可能与关闭 MTP 时略有不同。
- **使用 8-bit 常驻权重** 默认关闭。开启后，常驻内存的权重（专家除外）改为以 8-bit 代替 BF16 存储，可减少内存用量，通常也会加快输出，开启 MTP 时最明显。输出可能与关闭时略有不同。

### 图片

- 两款 Qwen 都能在 Chat、`/v1/chat/completions` 和 `/v1/responses` 输入静态 PNG、JPEG、WebP 图片。
- MTP 开或关都能发送图片。在完整的 Swift1.5 模型上，开启 MTP 生成的每个字都是模型自己会选的字，两张测试图片的输出速度约快 1.4 倍。图片请求不使用提示词缓存。
- 每次请求的限制：8 张图片、每个文件 8 MiB、合计 32 MiB、缩放后每张最多 1,048,576 像素、最多 2,048 个图片 tokens。
- 新安装已包含图片权重。旧的安装请点击 **Model → Verify and Repair**，只会下载缺失或损坏的文件。完整的 Qwen3.8 FP8 安装为 **126,189,355,659 bytes**（约 **117.52 GiB**）。
- Qwen 不支持视频、音频和文档输入。
- 已测试：完整的 Swift1.5 模型能识别红色和蓝色图片，图片请求和取消前后的短文本回答保持一致。完整 Qwen3.8 FP8 模型的图片生成、整体图片质量和图片速度尚未测试。

## 功能

Whallm 把共享权重放在内存，从 SSD 读取选中的专家。DeepSeek V4.1 的 Engram 数据行和 Qwen 的 N-gram 数据行也只在需要时读取。

| 模型 | 加速方式 |
| --- | --- |
| DeepSeek V4 | 逐层处理输入、批量计算专家、FP8 KV 缓存、可选的 ANE 投影和 DSpark |
| DeepSeek V4.1 | 逐层处理输入、批量计算专家、压缩的 KV 和索引缓存、只为候选索引评分、CED 输入处理、ANE 投影和 DSpark |
| Qwen3.8 | 处理输入时分组计算专家、每次读取完成就立即计算专家、QSA 缓存压缩、预读下一层和 MTP |

## 性能测试

除了图片表格，其他数据行都使用 **Code** 输入。Swift1.5 的数据行输出上限是 **512 tokens**，其余是 **128 tokens**。这些数据行测试时没有记录 build 版本和缓存状态，请把这些数字当作参考，而不是版本之间或加速方式之间的对照比较。

- **TTFT**：等到第一个 token 出现的时间。
- **Prefill**：处理输入的速度。**Decode**：生成输出的速度。两者单位都是每秒 tokens。
- **Peak MLX**：MLX 使用内存的最高值，单位 **GiB**，不是整台 Mac 的内存用量。较旧的 App 导出把这个值标为 GB，但实际按 GiB 计算。

### M5 Pro

| 模型 | 版本 | Slots | 输入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
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

1.1.11 的数据行开启 MTP，使用 temperature 0.0 和 seed 42。1.1.12 的数据行开启 MTP 和 **使用 8-bit 常驻权重**，使用 temperature 0.0 和 seed 42。较早的测量保留在 [BENCHMARK.md](BENCHMARK.md)。

### M5 Pro：Swift1.5 图片提示词搭配 MTP

基于 1.1.11 源代码测量：输出 160 tokens、temperature 0、关闭 Prompt Cache、专家缓存 3,084 slots（7.5 GiB）。每个数字是 3 轮的中位数。草稿接受率是 MTP 猜的字被模型采用的比例。

| 图片 | 图片 tokens | 输入 tokens | MTP | Decode (tok/s) | 第一个 token (s) | 草稿接受率 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 房子图 | 384 | 404 | 关 | 12.31 | 3.78 | — |
| 房子图 | 384 | 404 | 开 | 17.13 | 3.56 | 67% |
| 柱状图 | 256 | 281 | 关 | 11.92 | 3.19 | — |
| 柱状图 | 256 | 281 | 开 | 17.07 | 3.09 | 74% |

开启 MTP 后，输出速度分别是 1.38 倍和 1.43 倍，第一个 token 也稍早出现。

### M2 Max

| 模型 | 版本 | Slots | 输入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
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

1.1.12 的数据行开启 MTP 和 **使用 8-bit 常驻权重**，使用 temperature 0.0 和 seed 42。较早的测量保留在 [BENCHMARK.md](BENCHMARK.md)。

要测试自己的 Mac，请打开 **Throughput**，然后选择：

- 已安装的模型
- **Code** 或 **Novel** 输入
- **1K 至 200K** 的输入长度
- **128、512、1024 或 4096** 的输出上限

结果可以复制为纯文本、JSON 或 Markdown。测试完成或取消后，App 会卸载测试用的模型。本地 build 另有 **Dry run**，显示模拟结果。

结果会受 SSD 速度、提示词长度、缓存状态和设置影响。

## 连接 Codex

先启动 Whallm 的服务器，再把以下内容加到 `~/.codex/config.toml`：

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

把 `model` 改成 API model ID，或你在 **Model → Advanced Settings** 设置的 Alias，然后重启 Codex。

- Chat、Status 和 Throughput 显示的是显示名称，不是 API model ID。请求和性能测试 JSON 仍使用 API model ID。
- 这个示例使用默认地址，没有 API key。如果你在 Whallm 设置了 key，客户端也要设置同一个 key。

详见 [Codex 配置参考](https://developers.openai.com/codex/config-reference/)。

## API 与隐私

API 支持流式文本和工具调用，端点如下：

- `GET /healthz` 和 `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` 和 `POST /v1/completions`
- `POST /api/models/load` 和 `POST /api/models/unload`

**Seed**（v1.1.8 起）：三个生成端点都接受可选的 `seed`，范围 `0` 至 `4294967295`。省略或传 `null` 时，每次都使用新的随机 seed。在 Chat 中，Seed 只作用于下一条消息。固定 seed 只有在提示词、模型、设置和运行环境都相同时才能重现结果；版本、缓存状态或加速设置不同时，文本仍可能改变。`temperature=0` 总是选择概率最高的 token。

**工具**：由你的客户端执行工具并返回结果，Whallm 不会自行执行 MCP 工具。v1.1.9 起，所有模型的 `/v1/responses` 都接受字符串，或由 `input_text` 组成的列表作为 `function_call_output.output`。各段按顺序直接拼接，中间不加任何字符。请在请求历史中附上 `call_id` 相同的 `function_call`。

**MiMo（预览版）** 可通过 Chat、Responses 和 App 附件输入静态图片和短的 PCM WAV 音频。

- JSON 请求体：最多 **1 MiB**。
- `/api/assets` 上传（需认证）：PNG、JPEG、WebP、PCM WAV，以及小型 PDF、DOCX、PPTX、XLSX 文件，每个最多 **8 MiB**。
- 音频：24 kHz、16-bit PCM WAV、单声道或立体声，每段最多 30 秒，每次请求最多两段。
- 不支持：音频输出、视频、`logprobs`、`response_format` 和 `stop`。

详见 [MiMo 规格与限制](docs/mimo-development.md)。

**隐私**：模型在你的 Mac 上运行。Whallm 只在下载、更新和 API 连接时使用网络。你连接的客户端可能把数据发送到其他地方。**Debug** 日志可能包含完整的提示词和工具结果。

## 验证与限制

- 测试：**705 项 Python 测试通过**；**193 项 Swift 测试运行，4 项跳过，没有失败**。
- 打包检查涵盖签名、内含文件，以及在无法访问 build 文件夹时以英文、简体中文、繁体中文启动。App 和解压后的 ZIP 都会检查。正式发布时会在签名、公证后的下载文件上再检查一次。
- 工具调用测试使用固定的模型输出，不是完整的 Pi 或 Codex 会话。
- MiMo 仍是预览版，使用固定的预处理文件，支持文本、图片、短 WAV 音频和部分文档。视频、音视频同步和 MCP Agent 界面尚未完成。详见[开发状态](docs/mimo-development.md)。
- 较早的 MTP 测量早于当前的运行环境。MTP 不是对每个提示词都更快。
- 部分加速方式只在小模型或单个组件上测试过。
- 很长的提示词需要更多缓存内存。

## 许可证

Whallm 采用 [MIT 许可证](LICENSE)。模型权重有各自的条款。Whallm 与 DeepSeek、Qwen 无关。
