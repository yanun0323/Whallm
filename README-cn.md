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

Whallm 让你在 Apple Silicon Mac 上运行比内存更大的语言模型：共享权重留在内存，专家权重需要时才从 SSD 读取。内置聊天界面和兼容 OpenAI 的 API，可连接 Codex 等客户端。

## 使用要求

- **Apple Silicon Mac，macOS 15 或更高版本。**
- **建议 64 GiB 内存**，实际用量取决于模型和设置。
- **高速 SSD**：内置、Thunderbolt 或 USB4。模型权重需另行下载；App 会检查所需空间，包括下载到一半的文件。

## 支持模型

| 模型 | 支持输入 | 模型文件大小 |
| --- | --- | --- |
| Swift1.5 Qwen3.8 | 文本、图片 | ~119 GiB |
| Qwen3.8 | 文本、图片 | ~118 GiB |
| DeepSeek V4.1 | 文本 | >458 GiB |
| DeepSeek V4 | 文本 | 见 App |
| MiMo (预览版) | 文本、图片、短 WAV 音频、部分文档 | 见 App |

DeepSeek V4.1 **仅专家和 Engram 就需要约 458 GiB**，另有其他权重。Qwen 的大小已包含图片权重。详细设置和输入限制见[模型与 API 参考（英文）](docs/usage.md)。

## 快速开始

1. 从 [GitHub Releases](https://github.com/yanun0323/Whallm/releases) 下载 `Whallm-macOS-arm64.zip`，解压后打开 `Whallm.app`。
2. 打开“模型”，选择模型并点击“下载模型”。App 会先检查剩余空间；下载中断后可以接着继续。
3. 打开“服务器”，点击“启动 server”。
4. 打开“对话”选择模型，或按下方示例连接 API 客户端。

服务器默认地址是 `http://127.0.0.1:11434`。模型在第一次使用时才加载。服务器同一时间只加载一个模型、处理一个请求。如果刚安装的模型没有出现在“对话”的列表中，请重启服务器。

## 性能摘要

用户在 v1.1.13 上测量了 **Swift1.5-Qwen3.8-Flash-Next**，输入 16K token，回答 512 token。**Prefill** 是 Whallm 读取输入的速度，**Decode** 是写出回答的速度，单位都是每秒 token 数。

| 芯片 | 缓存模式 | Slots | Prefill (tok/s) | Decode (tok/s) | 峰值内存 (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | 自动 | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | 固定 | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | 自动 | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | 固定 | 2089 | 536.0 | 21.7 | 16.25 |

自动模式由 Whallm 决定专家缓存大小；本次使用的较小固定缓存更省内存，但回答较慢。Slots 是可缓存的专家数量。

**仅供参考：**这是用户提供的测量，缺少提示词类型、完整设置、缓存状态和完整环境，不是受控的跨版本比较。峰值内存是报告的进程用量，不是整台 Mac 的用量。

[完整结果与测试方法（英文）](BENCHMARK.md)。要测试自己的 Mac，请打开 App 的“吞吐量”。

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

把 `model` 改成 API model ID，或你在“模型 → 进阶设置”中设置的 Alias，然后重启 Codex。

API model ID 可通过 `GET /v1/models` 或[模型 ID 表（英文）](docs/usage.md#api-model-ids)查询；请勿用显示名称代替。

这个示例使用默认地址，没有 API key。如果你在 Whallm 中设置了 key，客户端也要设置同一个 key。

详见 [Codex 配置参考](https://developers.openai.com/codex/config-reference/)。

## 限制与隐私

- **同一时间只加载一个模型、处理一个请求。** 长提示词需要更多缓存内存。
- **MiMo 仍是预览版**，不支持视频和音频输出。Qwen 支持静态图片，不接受音频、视频或文档。
- 工具由客户端执行，不是 Whallm。完整 Pi／Codex 会话和广泛模型质量尚未完整验证。
- 速度受 SSD、提示词和设置影响；加速功能不是对每个提示词都更快。

**隐私：**推理在你的 Mac 上运行。下载、更新和 API 连接会使用网络；连接的客户端可能把数据发送到其他地方。调试日志不含完整提示词、工具结果、图片或 API key，但可能包含本地路径和网络地址，分享前请先检查。

## 详细文档

以下详细文档以英文提供：

- [模型设置、API ID、图片限制与 API 参考](docs/usage.md)
- [完整性能测试与历史结果](BENCHMARK.md)
- [版本说明与升级步骤](Packaging/ReleaseNotes/1.1.13.md)
- [MiMo 预览版状态](docs/mimo-development.md)

## 许可证

Whallm 采用 [MIT 许可证](LICENSE)。模型权重另有各自的条款。Whallm 与 DeepSeek、Qwen 无关。
