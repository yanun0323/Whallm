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

Whallm 讓你在 Apple Silicon Mac 上執行比記憶體更大的語言模型：共用權重留在記憶體，專家權重需要時才從 SSD 讀取。內建聊天介面與 OpenAI 相容 API，可連接 Codex 等用戶端。

## 使用需求

- **Apple Silicon Mac，macOS 15 以上。**
- **建議 64 GiB 記憶體**，實際用量依模型與設定而定。
- **高速 SSD**：內建、Thunderbolt 或 USB4。模型權重需另外下載；App 會檢查所需空間，包含下載到一半的檔案。

## 支援模型

| 模型 | 支援輸入 | 模型檔案大小 |
| --- | --- | --- |
| Swift1.5 Qwen3.8 | 文字、圖片 | ~119 GiB |
| Qwen3.8 | 文字、圖片 | ~118 GiB |
| DeepSeek V4.1 | 文字 | >458 GiB |
| DeepSeek V4 | 文字 | 見 App |
| MiMo (預覽版) | 文字、圖片、短 WAV 音訊、部分文件 | 見 App |

DeepSeek V4.1 **光是專家與 Engram 就需要約 458 GiB**，另有其他權重。Qwen 的大小已包含圖片權重。詳細設定與輸入限制見[模型與 API 參考（英文）](docs/usage.md)。

## 快速開始

1. 從 [GitHub Releases](https://github.com/yanun0323/Whallm/releases) 下載 `Whallm-macOS-arm64.zip`，解壓縮後開啟 `Whallm.app`。
2. 開啟「模型」，選擇模型並按「下載模型」。App 會先檢查剩餘空間；下載中斷可以接著繼續。
3. 開啟「伺服器」，按「啟動 server」。
4. 開啟「對話」選擇模型，或依下方範例連接 API 用戶端。

伺服器預設位址是 `http://127.0.0.1:11434`。模型在第一次使用時才載入。伺服器同一時間只載入一個模型、處理一個請求。如果剛安裝的模型沒有出現在「對話」的清單中，請重新啟動伺服器。

## 效能摘要

使用者在 v1.1.13 測量 **Swift1.5-Qwen3.8-Flash-Next**，輸入 16K token、回答 512 token。**Prefill** 是 Whallm 讀取輸入的速度，**Decode** 是寫出回答的速度，單位都是每秒 token 數。

| 晶片 | 快取模式 | Slots | Prefill (tok/s) | Decode (tok/s) | 峰值記憶體 (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | 自動 | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | 固定 | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | 自動 | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | 固定 | 2089 | 536.0 | 21.7 | 16.25 |

自動模式由 Whallm 決定專家快取大小；本次使用的較小固定快取省記憶體，但回答較慢。Slots 是可快取的專家數量。

**僅供參考：**這是使用者提供的測量，缺少提示詞類型、完整設定、快取狀態與完整環境，不是受控的跨版本比較。峰值記憶體是回報的程式用量，不是整台 Mac 的用量。

[完整結果與測試方法（英文）](BENCHMARK.md)。要測自己的 Mac，請開啟 App 的「吞吐量」。

## 連接 Codex

先啟動 Whallm 的伺服器，再把以下內容加到 `~/.codex/config.toml`：

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

把 `model` 改成 API model ID，或你在「模型 → 進階設定」設定的 Alias，然後重新啟動 Codex。

API model ID 可透過 `GET /v1/models` 或[模型 ID 表（英文）](docs/usage.md#api-model-ids)查詢；請勿用顯示名稱代替。

這個範例使用預設位址，沒有 API key。如果你在 Whallm 設了 key，用戶端也要設定同一把 key。

詳見 [Codex 設定參考](https://developers.openai.com/codex/config-reference/)。

## 限制與隱私

- **同一時間只載入一個模型、處理一個請求。** 長提示詞需要更多快取記憶體。
- **MiMo 仍是預覽版**，不支援影片與音訊輸出。Qwen 支援靜態圖片，不接受音訊、影片或文件。
- 工具由用戶端執行，不是 Whallm。完整 Pi／Codex 工作階段與廣泛模型品質尚未完整驗證。
- 速度受 SSD、提示詞與設定影響；加速功能不是對每個提示詞都更快。

**隱私：**推論在你的 Mac 上執行。下載、更新與 API 連線會使用網路；連接的用戶端可能把資料傳到其他地方。偵錯紀錄不含完整提示詞、工具結果、圖片或 API key，但可能包含本機路徑與網路位址，分享前請先檢查。

## 詳細文件

以下詳細文件以英文提供：

- [模型設定、API ID、圖片限制與 API 參考](docs/usage.md)
- [完整效能測試與歷史結果](BENCHMARK.md)
- [版本說明與升級步驟](Packaging/ReleaseNotes/1.1.13.md)
- [MiMo 預覽版狀態](docs/mimo-development.md)

## 授權

Whallm 採用 [MIT 授權](LICENSE)。模型權重另有各自的條款。Whallm 與 DeepSeek、Qwen 無關。
