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

Whallm 讓你在 Apple Silicon Mac 上執行大型語言模型。共用的權重放在記憶體裡，每個 token 需要的專家才從 SSD 讀取，所以能跑比記憶體大很多的模型。App 內建聊天視窗，也提供 OpenAI 相容 API。

支援的模型：DeepSeek V4、DeepSeek V4.1、Qwen3.8、Swift1.5 Qwen3.8，以及 MiMo（預覽版）。

## 效能摘要

使用者提供的 **Swift1.5-Qwen3.8-Flash-Next** v1.1.13 跑分，使用 Native 引擎。下方摘要採用獨立的 **16K 輸入／512 輸出**表格；完整的不同長度測試列在後面。兩組記錄的數字不同，沒有混算平均。

| 晶片 | 快取模式 | Slots | Prefill (tok/s) | Decode (tok/s) | 峰值記憶體 (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | 自動 | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | 固定 | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | 自動 | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | 固定 | 2089 | 536.0 | 21.7 | 16.25 |

來源的快速摘要與 16K 表格記載的峰值記憶體不同；此表採用 16K 表格中較精確的數值。舊 Qwen 與 DeepSeek 結果仍列在後面，保留原版本標示。

> 詳細資料見[完整效能測試](#效能測試)。

## 快速開始

1. 從 [GitHub Releases](https://github.com/yanun0323/Whallm/releases) 下載 `Whallm-macOS-arm64.zip`，解壓縮後開啟 `Whallm.app`。
2. 開啟 **Model**，選擇模型並按 **Download Model**。App 會先檢查剩餘空間；下載中斷可以接著繼續。
3. 開啟 **Server**，按 **Start Server**。
4. 開啟 **Chat** 選擇模型，或依下方範例連接 API 用戶端。

伺服器位址是 `http://127.0.0.1:11434`。模型在第一次使用時載入。伺服器同一時間只載入一個模型、處理一個請求。如果剛安裝的模型沒有出現在 Chat 清單中，請重新啟動伺服器。

[1.1.13 版本說明](Packaging/ReleaseNotes/1.1.13.md)列出變更、升級步驟與已知限制。

## 使用需求

| 項目 | 需求 |
| --- | --- |
| Mac | Apple Silicon，macOS 15 以上 |
| 記憶體 | 建議 64 GiB；實際用量依模型與設定而定 |
| 儲存裝置 | 高速內建、Thunderbolt 或 USB4 SSD |
| 剩餘空間 | App 會顯示每個模型需要多少空間，包含下載到一半的檔案 |
| 網路 | 下載模型與 App 更新時需要 |

App 不含模型權重。光是 DeepSeek V4.1 的專家與 Engram 檔案就需要約 **458 GiB**，另外還有其他權重。

## Qwen 模型

兩款 Qwen 現在都使用 Native 引擎（C++ 與 Metal，不使用 MLX）。既有下載、模型名稱、API ID 和 Alias 不變。兩個 `-Native` 測試項目已移除，使用這些 ID 的客戶端請改用下方原模型的 ID。MLX 專用設定不再生效，也不會沿用 MLX 提示詞快取。

### Swift1.5-Qwen3.8-Flash-Next

1.1.10 版新增。

- API model ID：`swift1.5-qwen3.8-flash-next-mxfp4`
- 來源：`Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4`，版本 `257cb509d72ecc07519be35a8b7558fde3b31b21`
- 下載容量：**127,714,522,478 bytes**（約 **118.94 GiB**），包含 59 個文字與 MTP 檔案，以及 **897,864,704 bytes** 的圖片權重。不下載稽核檔案。
- 與 Qwen3.8 FP8 使用同一套推論引擎，但資料夾、設定、Alias 與提示詞快取各自獨立。

### Qwen3.8-Flash-Next

- API model ID: `qwen3.8-flash-next-fp8`
- Qwen3.8 FP8 與 Swift1.5 使用同一個 Native 引擎與設定選項，但安裝、Alias 與提示詞快取各自獨立。

### 設定

- 兩款模型都保留生成參數、專家快取大小、讀取工作數、Prompt Cache、暖機、MTP 與 8-bit 常駐權重。不再顯示 MLX 記憶體指引、預讀工作數、快取淘汰策略與獨立的 MTP 快取大小。
- **進階設定**有「自動決定專家快取大小」，預設開啟。開啟時，引擎依模型載入時的可用記憶體決定大小，最多約 20 GiB，並至少為 macOS 和其他 App 保留 7 GiB 或這台 Mac 記憶體的 15%。關閉後可自行設定固定大小。其他設定為生成參數、讀取執行緒、提示詞快取、暖機、MTP 與 8-bit 常駐權重。
- MTP 預設開啟。它通常能加快輸出，但不是每個提示詞都會變快。輸出可能與關閉 MTP 時略有不同。
- **使用 8-bit 常駐權重** 在新的或恢復預設的設定中預設開啟；已保存的設定維持原本的選擇。開啟後，常駐記憶體的權重（專家除外）改以 8-bit 取代 BF16 儲存，可減少記憶體用量，通常也會加快輸出，開啟 MTP 時最明顯。輸出可能與關閉時略有不同。

### 圖片

- 兩款 Qwen 都能在 Chat、`/v1/chat/completions` 與 `/v1/responses` 輸入靜態 PNG、JPEG、WebP 圖片。
- MTP 開或關都能傳送圖片。圖片請求不使用提示詞快取。
- 每次請求的限制：8 張圖片、每個檔案 8 MiB、合計 32 MiB、縮放後每張最多 1,048,576 像素、最多 2,048 個圖片 tokens。
- 新安裝已包含圖片權重。舊的安裝請按 **Model → Verify and Repair**，只會下載缺少或損壞的檔案。完整的 Qwen3.8 FP8 安裝為 **126,189,355,659 bytes**（約 **117.52 GiB**）。
- Qwen 不支援影片、音訊與文件輸入。
- Native 在 M2 Max 的檢查涵蓋兩款模型的短文字、Chat、Responses、串流與圖片。下方列有使用者提供的 v1.1.13 Swift1.5 在 M2 Max 與 M5 Pro 的跑分。輸出可能與 MLX 不同；這些數據不代表已完成廣泛品質或效能驗證。

## 功能

Whallm 把共用權重放在記憶體，從 SSD 讀取選中的專家。DeepSeek V4.1 的 Engram 資料列與 Qwen 的 N-gram 資料列也只在需要時讀取。

| 模型 | 加速方式 |
| --- | --- |
| DeepSeek V4 | 逐層處理輸入、批次計算專家、FP8 KV 快取、可選的 ANE 投影與 DSpark |
| DeepSeek V4.1 | 逐層處理輸入、批次計算專家、壓縮的 KV 與索引快取、只為候選索引評分、CED 輸入處理、ANE 投影與 DSpark |
| Qwen3.8 | 處理輸入時分組計算專家、每次讀取完成就立即計算專家、QSA 快取壓縮、預先讀取下一層與 MTP |

## 效能測試

### v1.1.13：Swift1.5 · Native 引擎

以下各列的輸出上限皆為 512 tokens，記錄的輸出也都是 512 tokens。這兩台機器的自動快取使用 8,192 slots，固定快取使用 2,089 slots；換一台 Mac 或可用記憶體較少時，自動模式可能選擇不同大小。來源未記錄提示詞類型、MTP 與 8-bit 設定、temperature、seed、快取狀態、完整軟硬體環境、原始碼 commit、原始測試檔或輸出 token 雜湊。這些是參考數據，不是與舊版的受控比較。

**TTFT** 是首個 token 等待時間，**TPOT** 是每個輸出 token 的時間。**Prefill** 與 **Decode** 分別是每秒處理輸入與產生輸出的 tokens。**總時間**是整次耗時；**Throughput** 計算每秒輸入加輸出的 tokens。**峰值記憶體**是記錄的程序記憶體峰值，不是 Peak MLX 或整台 Mac 的用量。App 約每 10 ms 取樣實體記憶體用量，可能漏掉短暫峰值；來源未註明是否包含 App，或僅計算獨立推論程序。

#### M2 Max · 自動 · 完整長度測試

| Slots | 輸入 / 輸出 | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | 總時間 | Throughput (tok/s) | 峰值記憶體 (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 4096 / 512 | 13.6s | 48ms | 302.0 | 20.9 | 38s | 121.1 | 29.94 |
| 8192 | 8192 / 512 | 27.4s | 48ms | 298.8 | 20.8 | 52s | 167.5 | 30.39 |
| 8192 | 16384 / 512 | 56.8s | 51ms | 288.6 | 19.5 | 1m23s | 203.6 | 31.27 |
| 8192 | 32768 / 512 | 2m3s | 55ms | 266.1 | 18.3 | 2m31s | 220.2 | 32.30 |
| 8192 | 65536 / 512 | 4m45s | 60ms | 230.3 | 16.8 | 5m15s | 209.7 | 34.45 |

#### M2 Max · 固定 · 完整長度測試

| Slots | 輸入 / 輸出 | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | 總時間 | Throughput (tok/s) | 峰值記憶體 (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2089 | 4096 / 512 | 13.7s | 57ms | 299.3 | 17.5 | 42.8s | 107.6 | 15.17 |
| 2089 | 8192 / 512 | 27.3s | 57ms | 300.0 | 17.5 | 56.5s | 154.0 | 15.63 |
| 2089 | 16384 / 512 | 56.5s | 60ms | 290.2 | 16.6 | 1m27s | 193.6 | 16.50 |
| 2089 | 32768 / 512 | 2m2s | 62ms | 269.0 | 16.2 | 2m33s | 217.0 | 17.61 |

#### M5 Pro · 自動 · 完整長度測試

| Slots | 輸入 / 輸出 | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | 總時間 | Throughput (tok/s) | 峰值記憶體 (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 1024 / 512 | 5.2s | 35ms | 197.6 | 28.6 | 23s | 66.7 | 29.44 |
| 8192 | 4096 / 512 | 8.4s | 38ms | 489.6 | 26.6 | 27.6s | 166.9 | 29.84 |
| 8192 | 8192 / 512 | 15.9s | 37ms | 515.9 | 26.9 | 34.9s | 249.6 | 30.23 |
| 8192 | 16384 / 512 | 31.2s | 39ms | 525.7 | 25.6 | 51.1s | 330.6 | 31.06 |
| 8192 | 32768 / 512 | 1m5s | 39ms | 505.8 | 25.5 | 1m25s | 392.2 | 32.06 |

#### M5 Pro · 固定 · 完整長度測試

| Slots | 輸入 / 輸出 | TTFT | TPOT | Prefill (tok/s) | Decode (tok/s) | 總時間 | Throughput (tok/s) | 峰值記憶體 (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2089 | 1024 / 512 | 5.4s | 41ms | 188.9 | 24.3 | 26.5s | 58.1 | 14.60 |
| 2089 | 4096 / 512 | 8.6s | 48ms | 475.9 | 20.7 | 33.3s | 138.3 | 15.01 |
| 2089 | 8192 / 512 | 16.4s | 45ms | 499.7 | 22.0 | 39.6s | 219.8 | 15.41 |
| 2089 | 16384 / 512 | 31.8s | 51ms | 514.7 | 19.5 | 58.1s | 290.8 | 16.22 |
| 2089 | 32768 / 512 | 1m6s | 50ms | 492.8 | 19.9 | 1m32s | 361.1 | 17.26 |

### 歷史結果：v1.1.7–v1.1.12

以下 Qwen 資料使用先前的 MLX 引擎，不是 Native 引擎。其 Peak MLX 數值不能直接與上方的程序記憶體峰值比較。

除了圖片表格，其他資料列都使用 **Code** 輸入。Swift1.5 的資料列輸出上限是 **512 tokens**，其餘是 **128 tokens**。這些資料列測試時沒有記錄 build 版本與快取狀態，請把這些數字當作參考，而不是版本之間或加速方式之間的對照比較。

- **TTFT**：等到第一個 token 出現的時間。
- **Prefill**：處理輸入的速度。**Decode**：產生輸出的速度。兩者單位都是每秒 tokens。
- **Peak MLX**：MLX 使用記憶體的最高值，單位 **GiB**，不是整台 Mac 的記憶體用量。較舊的 App 匯出把這個值標成 GB，但實際以 GiB 計算。

#### M5 Pro

| 模型 | 版本 | Slots | 輸入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
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

1.1.11 的資料列開啟 MTP，使用 temperature 0.0 與 seed 42。1.1.12 的資料列開啟 MTP 與 **使用 8-bit 常駐權重**，使用 temperature 0.0 與 seed 42。較早的量測保留在 [BENCHMARK.md](BENCHMARK.md)。

#### M5 Pro：Swift1.5 圖片提示詞搭配 MTP

以 1.1.11 原始碼量測：輸出 160 tokens、temperature 0、關閉 Prompt Cache、專家快取 3,084 slots（7.5 GiB）。每個數字是 3 輪的中位數。草稿接受率是 MTP 猜的字被模型採用的比例。

| 圖片 | 圖片 tokens | 輸入 tokens | MTP | Decode (tok/s) | 第一個 token (s) | 草稿接受率 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 房子圖 | 384 | 404 | 關 | 12.31 | 3.78 | — |
| 房子圖 | 384 | 404 | 開 | 17.13 | 3.56 | 67% |
| 長條圖 | 256 | 281 | 關 | 11.92 | 3.19 | — |
| 長條圖 | 256 | 281 | 開 | 17.07 | 3.09 | 74% |

開啟 MTP 後，輸出速度分別是 1.38 倍與 1.43 倍，第一個 token 也稍早出現。

#### M2 Max

| 模型 | 版本 | Slots | 輸入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
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

1.1.12 的資料列開啟 MTP 與 **使用 8-bit 常駐權重**，使用 temperature 0.0 與 seed 42。較早的量測保留在 [BENCHMARK.md](BENCHMARK.md)。

要測試自己的 Mac，請開啟 **Throughput**，然後選擇：

- 已安裝的模型
- **Code** 或 **Novel** 輸入
- **1K 至 200K** 的輸入長度
- **128、512、1024 或 4096** 的輸出上限

結果可以複製成純文字、JSON 或 Markdown。測試完成或取消後，App 會卸載測試用的模型。本機 build 另有 **Dry run**，顯示模擬結果。

結果會受 SSD 速度、提示詞長度、快取狀態與設定影響。

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

把 `model` 改成 API model ID，或你在 **Model → Advanced Settings** 設定的 Alias，然後重新啟動 Codex。

- Chat、Status 與 Throughput 顯示的是顯示名稱，不是 API model ID。請求與效能測試 JSON 仍使用 API model ID。
- 這個範例使用預設位址，沒有 API key。如果你在 Whallm 設了 key，用戶端也要設定同一把 key。

詳見 [Codex 設定參考](https://developers.openai.com/codex/config-reference/)。

## API 與隱私

API 支援串流文字與工具呼叫，端點如下：

- `GET /healthz` 與 `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` 與 `POST /v1/completions`
- `POST /api/models/load` 與 `POST /api/models/unload`

**Seed**（v1.1.8 起）：三個生成端點都接受選填的 `seed`，範圍 `0` 至 `4294967295`。省略或傳 `null` 時，每次都用新的隨機 seed。在 Chat 中，Seed 只套用到下一則訊息。固定 seed 只有在提示詞、模型、設定與執行環境都相同時才能重現結果；版本、快取狀態或加速設定不同時，文字仍可能改變。`temperature=0` 一律選擇機率最高的 token。

**工具**：由你的用戶端執行工具並回傳結果，Whallm 不會自行執行 MCP 工具。v1.1.9 起，所有模型的 `/v1/responses` 都接受字串，或由 `input_text` 組成的清單作為 `function_call_output.output`。各段依序直接接起來，中間不加任何字元。請在請求歷史中附上 `call_id` 相同的 `function_call`。

**MiMo（預覽版）** 可透過 Chat、Responses 與 App 附件輸入靜態圖片和短的 PCM WAV 音訊。

- JSON 請求內容：最多 **1 MiB**。
- `/api/assets` 上傳（需驗證）：PNG、JPEG、WebP、PCM WAV，以及小型 PDF、DOCX、PPTX、XLSX 檔案，每個最多 **8 MiB**。
- 音訊：24 kHz、16-bit PCM WAV、單聲道或立體聲，每段最多 30 秒，每次請求最多兩段。
- 不支援：音訊輸出、影片、`logprobs`、`response_format` 與 `stop`。

詳見 [MiMo 規格與限制](docs/mimo-development.md)。

**隱私**：模型在你的 Mac 上執行。Whallm 只在下載、更新與 API 連線時使用網路。你連接的用戶端可能把資料傳到其他地方。**Debug** 記錄設定與快取資訊，不記錄完整提示詞、工具結果、圖片或 API key。可選擇儲存至 `~/.whallm/server.log`（最多 10 MB，優先刪除最舊內容）。紀錄仍可能包含本機路徑與網路位址，分享前請先檢查。

## 驗證與限制

- 測試：**705 項 Python 測試通過**；**193 項 Swift 測試執行，4 項略過，沒有失敗**。
- 打包檢查涵蓋簽章、內含檔案，以及在無法存取 build 資料夾時以英文、簡體中文、繁體中文啟動。App 與解壓縮後的 ZIP 都會檢查。正式發布時會在簽章、公證後的下載檔上再檢查一次。
- 工具呼叫測試使用固定的模型輸出，不是完整的 Pi 或 Codex 工作階段。
- MiMo 仍是預覽版，使用固定的預先處理檔案，支援文字、圖片、短 WAV 音訊與部分文件。影片、影音同步與 MCP Agent 介面尚未完成。詳見[開發狀態](docs/mimo-development.md)。
- 較早的 MTP 量測早於目前的執行環境。MTP 不是對每個提示詞都更快。
- 部分加速方式只在小模型或單一元件上測試過。
- 很長的提示詞需要更多快取記憶體。

## 授權

Whallm 採用 [MIT 授權](LICENSE)。模型權重另有各自的條款。Whallm 與 DeepSeek、Qwen 無關。
