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

Whallm 讓 Apple Silicon Mac 從 SSD 讀取需要的專家權重，在本機執行大型語言模型。支援 DeepSeek V4、DeepSeek V4.1、Qwen3.8、Swift1.5 Qwen 及 MiMo（預覽版），提供內建聊天和 OpenAI 相容 API。

1.1.10 版新增第五個模型 `Swift1.5-Qwen3.8-Flash-Next`，從 `Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4` 的固定版本 `257cb509d72ecc07519be35a8b7558fde3b31b21` 下載。它共用 Qwen FP8 模型的推論引擎，但安裝目錄、儲存設定、Alias 與提示詞快取各自獨立。API model ID 為 `swift1.5-qwen3.8-flash-next-mxfp4`。App 固定顯示的下載容量為 **127,714,522,478 bytes**（約 **118.94 GiB**）：原有 59 個文字／MTP 檔案，加上 **897,864,704 bytes** 的視覺權重；不下載稽核檔案。兩款 Qwen 的進階設定相同，只保留 Alias、生成、記憶體、讀取工作數、提示詞快取、暖機與 MTP；其他執行選項一律使用固定值，包括 MTP 策略、排序專家 Prefill 與遮罩 QSA Prefill。MTP 預設開啟，輸出可能與不使用 MTP 時略有不同。

兩款 Qwen 都可透過 App Chat、`/v1/chat/completions` 與 `/v1/responses` 輸入靜態 PNG／JPEG／WebP 圖片。新安裝包含視覺權重；既有安裝請在 **Model → Verify and Repair** 補齊缺少或損壞的檔案，不會重載完好的文字權重。原版 Qwen 的安裝容量為 **126,189,355,659 bytes**（約 **117.52 GiB**）。圖片請求須關閉 MTP，且不使用提示詞快取。限制：每次 8 張、每檔 8 MiB、每次合計 32 MiB、每張解碼／縮放後最多 1,048,576 像素、每次最多 2,048 個圖片 tokens。Qwen 不支援影片、音訊或文件輸入。Swift 完整模型已通過紅／藍圖片辨識，以及圖片請求與中止前後的短文字結果一致性檢查；原版 Qwen 的完整視覺生成、廣泛圖片品質與視覺效能尚未驗證。

## 效能摘要

| 模型 | 晶片 | Prefill | Decode | 峰值記憶體 | 專家快取 Slots |
| --- | --- | ---: | ---: | ---: | ---: |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `Qwen3.8-Flash-Next-FP8` | M5 Pro | 99.1–153.7 tok/s | 8.5–10.6 tok/s | 18 GiB | 3072 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> 使用 v1.1.7 內建 Throughput 效能測試，輸入長度為 1,024 至 16,384 tokens。
>
> 詳細資料見[完整效能測試](#效能測試)。

## 快速開始

1. 從 [GitHub Releases](https://github.com/yanun0323/Whallm/releases) 下載 `Whallm-macOS-arm64.zip`，解壓縮後開啟 `Whallm.app`。
2. 開啟 **Model**，選擇模型並按 **Download Model**。App 會檢查儲存空間，中斷的下載可以繼續。
3. 開啟 **Server**，按 **Start Server**。
4. 到 **Chat** 選擇模型開始對話，或依下方範例連接 API 用戶端。

預設位址為 `http://127.0.0.1:11434`。模型在首次使用時載入；伺服器一次保留一個模型，依序處理生成請求。若聊天選單沒有剛安裝的模型，請重新啟動伺服器。

變更、升級步驟與已知限制見 [1.1.10 英文發布說明](Packaging/ReleaseNotes/1.1.10.md)。

## 使用需求

| 項目 | 需求 |
| --- | --- |
| Mac | Apple Silicon，macOS 15 或更新版本 |
| 統一記憶體 | 建議 64 GiB；實際用量依模型與設定而定 |
| 儲存裝置 | 高速內接、Thunderbolt 或 USB4 SSD |
| 可用空間 | App 會依模型及現有的部分下載計算需求 |
| 網路 | 下載模型與 App 更新時需要 |

App 不包含模型權重。DeepSeek V4.1 的專家與 Engram 檔案就需要約 **458 GiB**，另需常駐權重與中繼資料的空間。

## 功能

Whallm 將共用權重留在記憶體，從 SSD 讀取選中的專家。DeepSeek V4.1 的 Engram 與 Qwen 的 N-gram 資料也按需讀取。

| 模型 | 加速選項 |
| --- | --- |
| DeepSeek V4 | 按層處理輸入、專家合批計算、FP8 KV 快取、選用 ANE 投影運算與 DSpark |
| DeepSeek V4.1 | 按層處理輸入、專家合批、壓縮 KV／索引快取、只計算候選索引、CED 輸入處理、ANE 投影運算與 DSpark |
| Qwen3.8 | 輸入階段的專家分組、專家讀取完成後立即計算、QSA 快取壓縮、下一層預讀、ANE 投影運算與 MTP |

## 效能測試

以下 v1.1.7 紀錄使用 **Code** 素材，輸出上限為 **128 tokens**。表格未附各次程式版本與快取狀態，因此僅供參考，不能當成新增加速選項的控制變因比較。

TTFT 是等待第一個 token 的時間。Prefill 為輸入處理速度，Decode 為輸出生成速度，單位均為 tokens／秒。Peak MLX 是以 **GiB** 表示的 MLX 配置量，不是整台 Mac 的記憶體用量。App 匯出雖標為 GB，實際以 bytes 除以 1024³ 計算。

### M5 Pro

| 模型 | Slots | 輸入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
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

| 模型 | Slots | 輸入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

要測試自己的 Mac，開啟 **Throughput**，選擇已安裝模型、**Code** 或 **Novel** 素材、**1K–200K** 輸入長度，以及 **128、512、1024 或 4096** 的輸出上限。結果可複製為純文字、JSON 或 Markdown。整輪完成或取消後，App 會卸載測試模型。本機打包版另有 **Dry run**，只產生模擬結果。

SSD 速度、輸入長度、快取狀態與設定都會影響結果。

## 連接 Codex

先啟動 Whallm 伺服器，再將以下內容加入使用者層級的 `~/.codex/config.toml`：

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

將 `model` 設為 API model ID，或在 **Model → Advanced Settings** 設定的 Alias，儲存後重新啟動 Codex。Chat、Status、Throughput 與文字／Markdown 結果表格統一使用 Model 頁面的顯示名稱；顯示名稱不是 API model ID。API 請求與測試結果的 JSON 保留原本的識別碼。此範例使用預設本機位址，且未設定 API key。若在 Whallm 設定了金鑰，用戶端也要設定相同金鑰。詳見 [Codex 設定參考](https://developers.openai.com/codex/config-reference/)。

## API 與隱私

API 支援文字串流與工具呼叫，提供以下端點：

- `GET /healthz` 與 `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` 與 `POST /v1/completions`
- `POST /api/models/load` 與 `POST /api/models/unload`

三個生成端點支援選填 `seed`，接受 `0` 到 `4294967295` 的整數；省略或傳入 `null` 時，每次請求使用新的隨機值。Playground Chat 的 Seed 只套用到下一則訊息，送出後清空。固定 seed 有助於在相同輸入、模型、設定與執行環境下重現結果，但不保證跨版本、快取狀態或加速設定仍逐字一致。`temperature=0` 維持選取最高機率的結果。自 v1.1.8 起提供。

工具由用戶端執行，再回傳結果；推論 API 不會自動執行 MCP 工具。MiMo（預覽版）支援靜態圖片及有界 PCM WAV 音訊輸入，可透過 Chat／Responses 與 App 附件使用。JSON 本文仍限 **1 MiB**；經驗證的 `/api/assets` 上傳接受每檔最多 **8 MiB** 的 PNG／JPEG／WebP、PCM WAV，以及有範圍限制的 PDF／DOCX／PPTX／XLSX 文件。音訊輸入僅接受 24 kHz、16 位元 PCM WAV、單聲道或立體聲，每段最長 30 秒、每次最多兩段。音訊輸出、影片、`logprobs`、`response_format` 與 `stop` 仍不支援。詳見 [MiMo 合約與限制](docs/mimo-development.md)。

所有模型都能透過 `/v1/responses` 接收字串或由 `input_text` 段落組成的 `function_call_output.output` 陣列。文字會依序合併，保留原有空白與換行，不另外加入分隔符號。請在請求歷史中附上具有相同 `call_id` 的 `function_call`。多媒體支援仍依模型而定。自 v1.1.9 起提供。

推論在你的 Mac 上執行。下載、更新與 API 連線會使用網路。連接的用戶端可能將資料傳往其他服務；**Debug** 記錄可能包含完整輸入與工具結果。

## 驗證與限制

原始碼驗證：**674 項 Python 測試通過**；**196 項 Swift 測試執行，4 項略過、零失敗**。打包檢查涵蓋 App 與 ZIP 解壓副本的簽章、包內資源，以及無法存取建置目錄時的英文、簡體中文、繁體中文啟動。發布流程會對正式簽署、公證的成品及 GitHub 下載檔再次檢查。工具呼叫回歸測試使用固定模型輸出，不等於完整 Pi 或 Codex 用戶端對話驗證。

MiMo 仍為預覽版，透過固定版本的預製 artifact 提供文字／圖片、有界 WAV 音訊及有限文件支援；整體驗收尚未完成。聊天附件包含分類選檔、拖放、額度提示及上傳狀態。影片／影音同步與 MCP Agent 介面仍未完成；音訊及文件輸入有格式及資源限制。詳見[開發狀態](docs/mimo-development.md)。Qwen 圖片驗證範圍見上方說明。歷史 MTP 量測不是最終修正版本的量測，MTP 也不保證對所有工作都更快。部分加速路徑僅有小模型與元件測試；很長的輸入需要更多快取記憶體。

## 授權

Whallm 採用 [MIT License](LICENSE)。模型權重另有使用條款。Whallm 與 DeepSeek、Qwen 無隸屬關係。
