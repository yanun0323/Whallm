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

Whallm は、必要なエキスパートを SSD から読み込むことで、Apple Silicon Mac 上で大規模言語モデルを動かします。DeepSeek V4、DeepSeek V4.1、Qwen3.8 に対応し、チャット画面と OpenAI 互換 API を備えています。

## ベンチマーク概要

| モデル | チップ | Prefill | Decode | ピークメモリ | エキスパートキャッシュのスロット数 |
| --- | --- | ---: | ---: | ---: | ---: |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `Qwen3.8-Flash-Next-FP8` | M5 Pro | 99.1–153.7 tok/s | 8.5–10.6 tok/s | 18 GiB | 3072 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> v1.1.7 内蔵の Throughput ベンチマークで、入力長 1,024～16,384 トークンを測定しました。
>
> 詳しくは[ベンチマークの全結果](#ベンチマーク)を参照してください。

## はじめに

1. [GitHub Releases](https://github.com/yanun0323/Whallm/releases) から `Whallm-macOS-arm64.zip` をダウンロードし、展開して `Whallm.app` を開きます。
2. **Model** でモデルを選び、**Download Model** を押します。アプリが空き容量を確認し、中断したダウンロードは再開できます。
3. **Server** で **Start Server** を押します。
4. **Chat** でモデルを選ぶか、下の設定例で API クライアントを接続します。

既定のアドレスは `http://127.0.0.1:11434` です。モデルは最初の利用時に読み込まれます。サーバーは一度に 1 つのモデルを保持し、生成リクエストを順番に処理します。新しくインストールしたモデルがチャットの選択肢に表示されない場合は、サーバーを再起動してください。

公開済みのダウンロードには、このソースコードの機能がまだ含まれていない場合があります。

## 動作環境

| 項目 | 要件 |
| --- | --- |
| Mac | Apple Silicon、macOS 15 以降 |
| ユニファイドメモリ | 64 GiB 推奨。使用量はモデルと設定によって変わります |
| ストレージ | 高速な内蔵 SSD、Thunderbolt SSD、または USB4 SSD |
| 空き容量 | ダウンロード済みの部分も考慮して、アプリがモデルごとの必要容量を計算します |
| ネットワーク | モデルのダウンロードとアプリの更新に必要です |

モデルの重みはアプリに含まれません。DeepSeek V4.1 はエキスパートと Engram のファイルだけで約 **458 GiB** を使い、さらに共通の重みとメタデータが必要です。

## 機能

Whallm は共通の重みをメモリに保持し、選択されたエキスパートを SSD から読み込みます。DeepSeek V4.1 の Engram と Qwen の N-gram も、必要な行だけを読み込みます。

| モデル | 高速化の選択肢 |
| --- | --- |
| DeepSeek V4 | レイヤー単位の入力処理、エキスパートの一括計算、FP8 KV キャッシュ、任意の ANE 射影と DSpark |
| DeepSeek V4.1 | レイヤー単位の入力処理、エキスパートの一括計算、圧縮した KV・インデックスキャッシュ、候補だけのインデックス評価、CED 入力処理、ANE 射影、DSpark |
| Qwen3.8 | 入力処理時のエキスパートの一括計算、読み込みが終わったエキスパートからの計算、QSA キャッシュ圧縮、次レイヤーの先読み、ANE 射影、MTP |

## ベンチマーク

以下は、**Code** のコンテキストと **128 トークン**の出力上限を使った v1.1.7 の記録です。各行にはビルドのリビジョンとキャッシュ状態が記録されていないため、新しい高速化設定の条件を揃えた比較ではなく、参考値として扱ってください。

TTFT は最初のトークンが出るまでの待ち時間です。Prefill は入力処理、Decode は出力生成の速度で、単位はトークン/秒です。Peak MLX は MLX のメモリ割り当て量を **GiB** で示し、Mac 全体のメモリ使用量ではありません。アプリの出力では GB と表示されますが、計算には 1024³ を使っています。

### M5 Pro

| モデル | Slots | 入力トークン | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
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

| モデル | Slots | 入力トークン | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

自分の Mac で測定するには、**Throughput** を開き、インストール済みのモデル、**Code** または **Novel**、**1K～200K** の入力長、**128・512・1024・4096** の出力上限を選びます。結果はプレーンテキスト、JSON、Markdown でコピーできます。測定の完了時またはキャンセル時に、アプリは測定用モデルをメモリから解放します。ローカルビルドでは、模擬結果を出す **Dry run** も使えます。

SSD の速度、プロンプトの長さ、キャッシュ状態、設定によって結果は変わります。

## Codex の接続

Whallm のサーバーを起動し、ユーザー設定の `~/.codex/config.toml` に以下を追加します。

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

`model` はWhallm に表示される API モデル ID、または Alias に設定し、Codex を再起動します。この例は既定のローカルアドレスを使い、API キーを設定しない場合のものです。Whallm にキーを設定した場合は、クライアントにも同じキーを設定してください。[Codex の設定リファレンス](https://developers.openai.com/codex/config-reference/)も参照できます。

## API とプライバシー

API はテキストのストリーミングとツール呼び出しに対応し、以下のエンドポイントを提供します。

- `GET /healthz` と `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` と `POST /v1/completions`
- `POST /api/models/load` と `POST /api/models/unload`

現在のソースでは、3つの生成エンドポイントで `0`〜`4294967295` の整数を任意の `seed` として指定できます。省略または `null` の場合、リクエストごとに新しい乱数シードを使います。Playground Chat の Seed は次のメッセージだけに適用され、送信後に空になります。固定シードは同じ入力、モデル、設定、実行環境での再現に役立ちますが、バージョン、キャッシュ状態、高速化設定が異なる場合の完全一致は保証しません。`temperature=0` は引き続き最大確率の候補を選びます。v1.1.8 から利用できます。

ツールはクライアント側で実行し、結果を返します。推論 API が MCP ツールを自動実行することはありません。開発中のソースでは MiMo の静止画像と制限付き PCM WAV 音声入力を Chat／Responses とアプリの添付機能から利用できます。JSON 本文は引き続き **1 MiB** までで、認証付き `/api/assets` は PNG／JPEG／WebP、PCM WAV と、対応範囲を限定した PDF／DOCX／PPTX／XLSX 文書を1ファイル **8 MiB** まで受け付けます。音声入力は 24 kHz・16ビット PCM WAV、モノラル／ステレオ、1本30秒以内、1リクエスト2本までです。音声出力、動画、`logprobs`、`response_format`、`stop` は未対応です。[MiMo の仕様と制限](docs/mimo-development.md)を参照してください。

推論は Mac 上で実行されます。ダウンロード、更新、API 接続にはネットワークを使います。接続先のクライアントがデータを外部に送信する場合があります。**Debug** ログにはプロンプトやツールの結果が全文で含まれることがあります。

## 検証と制限

v1.1.8 のソースは、**Python テスト 464 件**、**Swift テスト 135 件**に合格しました。アプリ本体と ZIP から展開したアプリの両方で、署名の検証と、ビルドディレクトリにアクセスできない環境での英語・簡体字中国語・繁体字中国語による起動確認に合格しました。

指定された3つのテキストモデルに加え、開発中のソースには MiMo のテキスト／画像、制限付き WAV 音声入力と限定的な文書対応、およびリビジョンを固定した完全な変換済み artifact があります。Python 624件、Swift 171件（3件スキップ）のテストが通過していますが、今回の MiMo 対応はローカルでパッケージ化済みですが、全体の受け入れ検証は未完了です。チャットの添付 UI に種類別ファイル選択、ドラッグ＆ドロップ、容量表示、アップロード状況を追加しました。動画／音声映像同期と MCP Agent 画面は未完成で、音声・文書入力には形式とリソースの制限があります。[開発状況](docs/mimo-development.md)を参照してください。新しい高速化処理は小規模モデルと部品単位でテストしていますが、実際のモデルでの速度と品質の比較は未完了です。長いプロンプトには、より多くのキャッシュメモリが必要です。

## ライセンス

Whallm は [MIT ライセンス](LICENSE)で公開しています。モデルの重みには別の利用条件が適用されます。Whallm は DeepSeek および Qwen と提携していません。
