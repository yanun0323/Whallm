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

Whallm は、Apple Silicon Mac でメモリ容量を超える言語モデルを動かします。共通の重みはメモリに保持し、エキスパートの重みは必要なときだけ SSD から読み込みます。チャット画面と OpenAI 互換 API を備え、Codex などのクライアントに接続できます。

## 動作環境

- **Apple Silicon Mac、macOS 15 以降。**
- **メモリは 64 GiB を推奨**。実際の使用量はモデルと設定によります。
- **高速 SSD**：内蔵、Thunderbolt、または USB4。モデルの重みは別途ダウンロードします。アプリが途中までのダウンロードも含め、必要な空き容量を確認します。

## 対応モデル

| モデル | 対応入力 | モデルファイル容量 |
| --- | --- | --- |
| Swift1.5 Qwen3.8 | テキスト、画像 | ~119 GiB |
| Qwen3.8 | テキスト、画像 | ~118 GiB |
| DeepSeek V4.1 | テキスト | >458 GiB |
| DeepSeek V4 | テキスト | アプリで確認 |
| MiMo (プレビュー) | テキスト、画像、短い WAV 音声、一部の文書 | アプリで確認 |

DeepSeek V4.1 は**エキスパートと Engram だけで約 458 GiB** が必要で、ほかの重みも別に必要です。Qwen の容量は画像用の重みを含みます。詳細は[モデル設定と入力制限（英語）](docs/usage.md)を参照してください。

## はじめに

1. [GitHub Releases](https://github.com/yanun0323/Whallm/releases) から `Whallm-macOS-arm64.zip` をダウンロードし、展開して `Whallm.app` を開きます。
2. **Model** を開き、モデルを選んで **Download Model** を選びます。アプリは先に空き容量を確認します。ダウンロードが止まっても、続きから再開できます。
3. **Server** を開き、**Start Server** を選びます。
4. **Chat** を開いてモデルを選ぶか、下の例のように API クライアントを接続します。

サーバーの既定のアドレスは `http://127.0.0.1:11434` です。モデルは初めて使うときに読み込まれます。サーバーが読み込むモデルは一度に 1 つで、リクエストも 1 件ずつ処理します。インストールしたばかりのモデルが Chat の一覧にない場合は、サーバーを再起動してください。

## ベンチマーク概要

ユーザーが v1.1.13 で **Swift1.5-Qwen3.8-Flash-Next** を測定しました。入力は 16K トークン、回答は 512 トークンです。**Prefill** は Whallm が入力を読む速さ、**Decode** は回答を書く速さで、どちらも 1 秒あたりのトークン数です。

| チップ | キャッシュモード | Slots | Prefill (tok/s) | Decode (tok/s) | ピークメモリ (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | 自動 | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | 固定 | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | 自動 | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | 固定 | 2089 | 536.0 | 21.7 | 16.25 |

自動モードでは Whallm がエキスパートキャッシュのサイズを決めます。今回の小さな固定キャッシュはメモリを節約しますが、回答は遅くなります。Slots はキャッシュできるエキスパート数です。

**参考値です：**ユーザー提供の測定で、プロンプトの種類、全設定、キャッシュ状態、環境の詳細は記録されていません。条件を揃えたバージョン比較ではありません。ピークメモリは報告されたプロセスの使用量で、Mac 全体の使用量ではありません。

[全結果と測定方法（英語）](BENCHMARK.md)。自分の Mac で測るには、アプリの **Throughput** を開いてください。

## Codex の接続

Whallm のサーバーを起動してから、次の内容を `~/.codex/config.toml` に追加します。

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

`model` には API model ID か、**Model → Advanced Settings** で設定した Alias を指定します。その後、Codex を再起動してください。

API model ID は `GET /v1/models` または[モデル ID 一覧（英語）](docs/usage.md#api-model-ids)で確認できます。表示名とは異なります。

この例は既定のアドレスを使い、API key は使いません。Whallm で key を設定した場合は、クライアントにも同じ key を設定してください。

[Codex の設定リファレンス](https://developers.openai.com/codex/config-reference/)も参照してください。

## 制限とプライバシー

- **読み込めるモデルは一度に 1 つ、処理するリクエストも 1 件です。** 長いプロンプトにはより多くのキャッシュメモリが必要です。
- **MiMo はプレビュー**です。動画と音声出力には対応していません。Qwen は静止画に対応しますが、音声、動画、文書は受け付けません。
- ツールを実行するのはクライアントで、Whallm ではありません。Pi／Codex の完全なセッションや幅広いモデル品質の検証は未完了です。
- 速度は SSD、プロンプト、設定によって変わります。高速化機能がすべてのプロンプトで速くなるわけではありません。

**プライバシー：**推論はあなたの Mac で動きます。ダウンロード、更新、API 接続にはネットワークを使い、接続したクライアントがデータを外部に送る場合もあります。Debug ログにはプロンプト全文、ツールの結果、画像、API key は含まれませんが、ローカルパスやネットワークアドレスが含まれる場合があります。共有前に確認してください。

## 詳細ドキュメント

以下の詳細ドキュメントは英語です。

- [モデル設定、API ID、画像の制限、API リファレンス](docs/usage.md)
- [ベンチマークの全結果と過去の測定](BENCHMARK.md)
- [リリースノートとアップグレード手順](Packaging/ReleaseNotes/1.1.13.md)
- [MiMo プレビューの状況](docs/mimo-development.md)

## ライセンス

Whallm は [MIT License](LICENSE) で提供しています。モデルの重みには、それぞれ別の条件があります。Whallm は DeepSeek や Qwen とは関係ありません。
