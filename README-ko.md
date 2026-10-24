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

Whallm은 Apple Silicon Mac에서 메모리 용량보다 큰 언어 모델을 실행합니다. 공통 가중치는 메모리에 두고 전문가 가중치는 필요할 때만 SSD에서 읽습니다. 채팅 화면과 OpenAI 호환 API를 제공하며 Codex 같은 클라이언트에 연결할 수 있습니다.

## 실행 환경

- **Apple Silicon Mac, macOS 15 이상.**
- **메모리 64 GiB 권장**. 실제 사용량은 모델과 설정에 따라 다릅니다.
- **빠른 SSD**: 내장, Thunderbolt 또는 USB4. 모델 가중치는 따로 내려받습니다. 앱이 받다 만 파일을 포함해 필요한 공간을 확인합니다.

## 지원 모델

| 모델 | 지원 입력 | 모델 파일 크기 |
| --- | --- | --- |
| Swift1.5 Qwen3.8 | 텍스트, 이미지 | ~119 GiB |
| Qwen3.8 | 텍스트, 이미지 | ~118 GiB |
| DeepSeek V4.1 | 텍스트 | >458 GiB |
| DeepSeek V4 | 텍스트 | 앱에서 확인 |
| MiMo (미리 보기) | 텍스트, 이미지, 짧은 WAV 오디오, 일부 문서 | 앱에서 확인 |

DeepSeek V4.1은 **전문가와 Engram에만 약 458 GiB**가 필요하며 다른 가중치도 따로 필요합니다. Qwen 크기에는 이미지 가중치가 포함됩니다. 자세한 내용은 [모델 설정과 입력 제한(영어)](docs/usage.md)을 참고하세요.

## 시작하기

1. [GitHub Releases](https://github.com/yanun0323/Whallm/releases)에서 `Whallm-macOS-arm64.zip`을 내려받아 압축을 풀고 `Whallm.app`을 엽니다.
2. **Model**을 열고 모델을 고른 뒤 **Download Model**을 선택합니다. 앱이 먼저 남은 공간을 확인합니다. 내려받기가 멈춰도 이어서 받을 수 있습니다.
3. **Server**를 열고 **Start Server**를 선택합니다.
4. **Chat**을 열어 모델을 고르거나, 아래 예시처럼 API 클라이언트를 연결합니다.

서버의 기본 주소는 `http://127.0.0.1:11434`입니다. 모델은 처음 사용할 때 불러옵니다. 서버는 한 번에 모델 하나만 불러오고, 요청도 하나씩 처리합니다. 방금 설치한 모델이 Chat 목록에 없으면 서버를 다시 시작하세요.

## 벤치마크 요약

사용자가 v1.1.13에서 **Swift1.5-Qwen3.8-Flash-Next**를 16K 토큰 입력, 512 토큰 답변으로 측정했습니다. **Prefill**은 Whallm이 입력을 읽는 속도이고, **Decode**는 답변을 쓰는 속도입니다. 둘 다 초당 토큰 수입니다.

| 칩 | 캐시 모드 | Slots | Prefill (tok/s) | Decode (tok/s) | 최대 메모리 (GiB) |
| --- | --- | --- | --- | --- | --- |
| M2 Max | 자동 | 8192 | 286.4 | 19.4 | 31.71 |
| M2 Max | 고정 | 2089 | 289.6 | 16.5 | 16.99 |
| M5 Pro | 자동 | 8192 | 521.1 | 25.6 | 31.11 |
| M5 Pro | 고정 | 2089 | 536.0 | 21.7 | 16.25 |

자동 모드에서는 Whallm이 전문가 캐시 크기를 정합니다. 이번에 사용한 작은 고정 캐시는 메모리를 아끼지만 답변은 느려집니다. Slots는 캐시할 수 있는 전문가 수입니다.

**참고용입니다:** 사용자 제공 측정으로, 프롬프트 종류, 전체 설정, 캐시 상태와 전체 환경은 기록되지 않았습니다. 조건을 통제한 버전 비교가 아닙니다. 최대 메모리는 보고된 프로세스 사용량이며 Mac 전체 사용량이 아닙니다.

[전체 결과와 측정 방법(영어)](BENCHMARK.md). 내 Mac에서 측정하려면 앱의 **Throughput**을 여세요.

## Codex 연결

Whallm 서버를 시작한 뒤, 다음 내용을 `~/.codex/config.toml`에 추가하세요.

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

`model`에는 API model ID나 **Model → Advanced Settings**에서 정한 Alias를 넣으세요. 그런 다음 Codex를 다시 시작하세요.

API model ID는 `GET /v1/models` 또는 [모델 ID 표(영어)](docs/usage.md#api-model-ids)에서 확인할 수 있습니다. 표시 이름과는 다릅니다.

이 예시는 기본 주소를 쓰고 API key는 쓰지 않습니다. Whallm에서 key를 정했다면 클라이언트에도 같은 key를 넣으세요.

[Codex 설정 참고 문서](https://developers.openai.com/codex/config-reference/)도 확인하세요.

## 제한과 개인정보

- **한 번에 모델 하나만 불러오고 요청 하나만 처리합니다.** 긴 프롬프트에는 더 많은 캐시 메모리가 필요합니다.
- **MiMo는 미리 보기**이며 동영상과 오디오 출력을 지원하지 않습니다. Qwen은 정지 이미지를 지원하지만 오디오, 동영상, 문서는 받지 않습니다.
- 도구는 Whallm이 아닌 클라이언트가 실행합니다. 전체 Pi/Codex 세션과 폭넓은 모델 품질 검증은 아직 완료되지 않았습니다.
- 속도는 SSD, 프롬프트, 설정에 따라 달라집니다. 가속 기능이 모든 프롬프트에서 더 빠른 것은 아닙니다.

**개인정보:** 추론은 내 Mac에서 실행됩니다. 내려받기, 업데이트, API 연결에는 네트워크를 사용하며 연결한 클라이언트가 데이터를 외부로 보낼 수 있습니다. Debug 로그에는 프롬프트 전문, 도구 결과, 이미지, API key가 포함되지 않지만 로컬 경로와 네트워크 주소가 들어갈 수 있습니다. 공유 전에 확인하세요.

## 상세 문서

아래 상세 문서는 영어로 제공됩니다.

- [모델 설정, API ID, 이미지 제한, API 참고 문서](docs/usage.md)
- [전체 벤치마크와 이전 결과](BENCHMARK.md)
- [릴리스 노트와 업그레이드 방법](Packaging/ReleaseNotes/1.1.13.md)
- [MiMo 미리 보기 현황](docs/mimo-development.md)

## 라이선스

Whallm은 [MIT License](LICENSE)를 따릅니다. 모델 가중치에는 각자의 조건이 있습니다. Whallm은 DeepSeek나 Qwen과 관련이 없습니다.
