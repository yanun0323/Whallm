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

Whallm은 Apple Silicon Mac에서 대규모 언어 모델을 실행합니다. 공통 가중치는 메모리에 두고, 토큰마다 필요한 전문가만 SSD에서 읽습니다. 그래서 메모리보다 훨씬 큰 모델도 실행할 수 있습니다. 채팅 창과 OpenAI 호환 API를 제공합니다.

지원 모델: DeepSeek V4, DeepSeek V4.1, Qwen3.8, Swift1.5 Qwen3.8, MiMo(미리 보기).

## 벤치마크 요약

| 모델 | 칩 | Prefill | Decode | 최대 메모리 | 전문가 캐시 Slots |
| --- | --- | ---: | ---: | ---: | ---: |
| `Swift1.5-Qwen3.8-Flash-Next` | M5 Pro | 571.3–613.3 tok/s | 14.6–17.5 tok/s | 19 GiB | 2089 |
| `Qwen3.8-Flash-Next-FP8` | M5 Pro | 484.4–547.3 tok/s | 15.0–17.2 tok/s | 18 GiB | 2089 |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> 내장 Throughput 테스트로 측정했습니다. 두 Qwen 모델은 v1.1.11에서 MTP를 켜고 입력 4,096~16,384 토큰으로, DeepSeek는 v1.1.7에서 입력 1,024~16,384 토큰으로 측정했습니다. 버전이 다르므로 이 행들을 직접 비교하지 마세요.
>
> 자세한 내용은 [전체 벤치마크](#벤치마크)를 참고하세요.

## 시작하기

1. [GitHub Releases](https://github.com/yanun0323/Whallm/releases)에서 `Whallm-macOS-arm64.zip`을 내려받아 압축을 풀고 `Whallm.app`을 엽니다.
2. **Model**을 열고 모델을 고른 뒤 **Download Model**을 누릅니다. 앱이 먼저 남은 공간을 확인합니다. 다운로드가 멈춰도 이어서 받을 수 있습니다.
3. **Server**를 열고 **Start Server**를 누릅니다.
4. **Chat**에서 모델을 고르거나, 아래 예시처럼 API 클라이언트를 연결합니다.

서버 주소는 `http://127.0.0.1:11434`입니다. 모델은 처음 사용할 때 불러옵니다. 서버는 한 번에 모델 하나만 불러오고, 요청도 한 번에 하나씩 처리합니다. 방금 설치한 모델이 Chat 목록에 없으면 서버를 다시 시작하세요.

변경 사항, 업그레이드 방법, 알려진 제한은 [1.1.11 릴리스 노트](Packaging/ReleaseNotes/1.1.11.md)에 있습니다.

## 실행 환경

| 항목 | 조건 |
| --- | --- |
| Mac | Apple Silicon, macOS 15 이상 |
| 메모리 | 64 GiB 권장. 실제 사용량은 모델과 설정에 따라 다릅니다 |
| 저장 장치 | 빠른 내장, Thunderbolt 또는 USB4 SSD |
| 남은 공간 | 받다 만 파일을 포함해 모델마다 필요한 공간을 앱이 보여 줍니다 |
| 네트워크 | 모델과 앱 업데이트를 내려받을 때 필요합니다 |

모델 가중치는 앱에 들어 있지 않습니다. DeepSeek V4.1은 전문가와 Engram 파일만 약 **458 GiB**가 필요하고, 다른 가중치도 따로 필요합니다.

## Qwen 모델

### Swift1.5-Qwen3.8-Flash-Next

1.1.10에서 추가되었습니다.

- API 모델 ID: `swift1.5-qwen3.8-flash-next-mxfp4`
- 출처: `Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4`, 리비전 `257cb509d72ecc07519be35a8b7558fde3b31b21`
- 다운로드 크기: **127,714,522,478 bytes**(약 **118.94 GiB**). 텍스트와 MTP 파일 59개, 이미지 가중치 **897,864,704 bytes**가 포함됩니다. 감사용 파일은 내려받지 않습니다.
- Qwen3.8 FP8과 같은 추론 엔진을 쓰지만, 폴더, 설정, Alias, 프롬프트 캐시는 따로 둡니다.

### 설정

- 두 Qwen 모델의 **Advanced Settings**는 같습니다. Alias, 생성, 메모리, 읽기 작업자 수, Prompt Cache, 워밍업, MTP만 보이고 나머지는 고정값을 씁니다.
- MTP는 기본으로 켜져 있습니다. 대부분 출력이 빨라지지만 모든 프롬프트에서 빨라지지는 않습니다. MTP를 껐을 때와 출력이 조금 다를 수 있습니다.

### 이미지

- 두 Qwen 모델 모두 Chat, `/v1/chat/completions`, `/v1/responses`에서 정지 PNG, JPEG, WebP 이미지를 입력할 수 있습니다.
- MTP를 켜든 끄든 이미지를 보낼 수 있습니다. 전체 Swift1.5 모델에서 MTP가 만든 토큰은 모두 모델이 스스로 고를 토큰과 같았고, 테스트 이미지 두 장에서 출력이 약 1.4배 빨랐습니다. 이미지 요청은 프롬프트 캐시를 쓰지 않습니다.
- 요청당 제한: 이미지 8장, 파일당 8 MiB, 합계 32 MiB, 크기 조정 후 이미지당 1,048,576 픽셀, 이미지 토큰 2,048개.
- 새로 설치하면 이미지 가중치가 포함됩니다. 예전 설치에서는 **Model → Verify and Repair**를 선택하세요. 없거나 손상된 파일만 내려받습니다. Qwen3.8 FP8 전체 설치는 **126,189,355,659 bytes**(약 **117.52 GiB**)입니다.
- Qwen은 동영상, 오디오, 문서 입력을 지원하지 않습니다.
- 테스트한 것: 전체 Swift1.5 모델이 빨간색과 파란색 이미지를 알아봤고, 이미지 요청과 취소 전후에 짧은 텍스트 답이 같게 유지됐습니다. 전체 Qwen3.8 FP8 모델의 이미지 생성, 전반적인 이미지 품질, 이미지 처리 속도는 아직 테스트하지 않았습니다.

## 기능

Whallm은 공통 가중치를 메모리에 두고, 선택된 전문가를 SSD에서 읽습니다. DeepSeek V4.1의 Engram 행과 Qwen의 N-gram 행도 필요할 때만 읽습니다.

| 모델 | 속도 향상 방법 |
| --- | --- |
| DeepSeek V4 | 층별 입력 처리, 전문가 계산 묶음 처리, FP8 KV 캐시, 선택형 ANE 투영과 DSpark |
| DeepSeek V4.1 | 층별 입력 처리, 전문가 묶음 처리, 압축한 KV와 인덱스 캐시, 후보 인덱스만 점수 계산, CED 입력 처리, ANE 투영과 DSpark |
| Qwen3.8 | 입력 처리 중 전문가 묶기, 읽기가 끝나는 대로 전문가 계산, QSA 캐시 압축, 다음 층 미리 읽기, MTP |

## 벤치마크

이미지 표를 제외한 행은 **Code** 입력과 **128 토큰** 출력 제한을 사용했습니다. 이 행들은 빌드 리비전과 캐시 상태는 기록하지 않았으므로, 버전이나 속도 향상 방법을 비교한 결과가 아니라 참고값으로 보세요.

- **TTFT**: 첫 토큰이 나올 때까지 걸린 시간.
- **Prefill**: 입력 처리 속도. **Decode**: 출력 생성 속도. 둘 다 초당 토큰 수입니다.
- **Peak MLX**: MLX가 사용한 메모리의 최댓값(**GiB**). Mac 전체 메모리 사용량이 아닙니다. 예전 앱 내보내기는 이 값을 GB로 표시하지만 실제로는 GiB로 계산합니다.

### M5 Pro

| 모델 | 버전 | Slots | 입력 토큰 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Swift1.5 Qwen3.8 | 1.1.11 | 2089 | 4096 | 7140.3 | 573.6 | 14.6 | 16.93 |
| Swift1.5 Qwen3.8 | 1.1.11 | 2089 | 8192 | 13357.9 | 613.3 | 17.5 | 17.83 |
| Swift1.5 Qwen3.8 | 1.1.11 | 2089 | 16384 | 28679.3 | 571.3 | 14.9 | 18.81 |
| Qwen3.8 | 1.1.11 | 2089 | 4096 | 8456.6 | 484.4 | 15.0 | 16.58 |
| Qwen3.8 | 1.1.11 | 2089 | 8192 | 14968.3 | 547.3 | 17.2 | 17.41 |
| Qwen3.8 | 1.1.11 | 2089 | 16384 | 32572.7 | 503.0 | 15.0 | 18.38 |
| DeepSeek V4 | 1.1.7 | 1152 | 1024 | 19112.3 | 53.6 | 7.7 | 22.77 |
| DeepSeek V4 | 1.1.7 | 1152 | 4096 | 24498.6 | 167.2 | 5.9 | 22.80 |
| DeepSeek V4 | 1.1.7 | 1152 | 8192 | 42137.5 | 194.4 | 6.8 | 22.83 |
| DeepSeek V4 | 1.1.7 | 1152 | 16384 | 81517.9 | 201.0 | 6.3 | 22.90 |

1.1.11 행은 MTP를 켜고 temperature 0.0, seed 42로 측정했습니다. 이전 측정은 [BENCHMARK.md](BENCHMARK.md)에 있습니다.

### M5 Pro: Swift1.5 이미지 프롬프트와 MTP

1.1.11 소스로 측정했습니다. 출력 160 토큰, temperature 0, Prompt Cache 끔, 전문가 캐시 3,084 slots(7.5 GiB). 각 값은 3회 측정의 중앙값입니다. 초안 채택률은 MTP가 추측한 토큰 중 모델이 받아들인 비율입니다.

| 이미지 | 이미지 토큰 | 입력 토큰 | MTP | Decode (tok/s) | 첫 토큰 (s) | 초안 채택률 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 집 그림 | 384 | 404 | 끔 | 12.31 | 3.78 | — |
| 집 그림 | 384 | 404 | 켬 | 17.13 | 3.56 | 67% |
| 막대 그래프 | 256 | 281 | 끔 | 11.92 | 3.19 | — |
| 막대 그래프 | 256 | 281 | 켬 | 17.07 | 3.09 | 74% |

MTP를 켜면 출력이 각각 1.38배, 1.43배 빨랐고 첫 토큰도 조금 더 빨리 나왔습니다.

### M2 Max

| 모델 | 버전 | Slots | 입력 토큰 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 1.1.7 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 1.1.7 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 1.1.7 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 1.1.7 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1.1.7 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

내 Mac에서 측정하려면 **Throughput**을 열고 다음을 고르세요.

- 설치된 모델
- **Code** 또는 **Novel** 입력
- **1K~200K** 입력 길이
- **128, 512, 1024, 4096** 중 하나의 출력 제한

결과는 일반 텍스트, JSON, Markdown으로 복사할 수 있습니다. 측정이 끝나거나 취소되면 앱은 테스트 모델을 내립니다. 로컬 빌드에는 시뮬레이션 결과를 보여 주는 **Dry run**도 있습니다.

결과는 SSD 속도, 프롬프트 길이, 캐시 상태, 설정에 따라 달라집니다.

## Codex 연결

Whallm 서버를 시작한 뒤 `~/.codex/config.toml`에 다음을 추가합니다.

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

`model`을 API 모델 ID나 **Model → Advanced Settings**에서 정한 Alias로 바꾼 뒤 Codex를 다시 시작하세요.

- Chat, Status, Throughput에 보이는 이름은 표시 이름이며 API 모델 ID가 아닙니다. 요청과 벤치마크 JSON에는 계속 API 모델 ID를 씁니다.
- 이 예시는 기본 주소를 쓰고 API 키가 없습니다. Whallm에 키를 설정했다면 클라이언트에도 같은 키를 설정하세요.

자세한 내용은 [Codex 설정 참고 문서](https://developers.openai.com/codex/config-reference/)를 보세요.

## API와 개인정보

API는 텍스트 스트리밍과 도구 호출을 지원합니다. 엔드포인트는 다음과 같습니다.

- `GET /healthz`, `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`, `POST /v1/completions`
- `POST /api/models/load`, `POST /api/models/unload`

**Seed**(v1.1.8부터): 세 생성 엔드포인트 모두 `0`~`4294967295` 범위의 `seed`를 선택적으로 받습니다. 빼거나 `null`을 보내면 매번 새 무작위 seed를 씁니다. Chat에서 Seed는 다음 메시지에만 적용됩니다. 고정 seed로 결과를 다시 얻으려면 프롬프트, 모델, 설정, 실행 환경이 모두 같아야 합니다. 버전, 캐시 상태, 속도 향상 설정이 다르면 텍스트가 달라질 수 있습니다. `temperature=0`은 항상 확률이 가장 높은 토큰을 고릅니다.

**도구**: 도구를 실행하고 결과를 돌려보내는 것은 클라이언트입니다. Whallm은 MCP 도구를 스스로 실행하지 않습니다. v1.1.9부터 모든 모델의 `/v1/responses`는 `function_call_output.output`으로 문자열 또는 `input_text` 목록을 받습니다. 각 부분은 순서대로, 사이에 아무것도 넣지 않고 이어 붙입니다. 요청 기록에는 같은 `call_id`를 가진 `function_call`을 넣어 주세요.

**MiMo(미리 보기)** 는 Chat, Responses, 앱 첨부 파일로 정지 이미지와 짧은 PCM WAV 오디오를 입력받습니다.

- JSON 요청 본문: 최대 **1 MiB**.
- `/api/assets` 업로드(인증 필요): PNG, JPEG, WebP, PCM WAV와 작은 PDF, DOCX, PPTX, XLSX 파일. 파일당 최대 **8 MiB**.
- 오디오: 24 kHz, 16-bit PCM WAV, 모노 또는 스테레오. 클립당 최대 30초, 요청당 최대 2개.
- 지원하지 않음: 오디오 출력, 동영상, `logprobs`, `response_format`, `stop`.

자세한 내용은 [MiMo 사양과 제한](docs/mimo-development.md)을 보세요.

**개인정보**: 모델은 내 Mac에서 실행됩니다. Whallm은 다운로드, 업데이트, API 연결에만 네트워크를 씁니다. 연결한 클라이언트는 데이터를 다른 곳으로 보낼 수 있습니다. **Debug** 로그에는 프롬프트와 도구 결과 전체가 남을 수 있습니다.

## 검증과 제한

- 테스트: **Python 테스트 705개 통과**, **Swift 테스트 193개 실행, 4개 건너뜀, 실패 없음**.
- 패키징 검사는 서명, 포함된 파일, 빌드 폴더에 접근할 수 없는 상태에서 영어·중국어 간체·중국어 번체로 실행되는지를 확인합니다. 앱과 압축을 푼 ZIP 모두 검사합니다. 릴리스 때는 서명과 공증을 마친 다운로드 파일로 다시 검사합니다.
- 도구 호출 테스트는 고정된 모델 출력을 쓰며, 완전한 Pi나 Codex 세션이 아닙니다.
- MiMo는 미리 보기입니다. 고정된 사전 처리 파일을 쓰며 텍스트, 이미지, 짧은 WAV 오디오, 일부 문서를 지원합니다. 동영상, 오디오·동영상 동기화, MCP Agent 화면은 아직 완성되지 않았습니다. 자세한 내용은 [개발 현황](docs/mimo-development.md)을 보세요.
- 예전 MTP 측정은 지금의 실행 환경보다 앞선 것입니다. MTP가 모든 프롬프트에서 빠르지는 않습니다.
- 일부 속도 향상 방법은 작은 모델이나 개별 부품에서만 테스트했습니다.
- 아주 긴 프롬프트는 캐시 메모리가 더 필요합니다.

## 라이선스

Whallm은 [MIT 라이선스](LICENSE)로 배포합니다. 모델 가중치에는 각자의 이용 조건이 있습니다. Whallm은 DeepSeek나 Qwen과 관련이 없습니다.
