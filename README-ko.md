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

Whallm은 필요한 전문가를 SSD에서 읽어 Apple Silicon Mac에서 대규모 언어 모델을 실행합니다. DeepSeek V4, DeepSeek V4.1, Qwen3.8을 지원하며, 채팅 화면과 OpenAI 호환 API를 제공합니다.

## 벤치마크 요약

| 모델 | 칩 | Prefill | Decode | 최대 메모리 | 전문가 캐시 슬롯 |
| --- | --- | ---: | ---: | ---: | ---: |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `Qwen3.8-Flash-Next-FP8` | M5 Pro | 99.1–153.7 tok/s | 8.5–10.6 tok/s | 18 GiB | 3072 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> v1.1.7 내장 Throughput 벤치마크에서 입력 길이 1,024~16,384토큰으로 측정했습니다.
>
> 자세한 내용은 [전체 벤치마크](#벤치마크)를 참고하세요.

## 시작하기

1. [GitHub Releases](https://github.com/yanun0323/Whallm/releases)에서 `Whallm-macOS-arm64.zip`을 다운로드하고, 압축을 푼 뒤 `Whallm.app`을 엽니다.
2. **Model**에서 모델을 고르고 **Download Model**을 누릅니다. 앱이 저장 공간을 확인하며, 중단된 다운로드는 이어받을 수 있습니다.
3. **Server**에서 **Start Server**를 누릅니다.
4. **Chat**에서 모델을 선택하거나, 아래 예제로 API 클라이언트를 연결합니다.

기본 주소는 `http://127.0.0.1:11434`입니다. 모델은 처음 사용할 때 로드됩니다. 서버는 한 번에 모델 하나를 메모리에 유지하며, 생성 요청을 순서대로 처리합니다. 새로 설치한 모델이 채팅 선택 목록에 없으면 서버를 다시 시작하세요.

공개된 다운로드에는 여기서 설명하는 소스 코드의 기능이 아직 포함되지 않았을 수 있습니다.

## 실행 환경

| 항목 | 요구 사항 |
| --- | --- |
| Mac | Apple Silicon, macOS 15 이상 |
| 통합 메모리 | 64 GiB 권장. 사용량은 모델과 설정에 따라 달라집니다 |
| 저장 장치 | 빠른 내장 SSD, Thunderbolt SSD 또는 USB4 SSD |
| 여유 공간 | 앱이 부분 다운로드를 포함해 모델별로 필요한 공간을 계산합니다 |
| 네트워크 | 모델 다운로드와 앱 업데이트에 필요합니다 |

모델 가중치는 앱에 포함되지 않습니다. DeepSeek V4.1의 전문가 및 Engram 파일만 약 **458 GiB**가 필요하며, 공통 가중치와 메타데이터 공간도 추가로 필요합니다.

## 기능

Whallm은 공통 가중치를 메모리에 유지하고, 선택된 전문가를 SSD에서 읽습니다. DeepSeek V4.1 Engram과 Qwen N-gram도 필요한 행만 읽습니다.

| 모델 | 가속 옵션 |
| --- | --- |
| DeepSeek V4 | 레이어별 입력 처리, 전문가 일괄 계산, FP8 KV 캐시, 선택형 ANE 투영 및 DSpark |
| DeepSeek V4.1 | 레이어별 입력 처리, 전문가 일괄 계산, 압축 KV·인덱스 캐시, 후보만 인덱스 점수 계산, CED 입력 처리, ANE 투영, DSpark |
| Qwen3.8 | 입력 처리 시 전문가 일괄 계산, 읽기가 끝난 전문가부터 계산, QSA 캐시 압축, 다음 레이어 미리 읽기, ANE 투영, MTP |

## 벤치마크

아래는 **Code** 문맥과 **128토큰** 출력 제한으로 실행한 v1.1.7의 기록입니다. 각 행에 빌드 리비전과 캐시 상태가 기록되지 않았으므로, 새 가속 옵션의 조건을 통제한 비교가 아닌 참고 결과입니다.

TTFT는 첫 토큰이 나올 때까지의 대기 시간입니다. Prefill은 입력 처리 속도, Decode는 출력 생성 속도이며 둘 다 초당 토큰 수로 표시합니다. Peak MLX는 MLX가 할당한 메모리를 **GiB**로 나타내며, Mac 전체의 메모리 사용량이 아닙니다. 앱 내보내기에서는 GB로 표시하지만 바이트를 1024³으로 나누어 계산합니다.

### M5 Pro

| 모델 | Slots | 입력 토큰 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
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

| 모델 | Slots | 입력 토큰 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

내 Mac에서 측정하려면 **Throughput**을 열고 설치된 모델, **Code** 또는 **Novel**, **1K~200K** 입력 길이, **128·512·1024·4096** 출력 제한을 선택하세요. 결과는 일반 텍스트, JSON 또는 Markdown으로 복사할 수 있습니다. 실행이 끝나거나 취소되면 앱이 측정용 모델을 메모리에서 해제합니다. 로컬 빌드에서는 모의 결과를 만드는 **Dry run**도 사용할 수 있습니다.

SSD 속도, 프롬프트 길이, 캐시 상태, 설정에 따라 결과가 달라집니다.

## Codex 연결

Whallm 서버를 시작한 뒤, 사용자 설정 파일 `~/.codex/config.toml`에 다음 내용을 추가합니다.

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

`model`을 Whallm에 표시된 API 모델 ID 또는 Alias로 설정하고 Codex를 다시 시작하세요. 이 예제는 기본 로컬 주소를 사용하며 API 키가 없는 경우를 기준으로 합니다. Whallm에서 키를 설정했다면 클라이언트에도 같은 키를 설정하세요. [Codex 설정 문서](https://developers.openai.com/codex/config-reference/)도 참고할 수 있습니다.

## API와 개인정보

API는 텍스트 스트리밍과 도구 호출을 지원하며, 다음 엔드포인트를 제공합니다.

- `GET /healthz` 및 `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` 및 `POST /v1/completions`
- `POST /api/models/load` 및 `POST /api/models/unload`

현재 소스의 세 생성 엔드포인트는 `0`부터 `4294967295`까지의 정수를 선택적 `seed`로 받습니다. 생략하거나 `null`을 보내면 요청마다 새 무작위 시드를 사용합니다. Playground Chat의 Seed는 다음 메시지에만 적용되며 전송 후 비워집니다. 고정 시드는 동일한 입력, 모델, 설정, 실행 환경에서 결과를 재현하는 데 도움이 되지만 버전, 캐시 상태, 가속 설정이 달라지면 완전히 같은 텍스트를 보장하지 않습니다. `temperature=0`은 계속 가장 확률이 높은 후보를 선택합니다. v1.1.8부터 사용할 수 있습니다.

도구는 클라이언트가 실행하고 결과를 돌려줍니다. 추론 API는 MCP 도구를 자동 실행하지 않습니다. 현재 개발 소스는 Chat／Responses와 앱 첨부 기능을 통해 MiMo 정지 이미지와 제한적인 PCM WAV 오디오 입력을 지원합니다. JSON 본문은 계속 **1 MiB**로 제한되며, 인증된 `/api/assets` 업로드는 PNG／JPEG／WebP, PCM WAV와 제한적으로 지원하는 PDF／DOCX／PPTX／XLSX 문서를 파일당 **8 MiB**까지 받습니다. 오디오 입력은 24 kHz, 16비트 PCM WAV, 모노／스테레오, 파일당 최대 30초, 요청당 최대 2개로 제한됩니다. 오디오 출력, 동영상, `logprobs`, `response_format`, `stop`은 아직 지원하지 않습니다. [MiMo 계약과 제한](docs/mimo-development.md)을 참고하세요.

현재 개발 소스에서는 모든 모델의 `/v1/responses`에서 `function_call_output.output`을 문자열 또는 `input_text` 배열로 보낼 수 있습니다. 텍스트는 순서대로 합쳐지며 기존 공백과 줄바꿈을 유지하고 구분자를 추가하지 않습니다. 요청 기록에 같은 `call_id`를 가진 `function_call`을 포함하세요. 미디어 지원 범위는 여전히 모델에 따라 다릅니다. 이 수정은 v1.1.8에 포함되지 않았습니다.

추론은 Mac에서 실행됩니다. 다운로드, 업데이트, API 연결에는 네트워크를 사용합니다. 연결된 클라이언트가 데이터를 외부로 보낼 수 있으며, **Debug** 로그에는 전체 프롬프트와 도구 결과가 포함될 수 있습니다.

## 검증과 제한

v1.1.8 소스는 **Python 테스트 464개**와 **Swift 테스트 135개**를 통과했습니다. 앱과 ZIP에서 추출한 앱 모두 서명 검증을 통과했으며, 빌드 디렉터리에 접근할 수 없는 환경에서 영어·간체 중국어·번체 중국어로 정상 실행되는 것을 확인했습니다.

지정된 세 가지 텍스트 모델 외에 현재 개발 소스는 MiMo 텍스트／이미지, 제한적인 WAV 오디오 입력 및 문서 지원과 고정 리비전의 완전한 변환 완료 artifact를 제공합니다. Python 테스트 624개와 Swift 테스트 171개(3개 건너뜀)가 통과했지만, 이번 MiMo 작업은 로컬 패키징을 완료했지만 전체 인수 검증은 아직 미완료입니다. 채팅 첨부 UI에는 유형별 파일 선택, 드래그 앤 드롭, 사용량 표시 및 업로드 상태가 추가되었습니다. 동영상／시청각 동기화와 MCP Agent 화면은 미완성이며, 오디오 및 문서 입력에는 형식과 리소스 제한이 있습니다. [개발 현황](docs/mimo-development.md)을 참고하세요. 새 가속 경로는 소규모 모델과 구성 요소 단위로 테스트했으며, 전체 모델의 속도 및 품질 비교는 아직 완료되지 않았습니다. 매우 긴 프롬프트에는 더 많은 캐시 메모리가 필요합니다.

## 라이선스

Whallm은 [MIT 라이선스](LICENSE)로 배포됩니다. 모델 가중치에는 별도 이용 조건이 적용됩니다. Whallm은 DeepSeek 및 Qwen과 제휴 관계가 없습니다.
