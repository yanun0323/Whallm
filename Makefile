-include Makefile.env
export

.PHONY: $(wildcard *)

MODEL ?= $(HOME)/.dsmodel/deepseek-v4-flash-0731.dsv4
HOST ?= 127.0.0.1
PORT ?= 11434
SPEED_BENCH_DIR ?= scratch/speed-bench
SPARKLE_FRAMEWORK_PATH := $(CURDIR)/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64

ARGS := $(word 2,$(MAKECMDGOALS))

## help: show help
help:
	@echo ""
	@echo "Usage:"
	@echo ""
	@sed -n 's/^## //p' Makefile | column -t -s ':' | sed -e 's/^/\t/'
	@echo ""

## run: start the SwiftUI macOS app
run:
	swift run -Xswiftc -DWHALLM_LOCAL_BUILD dsv4-app $(ARGS)

## build: build the Swift package
build:
	swift build -Xswiftc -DWHALLM_LOCAL_BUILD $(ARGS)

## test: run all Swift tests
test:
	mkdir -p .build/debug/PackageFrameworks
	ln -sfn "$(SPARKLE_FRAMEWORK_PATH)/Sparkle.framework" .build/debug/PackageFrameworks/Sparkle.framework
	swift test -Xswiftc -DWHALLM_LOCAL_BUILD $(ARGS)

# Set before Python starts: MLX caches this flag on first use. Strict FP32
# parity tests must not compare shape-dependent TF32 kernels on M5.
# This is test-only; packaged and standalone runtime defaults are unchanged.
## test-python: run all Python tests with full float32 matrix precision
test-python:
	MLX_ENABLE_TF32=0 PYTHONPATH=runtime:. .venv/bin/python -m unittest discover -s runtime/tests $(ARGS)

PYTHON ?= .venv/bin/python
QWEN_VARIANT ?= baseline
QWEN_FLASH_OUTPUT ?= scratch/qwen-flash/$(QWEN_VARIANT).json
QWEN_FLASH_MAX_TOKENS ?= 32
QWEN_FLASH_ARGS ?=

.PHONY: test-qwen-flash-portable test-qwen-flash benchmark-qwen-flash-host pilot-qwen-flash

## test-qwen-flash-portable: run host-only row I/O, wave and config tests
test-qwen-flash-portable:
	$(PYTHON) -m unittest discover -s runtime/tests/portable -v

## test-qwen-flash: run synthetic MLX tests and Qwen regression tests on Apple Silicon
test-qwen-flash: test-qwen-flash-portable
	MLX_ENABLE_TF32=0 PYTHONPATH=runtime:. $(PYTHON) -m unittest discover -s runtime/tests -p 'test_qwen*.py' -v

## benchmark-qwen-flash-host: measure synthetic row I/O (not an inference benchmark)
benchmark-qwen-flash-host:
	$(PYTHON) Scripts/benchmark_qwen_flash_host.py --output "$(QWEN_FLASH_OUTPUT)"

## pilot-qwen-flash: run a real-model correctness pilot (QWEN_MODEL, PROMPT, QWEN_VARIANT)
pilot-qwen-flash:
	@test -n "$(QWEN_MODEL)" || (echo "QWEN_MODEL must name an installed Qwen model" >&2; exit 2)
	@test -n "$(PROMPT)" || (echo "PROMPT must name a UTF-8 prompt file" >&2; exit 2)
	MLX_ENABLE_TF32=0 PYTHONPATH=runtime:. $(PYTHON) Scripts/research_qwen_optimizations.py \
		--model "$(QWEN_MODEL)" --prompt "$(PROMPT)" --variant "$(QWEN_VARIANT)" \
		--output "$(QWEN_FLASH_OUTPUT)" --max-tokens "$(QWEN_FLASH_MAX_TOKENS)" $(QWEN_FLASH_ARGS)

## package: build a local macOS app with Python, runtime, and local debug features
package:
	WHALLM_BUILD_FLAVOR=local ./Scripts/package-app.sh

## ane-bridge: build the private Apple Neural Engine runtime bridge
ane-bridge:
	./Scripts/build-ane-bridge.sh

## release: publish a signed update (CHANNEL=stable or dev, DEV_BUILD=1)
release:
	./Scripts/release.sh

## server: start the OpenAI-compatible API server directly
server: ane-bridge
	PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.server \
		--model "$(MODEL)" --host "$(HOST)" --port "$(PORT)" $(ARGS)

## benchmark-dsv4: benchmark DeepSeek with SPEED-Bench mixed inputs
benchmark-dsv4:
	.venv/bin/python Scripts/benchmark_api.py --model deepseek-v4-flash-0731 \
		--speed-bench-dir "$(SPEED_BENCH_DIR)" --runs 3 $(ARGS)

## benchmark-qwen: benchmark Qwen with SPEED-Bench mixed inputs
benchmark-qwen:
	.venv/bin/python Scripts/benchmark_api.py --model qwen3.8-flash-next-fp8 \
		--speed-bench-dir "$(SPEED_BENCH_DIR)" --runs 3 $(ARGS)

%:
	@:
