#!/usr/bin/env python3
"""Synthetic N-gram row I/O experiment, NOT an inference/physical SSD benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=65_536)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.rows < 512 or args.requests < 1:
        parser.error("rows must be at least 512 and requests must be positive")
    if args.output.exists():
        parser.error("output exists; preserve prior evidence")
    root = Path(__file__).resolve().parents[1]
    path = root / "runtime/deepseek_v4_ssd/qwen_flash_io.py"
    spec = importlib.util.spec_from_file_location("qwen_flash_host_io", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    rng = np.random.default_rng(args.seed)
    rows = rng.integers(0, 256, (args.rows, 160), dtype=np.uint8)
    traces = {
        "reused_rows": rng.integers(0, 256, (args.requests, 128)),
        "uniform_rows": rng.integers(0, args.rows, (args.requests, 128)),
    }
    report = dict(formal_performance_result=False,
        scope="synthetic packed-row lookup only; no MLX, FP8 decoding or model inference",
        created_at=datetime.now(timezone.utc).isoformat(),
        environment=dict(platform=platform.platform(), machine=platform.machine(),
                         python=sys.version, numpy=np.__version__),
        source_sha256={str(path.relative_to(root)): sha(path.read_bytes()),
                       "Scripts/benchmark_qwen_flash_host.py": sha(Path(__file__).read_bytes())},
        workload=dict(rows=args.rows, row_bytes=160, requests=args.requests, rows_per_request=128,
                      seed=args.seed, payload_sha256=sha(rows.tobytes())),
        cache_state="new reader per variant; OS page cache uncontrolled; generated file is warm",
        limits=["Not evidence of Apple Silicon or SSD throughput.",
                "Single timing sample; do not promote defaults from these results.",
                "Byte counters are pread-returned bytes, not physical storage traffic.",
                "Row-cache ceiling excludes Python metadata, returned arrays and OS caches."],
        experiments=[])
    with tempfile.TemporaryDirectory() as directory:
        data_path = Path(directory) / "ngram.bin"
        rows.tofile(data_path)
        for name, queries in traces.items():
            expected = sha(rows[queries].tobytes())
            for variant in ("mmap", "pread", "pread-cache"):
                mapped = None
                reader = None
                if variant == "mmap":
                    mapped = np.memmap(data_path, mode="r", dtype=np.uint8, shape=rows.shape)
                    lookup = lambda ids: np.array(mapped[ids], copy=True)
                else:
                    reader = module.NGramRowReader(data_path, args.rows, 160,
                        cache_bytes=512 * 160 if variant == "pread-cache" else 0)
                    lookup = reader.lookup
                digest = hashlib.sha256()
                started = time.perf_counter()
                try:
                    for query in queries:
                        digest.update(lookup(query).tobytes())
                    elapsed = time.perf_counter() - started
                    actual = digest.hexdigest()
                    if actual != expected:
                        raise AssertionError("row lookup changed output bytes")
                    report["experiments"].append(dict(trace=name, variant=variant, seconds=elapsed,
                        queries_sha256=sha(queries.tobytes()), output_sha256=actual,
                        exact_match=True, counters=reader.snapshot() if reader else None))
                finally:
                    if reader:
                        reader.close()
                    if mapped is not None:
                        mapped._mmap.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
