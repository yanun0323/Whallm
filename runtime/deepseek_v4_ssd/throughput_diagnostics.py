"""Benchmark evidence: effective settings, logical I/O and explicit wait counters.

No timers force GPU evaluation here. Boundary snapshots are taken at the first
YIELDED token, so asynchronous next-token work can straddle that boundary.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=1)
def source_files_hash() -> str:
    root = Path(__file__).parent
    result = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        result.update(path.relative_to(root).as_posix().encode() + b"\0")
        result.update(path.read_bytes())
    return result.hexdigest()


def config_evidence(config):
    # RuntimeConfig contains paths; export scalar knobs but no local paths,
    # prompt content, trace filenames, or future arbitrary string credentials.
    values = asdict(config) if is_dataclass(config) else vars(config)
    allowed_strings = {"expert_eviction_policy", "expert_file_cache_policy", "qwen_ngram_io"}
    clean = {key: value for key, value in values.items()
             if type(value) in (bool, int, float) or key in allowed_strings}
    text = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text, hashlib.sha256(text.encode()).hexdigest()


def cache_counters(cache):
    if cache is None or not callable(getattr(cache, "metrics_snapshot", None)):
        return None
    snapshot = cache.metrics_snapshot()
    if not is_dataclass(snapshot):
        return None
    values = {key: value for key, value in asdict(snapshot).items()
              if type(value) in (int, float)}
    io = cache.prefill_io_snapshot() if callable(getattr(cache, "prefill_io_snapshot", None)) else None
    if io is not None:
        for name in ("read_calls", "direct_bytes", "copied_bytes"):
            values["prefill_io_" + name] = io.get(name, 0)
    return values


def difference(after, before):
    if after is None or before is None:
        return None
    return {key: value - before.get(key, 0) for key, value in after.items()}


def make_report(config, before, first, after, metrics, initial_residents, final_residents):
    text, digest = config_evidence(config)
    return {
        "schema_version": 1,
        "runtime_config_json": text,
        "config_sha256": digest,
        "source_files_sha256": source_files_hash(),
        "through_first_token": difference(first, before),
        "after_first_token": difference(after, first),
        "request_total": difference(after, before),
        "initial_resident_experts": initial_residents,
        "final_resident_experts": final_residents,
        "process_disk_bytes_read": metrics.get("request_process_disk_bytes_read"),
        "decode_cache_eval_seconds": metrics.get("decode_cache_eval_seconds"),
        "notes": "Main-model expert cache only; auxiliary draft caches and N-gram I/O are excluded. First yielded token boundary; pipelined work may cross it. bytes_read is logical expert payload, not physical SSD traffic. read_seconds may overlap compute; wait_seconds counts only consumer blocking in expert futures. Source hash identifies files observed by this server, not a signed build. Prompt cache is disabled; expert/OS caches are not flushed.",
    }
