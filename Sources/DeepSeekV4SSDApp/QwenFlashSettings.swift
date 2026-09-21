import Foundation

/// App choices are independent of the effective runtime configuration.
/// Inactive values survive switching backends or temporarily selecting waves.
extension ModelAdvancedSettings {
  var qwenFlashWavesEnabled: Bool { (qwenExpertWaveSlots ?? 0) > 0 }
  var qwenWholeLayerExperimentsActive: Bool {
    layerMajorPrefill && batchedExpertPrefill == true && !qwenFlashWavesEnabled
  }

  var effectiveQwenNgramCacheBytes: Int {
    let mib = qwenNgramCacheMiB ?? 0
    guard qwenNgramIO == "pread", (0...512).contains(mib) else { return 0 }
    return mib * 1_048_576
  }

  func validateQwenFlashSettings() throws {
    guard (1...32).contains(qwenPrefillReadExperts ?? 1) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidReadBatch))
    }
    guard (0...128).contains(qwenPrefillSeedExperts ?? 0) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidSeeds))
    }
    guard (1...128).contains(qwenQSAQueryChunk ?? 4) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidQueryChunk))
    }
    guard (0...512).contains(qwenExpertWaveSlots ?? 0) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidWaves))
    }
    guard ["mmap", "pread"].contains(qwenNgramIO ?? "mmap") else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidBackend))
    }
    guard (0...512).contains(qwenNgramCacheMiB ?? 0) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidCache))
    }
  }
}

/// English localization keys are shared by the controls and validation tests.
enum QwenFlashCopy {
  static let title = "Qwen Flash experiments"
  static let warning = "Experimental; full-model speed and quality are not yet verified. New QSA controls retain the original chunk size and disable indexed attention by default."
  static let lifecycle = "Saved automatically for this model. Changes apply on the next model load. Unload the model before editing."
  static let locked = "Unload this model to edit these settings. Loading and unloading also lock the controls."
  static let waves = "Experts per wave"
  static let wavesHint = "0 disables waves; 1–512 limits the experts acquired per wave, capped by the expert cache. This is not a total memory limit. Small waves may be slower."
  static let wavesConflict = "Expert waves temporarily disable whole-layer batching, next-layer prefetch, and grouped prefill. Their saved choices are restored when waves are disabled."
  static let backend = "N-gram read backend"
  static let backendHint = "mmap is the default. Buffered pread reads selected rows explicitly; it is not direct SSD-to-GPU I/O and may be slower."
  static let mmap = "mmap (default)"
  static let pread = "pread (experimental)"
  static let cache = "N-gram row cache MiB"
  static let cacheHint = "0 disables retention; 1–512 MiB retains packed FP8 rows with pread. Metadata, temporary buffers and the OS page cache are extra. The saved size is inactive with mmap."
  static let sdpa = "Use fused QSA SDPA"
  static let sdpaHint = "Uses the same selected attention cells with a fused kernel. Floating-point rounding and generated text may differ; a speed improvement is not guaranteed."
  static let invalidWaves = "Experts per wave must be an integer from 0 to 512."
  static let invalidBackend = "Choose mmap or pread for the N-gram read backend."
  static let invalidCache = "N-gram row cache must be an integer from 0 to 512 MiB."
  static let queryChunk = "QSA queries per chunk"
  static let queryChunkHint = "1–128 queries per attention chunk; default 4. Try 16 or 32 for prefill. Estimated per-chunk workspace is capped at 256 MiB; this is not a process memory limit. Numerical rounding may differ."
  static let indexed = "Use indexed QSA decode"
  static let indexedHint = "Requires fused QSA SDPA. Reads selected KV rows directly for up to 8 queries above the sparse threshold. Prefill and unsupported layouts keep the gathered path. FP32 reduction can change generated text."
  static let invalidQueryChunk = "QSA queries per chunk must be an integer from 1 to 128."
  static let readBatch = "Experts per prefill read"
  static let readBatchHint = "1 keeps existing reads; 2–32 merges adjacent expert payloads without reading unselected gaps. Requires layer-major whole-layer prefill with waves off. Larger batches retain more per-worker staging memory."
  static let seeds = "Warm decode experts per layer"
  static let seedsHint = "0 keeps the cold handoff; 1–128 copies hot experts from the last 128 prompt tokens into free cache slots. Requires whole-layer prefill. Adds prefill time and memory, but may reduce early decode reads. Actual retention is capped by cache capacity."
  static let sharedOverlap = "Overlap shared expert compute with reads"
  static let sharedOverlapHint = "Submit the resident shared expert and router before waiting for routed experts on single-token calls. No experts are skipped. Submission overhead may outweigh the overlap; compare the same throughput workload."
  static let wholeLayerDependency = "Read batching and warm handoff are inactive without layer-major whole-layer prefill, or while expert waves are enabled. Saved choices are retained."
  static let invalidReadBatch = "Experts per prefill read must be an integer from 1 to 32."
  static let invalidSeeds = "Warm decode experts per layer must be an integer from 0 to 128."
  static let allKeys = [readBatch, readBatchHint, seeds, seedsHint, sharedOverlap,
    sharedOverlapHint, wholeLayerDependency, invalidReadBatch, invalidSeeds,queryChunk, queryChunkHint, indexed, indexedHint, invalidQueryChunk,title, warning, lifecycle, locked, waves, wavesHint,
    wavesConflict, backend, backendHint, mmap, pread, cache, cacheHint,
    sdpa, sdpaHint, invalidWaves, invalidBackend, invalidCache]
}
