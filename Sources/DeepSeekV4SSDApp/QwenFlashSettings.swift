import Foundation

/// App choices are independent of the effective runtime configuration.
/// Inactive values survive switching backends or temporarily selecting waves.
extension ModelAdvancedSettings {
  var qwenFlashWavesEnabled: Bool { (qwenExpertWaveSlots ?? 0) > 0 }

  var effectiveQwenNgramCacheBytes: Int {
    let mib = qwenNgramCacheMiB ?? 0
    guard qwenNgramIO == "pread", (0...512).contains(mib) else { return 0 }
    return mib * 1_048_576
  }

  func validateQwenFlashSettings() throws {
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
  static let warning = "Experimental; full-model speed and quality are not yet verified. All four controls default to off or mmap."
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
  static let allKeys = [title, warning, lifecycle, locked, waves, wavesHint,
    wavesConflict, backend, backendHint, mmap, pread, cache, cacheHint,
    sdpa, sdpaHint, invalidWaves, invalidBackend, invalidCache]
}
