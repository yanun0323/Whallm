import DeepSeekRepack
import Foundation
import XCTest
@testable import DeepSeekV4SSDApp

/// Both Qwen models hide their tuning controls (2026-10-09). These tests cover the
/// fixed values, old preference and catalog decoding, and range validation.
final class QwenFlashSettingsTests: XCTestCase {
  private let qwenKinds = [ModelKind.qwen3_8FlashNext, .swift1_5Qwen3_8FlashNext]

  func testHiddenQwenTuningIsFixedAndOtherModelsStayNeutral() throws {
    for kind in qwenKinds {
      var saved = ModelAdvancedSettings.defaults(for: kind)
      saved.qwenExpertWaveSlots = 32
      saved.qwenNgramIO = "pread"
      saved.qwenNgramCacheMiB = 64
      saved.qwenQSAQueryChunk = 32
      saved.qwenQSASkipCompleteGather = false
      saved.qwenPrefillReadExperts = 4
      saved.qwenPrefillSeedExperts = 32
      let loaded = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONEncoder().encode(saved)).normalized(for: kind)
      XCTAssertEqual(loaded.qwenExpertWaveSlots, 0)
      XCTAssertEqual(loaded.qwenNgramIO, "mmap")
      XCTAssertEqual(loaded.qwenNgramCacheMiB, 0)
      XCTAssertEqual(loaded.qwenQSAQueryChunk, 16)
      XCTAssertEqual(loaded.qwenQSASkipCompleteGather, true)
      XCTAssertEqual(loaded.qwenPrefillReadExperts, 1)
      XCTAssertEqual(loaded.qwenPrefillSeedExperts, 0)
      XCTAssertEqual(loaded.qwenSortedExpertPrefill, true)
      XCTAssertEqual(loaded.qwenQSAMaskedPrefill, true)
      XCTAssertEqual(loaded.qwenPackedGDNPrefill, false)
      XCTAssertNoThrow(try loaded.validate(for: kind))
      var reset = ModelSettingsResetConfirmation()
      reset.begin(for: kind, locked: false); reset.advance()
      XCTAssertEqual(reset.confirm(for: kind, locked: false), .defaults(for: kind))
    }
    var qwenOnly = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    qwenOnly.qwenExpertWaveSlots = 32
    qwenOnly.qwenNgramIO = "pread"
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .mimoV26FlashRL] {
      let other = qwenOnly.normalized(for: kind)
      XCTAssertEqual(other.qwenExpertWaveSlots, 0)
      XCTAssertEqual(other.qwenNgramIO, "mmap")
      XCTAssertEqual(other.qwenSortedExpertPrefill, false)
      XCTAssertEqual(other.qwenQSAMaskedPrefill, false)
      XCTAssertEqual(ModelAdvancedSettings.defaults(for: kind).qwenQSAQueryChunk, 16)
    }
  }

  func testValidationRangesAndInactiveCacheBytes() throws {
    // Validation still guards values that reach it directly, outside normalization.
    var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    for value in [-1, 513, Int.max, Int.min] {
      settings.qwenExpertWaveSlots = value
      XCTAssertThrowsError(try settings.validateQwenFlashSettings())
      settings.qwenExpertWaveSlots = 0
      settings.qwenNgramIO = "pread"
      settings.qwenNgramCacheMiB = value
      XCTAssertThrowsError(try settings.validateQwenFlashSettings())
      XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, 0)
      settings.qwenNgramCacheMiB = 0
    }
    settings.qwenNgramCacheMiB = 127
    XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, 133_169_152)
    settings.qwenNgramIO = "mmap"
    XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, 0)
    for backend in ["", "PREAD", "direct"] {
      settings.qwenNgramIO = backend
      XCTAssertThrowsError(try settings.validateQwenFlashSettings())
    }
    settings.qwenNgramIO = "mmap"
    for (keyPath, invalid) in [(\ModelAdvancedSettings.qwenQSAQueryChunk, [0, 129]),
                               (\.qwenPrefillReadExperts, [0, 33]), (\.qwenPrefillSeedExperts, [-1, 129])] {
      let original = settings[keyPath: keyPath]
      for value in invalid {
        settings[keyPath: keyPath] = value
        XCTAssertThrowsError(try settings.validateQwenFlashSettings())
      }
      settings[keyPath: keyPath] = original
    }
    XCTAssertNoThrow(try settings.validateQwenFlashSettings())
  }

  @MainActor
  func testOldRuntimeCatalogDecodesAndEncodesConcreteDefaults() throws {
    let runtime = try catalog(.defaults(for: .qwen3_8FlashNext)).models[0].runtime
    var json = try object(runtime)
    for key in ["qwen_expert_wave_slots", "qwen_ngram_io", "qwen_ngram_cache_bytes",
                "qwen_packed_gdn_prefill", "qwen_sorted_expert_prefill", "qwen_qsa_masked_prefill"] {
      json.removeValue(forKey: key)
    }
    let legacy = try JSONDecoder().decode(ModelCatalog.Entry.Runtime.self,
      from: JSONSerialization.data(withJSONObject: json))
    let encoded = try object(legacy)
    XCTAssertEqual(encoded["qwen_expert_wave_slots"] as? Int, 0)
    XCTAssertEqual(encoded["qwen_ngram_io"] as? String, "mmap")
    XCTAssertEqual(encoded["qwen_ngram_cache_bytes"] as? Int, 0)
    XCTAssertEqual(encoded["qwen_packed_gdn_prefill"] as? Bool, false)
    XCTAssertEqual(encoded["qwen_sorted_expert_prefill"] as? Bool, false)
    XCTAssertEqual(encoded["qwen_qsa_masked_prefill"] as? Bool, false)
  }

  func testRemainingCopyIsLocalizedAndSettingsLockWhileLoaded() {
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      for key in QwenFlashCopy.allKeys {
        let value = L10n.string(key, language: language)
        XCTAssertFalse(value.isEmpty)
        if language != .english { XCTAssertNotEqual(value, key) }
      }
    }
    let id = ModelKind.qwen3_8FlashNext.apiModelID
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: id, loadingModel: nil, modelActionID: nil))
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: nil, loadingModel: id, modelActionID: nil))
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: nil, loadingModel: nil, modelActionID: id))
    XCTAssertFalse(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: ModelKind.deepSeekV4.apiModelID,
      loadingModel: nil, modelActionID: nil))
  }

  /// Writes the catalogs `Scripts/validate_qwen_flash_app_catalog.py` reads.
  @MainActor
  func testCatalogFixturesForPythonConsumer() throws {
    for (name, kind) in [("qwen-fixed", ModelKind.qwen3_8FlashNext), ("swift-fixed", .swift1_5Qwen3_8FlashNext),
                         ("deepseek-v4", .deepSeekV4), ("deepseek-v41", .deepSeekV41)] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      settings.qwenExpertWaveSlots = 32
      settings.qwenNgramIO = "pread"
      settings.qwenNgramCacheMiB = 64
      settings.qwenQSAQueryChunk = 32
      let result = try catalog(settings, kind: kind)
      let json = try object(result.models[0].runtime)
      let qwen = kind.usesQwenEngine
      XCTAssertEqual(json["qwen_expert_wave_slots"] as? Int, 0)
      XCTAssertEqual(json["qwen_ngram_io"] as? String, "mmap")
      XCTAssertEqual(json["qwen_ngram_cache_bytes"] as? Int, 0)
      XCTAssertEqual(json["qwen_qsa_query_chunk"] as? Int, 16)
      XCTAssertEqual(json["qwen_sorted_expert_prefill"] as? Bool, qwen)
      XCTAssertEqual(json["qwen_qsa_masked_prefill"] as? Bool, qwen)
      XCTAssertEqual(json["qwen_packed_gdn_prefill"] as? Bool, false)
      XCTAssertEqual(json["mtp_enabled"] as? Bool, qwen)
      if let base = ProcessInfo.processInfo.environment["WHALLM_QWEN_UI_ARTIFACTS"] {
        let directory = URL(fileURLWithPath: base).appendingPathComponent("catalogs")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try result.encoded().write(to: directory.appendingPathComponent("\(name).json"))
      }
    }
  }

  @MainActor
  func testRemovedQwenExperimentsAreIgnoredInSavedSettingsAndOmittedFromCatalogs() throws {
    for kind in qwenKinds {
      var saved = try object(ModelAdvancedSettings.defaults(for: kind))
      // Preferences written before 2026-10-08 can still hold the removed switches.
      for key in ["qwenSparseSDPA", "qwenQSAIndexed", "qwenSharedExpertOverlap", "approximationEnabled"] {
        saved[key] = true
      }
      let settings = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONSerialization.data(withJSONObject: saved)).normalized(for: kind)
      XCTAssertEqual(settings.approximationEnabled, false)
      XCTAssertFalse(kind.descriptor.supports("approximation"))
      XCTAssertNoThrow(try settings.validate(for: kind))
      let entry = try catalog(settings, kind: kind).models[0]
      let runtime = try object(entry.runtime)
      for key in ["qwen_sparse_sdpa", "qwen_qsa_indexed", "qwen_shared_expert_overlap"] {
        XCTAssertNil(runtime[key], key)
      }
      XCTAssertEqual(entry.defaults.approximationMode, "exact")
    }
    XCTAssertTrue(ModelKind.deepSeekV4.descriptor.supports("approximation"))
    XCTAssertTrue(ModelKind.deepSeekV41.descriptor.supports("approximation"))
  }

  private func object<T: Encodable>(_ value: T) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
  }

  @MainActor
  private func catalog(_ settings: ModelAdvancedSettings, kind: ModelKind = .qwen3_8FlashNext) throws -> ModelCatalog {
    let model = InstalledModelInfo(url: URL(fileURLWithPath: "/tmp/qwen-flash-ui-\(kind.rawValue)"),
      size: 1, quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: kind,
      modelID: kind.descriptor.checkpointModelID)
    return try ModelLibrary.makeServerCatalog(models: [model], aliases: [:], settings: [kind: settings],
      powerSavingLimitGBps: nil)
  }
}
