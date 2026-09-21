import AppKit
import DeepSeekRepack
import Foundation
import SwiftUI
import XCTest
@testable import DeepSeekV4SSDApp

final class QwenFlashSettingsTests: XCTestCase {
  private let qwen = ModelKind.qwen3_8FlashNext
  private let preferenceKeys = ["qwenExpertWaveSlots", "qwenNgramIO", "qwenNgramCacheMiB", "qwenSparseSDPA"]
  private let runtimeKeys = ["qwen_expert_wave_slots", "qwen_ngram_io", "qwen_ngram_cache_bytes", "qwen_sparse_sdpa"]

  func testDefaultsAndMissingPreferencesRemainOptIn() throws {
    for kind in [qwen, .deepSeekV4, .deepSeekV41] {
      let original = ModelAdvancedSettings.defaults(for: kind)
      var json = try object(original)
      for key in preferenceKeys { json.removeValue(forKey: key) }
      let old = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONSerialization.data(withJSONObject: json))
      let migrated = old.normalized(for: kind)
      XCTAssertEqual(migrated.qwenExpertWaveSlots, 0)
      XCTAssertEqual(migrated.qwenNgramIO, "mmap")
      XCTAssertEqual(migrated.qwenNgramCacheMiB, 0)
      XCTAssertEqual(migrated.qwenSparseSDPA, false)
      XCTAssertEqual(migrated, original.normalized(for: kind))
    }
  }

  func testSaveLoadAndModelIsolation() throws {
    let name = "QwenFlashSettingsTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: name))
    defer { store.removePersistentDomain(forName: name) }
    var settings = ModelAdvancedSettings.defaults(for: qwen)
    settings.qwenExpertWaveSlots = 32
    settings.qwenNgramIO = "pread"
    settings.qwenNgramCacheMiB = 64
    settings.qwenSparseSDPA = true
    settings.save(for: qwen, defaults: store)
    XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: qwen, defaults: store), settings.normalized(for: qwen))
    for kind in [ModelKind.deepSeekV4, .deepSeekV41] {
      XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: kind, defaults: store), .defaults(for: kind))
      settings.save(for: kind, defaults: store)
      let other = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: store)
      XCTAssertEqual(other.qwenExpertWaveSlots, 0)
      XCTAssertEqual(other.qwenNgramIO, "mmap")
      XCTAssertEqual(other.qwenNgramCacheMiB, 0)
      XCTAssertEqual(other.qwenSparseSDPA, false)
    }
    XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: qwen, defaults: store).qwenNgramCacheMiB, 64)
  }

  func testValidationAndOverflowSafety() throws {
    for value in [-1, 513, Int.max, Int.min] {
      var settings = ModelAdvancedSettings.defaults(for: qwen)
      settings.qwenExpertWaveSlots = value
      XCTAssertThrowsError(try settings.validate(for: qwen))
      settings.qwenExpertWaveSlots = 0
      settings.qwenNgramIO = "pread"
      settings.qwenNgramCacheMiB = value
      XCTAssertThrowsError(try settings.validate(for: qwen))
      XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, 0)
    }
    for value in [0, 1, 32, 512] {
      var settings = ModelAdvancedSettings.defaults(for: qwen)
      settings.qwenExpertWaveSlots = value
      settings.qwenNgramIO = "pread"
      settings.qwenNgramCacheMiB = value
      XCTAssertNoThrow(try settings.validate(for: qwen))
      XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, value * 1_048_576)
    }
    var settings = ModelAdvancedSettings.defaults(for: qwen)
    for backend in ["", "PREAD", "direct"] {
      settings.qwenNgramIO = backend
      XCTAssertThrowsError(try settings.validate(for: qwen))
    }
  }

  func testInactiveCacheRetainsChoiceAndMTPDoesNotResetExperiments() throws {
    var settings = ModelAdvancedSettings.defaults(for: qwen)
    settings.qwenNgramCacheMiB = 127
    XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, 0)
    settings.qwenNgramIO = "pread"
    XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, 133_169_152)
    settings.qwenExpertWaveSlots = 1
    settings.qwenSparseSDPA = true
    for mtp in [true, false] {
      settings.mtpEnabled = mtp
      settings.qwenNgramIO = "mmap"
      let normalized = settings.normalized(for: qwen)
      XCTAssertEqual(normalized.qwenNgramCacheMiB, 127)
      XCTAssertEqual(normalized.effectiveQwenNgramCacheBytes, 0)
      XCTAssertEqual(normalized.qwenExpertWaveSlots, 1)
      XCTAssertEqual(normalized.qwenSparseSDPA, true)
      settings.qwenNgramIO = "pread"
      XCTAssertEqual(settings.effectiveQwenNgramCacheBytes, 133_169_152)
    }
  }

  @MainActor
  func testWavesSuppressOnlyEffectiveConflictsAndRestoreBothChoices() throws {
    for saved in [true, false] {
      var settings = ModelAdvancedSettings.defaults(for: qwen)
      settings.nextLayerPrefetch = saved
      settings.batchedExpertPrefill = saved
      settings.qwenGroupedExperts = saved
      settings.qwenExpertWaveSlots = 32
      for mtp in [true, false] {
        settings.mtpEnabled = mtp
        let original = settings
        let runtime = try catalog(settings).models[0].runtime
        XCTAssertFalse(runtime.qwenNextLayerPrefetch)
        XCTAssertFalse(runtime.batchedExpertPrefill)
        XCTAssertFalse(runtime.qwenGroupedExperts)
        XCTAssertEqual(settings, original)
      }
      settings.qwenExpertWaveSlots = 0
      let restored = try catalog(settings).models[0].runtime
      XCTAssertEqual(restored.qwenNextLayerPrefetch, saved)
      XCTAssertEqual(restored.batchedExpertPrefill, saved)
      XCTAssertEqual(restored.qwenGroupedExperts, saved)
    }
  }

  @MainActor
  func testOldRuntimeCatalogDecodesAndEncodesConcreteDefaults() throws {
    let runtime = try catalog(.defaults(for: qwen)).models[0].runtime
    var json = try object(runtime)
    for key in runtimeKeys { json.removeValue(forKey: key) }
    let legacy = try JSONDecoder().decode(ModelCatalog.Entry.Runtime.self,
      from: JSONSerialization.data(withJSONObject: json))
    let encoded = try object(legacy)
    XCTAssertEqual(encoded["qwen_expert_wave_slots"] as? Int, 0)
    XCTAssertEqual(encoded["qwen_ngram_io"] as? String, "mmap")
    XCTAssertEqual(encoded["qwen_ngram_cache_bytes"] as? Int, 0)
    XCTAssertEqual(encoded["qwen_sparse_sdpa"] as? Bool, false)
  }

  func testRestoreDefaultsClearsExperimentsWithoutChangingOtherModels() throws {
    var flow = ModelSettingsResetConfirmation()
    flow.begin(for: qwen, locked: false)
    flow.advance()
    let reset = try XCTUnwrap(flow.confirm(for: qwen, locked: false))
    XCTAssertEqual(reset.qwenExpertWaveSlots, 0)
    XCTAssertEqual(reset.qwenNgramIO, "mmap")
    XCTAssertEqual(reset.qwenNgramCacheMiB, 0)
    XCTAssertEqual(reset.qwenSparseSDPA, false)
    let id = qwen.apiModelID
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: id, loadingModel: nil, modelActionID: nil))
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: nil, loadingModel: id, modelActionID: nil))
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: nil, loadingModel: nil, modelActionID: id))
    XCTAssertFalse(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: ModelKind.deepSeekV4.apiModelID,
      loadingModel: nil, modelActionID: nil))
  }

  func testAllCopyIsLocalized() {
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      for key in QwenFlashCopy.allKeys {
        let value = L10n.string(key, language: language)
        XCTAssertFalse(value.isEmpty)
        if language != .english { XCTAssertNotEqual(value, key) }
      }
    }
  }

  @MainActor
  func testCatalogFixturesForPythonConsumer() throws {
    for (name, kind, wave, backend, mib, sdpa) in [
      ("qwen-default", qwen, 0, "mmap", 0, false),
      ("qwen-waves", qwen, 32, "mmap", 0, false),
      ("qwen-pread", qwen, 0, "pread", 64, false),
      ("qwen-sdpa", qwen, 0, "mmap", 0, true),
      ("qwen-combined", qwen, 32, "pread", 64, true),
      ("qwen-mmap-restored", qwen, 0, "mmap", 64, false),
      ("deepseek-v4", ModelKind.deepSeekV4, 32, "pread", 64, true),
      ("deepseek-v41", ModelKind.deepSeekV41, 32, "pread", 64, true),
    ] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      settings.qwenExpertWaveSlots = wave
      settings.qwenNgramIO = backend
      settings.qwenNgramCacheMiB = mib
      settings.qwenSparseSDPA = sdpa
      let result = try catalog(settings, kind: kind)
      let json = try object(result.models[0].runtime)
      let isQwen = kind == qwen
      XCTAssertEqual(json["qwen_expert_wave_slots"] as? Int, isQwen ? wave : 0)
      XCTAssertEqual(json["qwen_ngram_io"] as? String, isQwen ? backend : "mmap")
      XCTAssertEqual(json["qwen_ngram_cache_bytes"] as? Int, isQwen && backend == "pread" ? mib * 1_048_576 : 0)
      XCTAssertEqual(json["qwen_sparse_sdpa"] as? Bool, isQwen && sdpa)
      if let base = ProcessInfo.processInfo.environment["WHALLM_QWEN_UI_ARTIFACTS"] {
        let directory = URL(fileURLWithPath: base).appendingPathComponent("catalogs")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try result.encoded().write(to: directory.appendingPathComponent("\(name).json"))
      }
    }
  }

  @MainActor
  func testControlsRenderForAllLanguagesAndLockStates() throws {
    _ = NSApplication.shared
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      for state in ["default", "enabled", "locked"] {
        var settings = ModelAdvancedSettings.defaults(for: qwen)
        if state != "default" {
          settings.qwenExpertWaveSlots = 32
          settings.qwenNgramIO = "pread"
          settings.qwenNgramCacheMiB = 64
          settings.qwenSparseSDPA = true
          settings.qwenQSAQueryChunk = 32
          settings.qwenQSAIndexed = true
        }
        let host = NSHostingView(rootView: QwenFlashSettingsSection(settings: .constant(settings),
          modelKind: qwen, settingsLocked: state == "locked", language: language)
          .padding(24).frame(width: 760)
          .background(Color(nsColor: .windowBackgroundColor)).preferredColorScheme(.dark))
        let size = host.fittingSize
        XCTAssertGreaterThan(size.height, 200)
        XCTAssertLessThan(size.height, 1_400, "Controls must fit a scrollable settings card")
        host.frame = NSRect(origin: .zero, size: size)
        let window = NSWindow(contentRect: NSRect(x: -10_000, y: -10_000, width: size.width, height: size.height),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        defer { window.close() }
        window.orderBack(nil)
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(png.count, 1_000)
        if let base = ProcessInfo.processInfo.environment["WHALLM_QWEN_UI_ARTIFACTS"] {
          let directory = URL(fileURLWithPath: base).appendingPathComponent("screenshots")
          try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
          try png.write(to: directory.appendingPathComponent("qwen-flash-\(language.rawValue)-\(state).png"))
        }
      }
    }
  }

  @MainActor
  func testSectionIsAbsentForNonQwenModels() {
    for kind in [ModelKind.deepSeekV4, .deepSeekV41] {
      let host = NSHostingView(rootView: QwenFlashSettingsSection(settings: .constant(.defaults(for: kind)),
        modelKind: kind, settingsLocked: false, language: .english))
      XCTAssertEqual(host.fittingSize.height, 0)
    }
  }


  @MainActor
  func testQSAThroughputPreferencesCatalogAndDependencies() throws {
    var settings = ModelAdvancedSettings.defaults(for: qwen)
    XCTAssertEqual(settings.qwenQSAQueryChunk, 4)
    XCTAssertEqual(settings.qwenQSAIndexed, false)
    settings.qwenQSAQueryChunk = 32
    settings.qwenQSAIndexed = true
    settings.qwenSparseSDPA = false
    let inactive = try object(catalog(settings).models[0].runtime)
    XCTAssertEqual(inactive["qwen_qsa_query_chunk"] as? Int, 32)
    XCTAssertEqual(inactive["qwen_qsa_indexed"] as? Bool, false)
    XCTAssertEqual(settings.qwenQSAIndexed, true, "Saved choice must survive SDPA off")
    settings.qwenSparseSDPA = true
    let active = try object(catalog(settings).models[0].runtime)
    XCTAssertEqual(active["qwen_qsa_indexed"] as? Bool, true)
    let encoded = try JSONEncoder().encode(settings)
    let restored = try JSONDecoder().decode(ModelAdvancedSettings.self, from: encoded).normalized(for: qwen)
    XCTAssertEqual(restored.qwenQSAQueryChunk, 32)
    XCTAssertEqual(restored.qwenQSAIndexed, true)
    for kind in [ModelKind.deepSeekV4, .deepSeekV41] {
      let other = try object(catalog(settings, kind: kind).models[0].runtime)
      XCTAssertEqual(other["qwen_qsa_query_chunk"] as? Int, 4)
      XCTAssertEqual(other["qwen_qsa_indexed"] as? Bool, false)
    }
    if let base = ProcessInfo.processInfo.environment["WHALLM_QWEN_UI_ARTIFACTS"] {
      let directory = URL(fileURLWithPath: base).appendingPathComponent("catalogs")
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      try catalog(settings).encoded().write(to: directory.appendingPathComponent("qwen-throughput.json"))
      settings.qwenSparseSDPA = false
      try catalog(settings).encoded().write(to: directory.appendingPathComponent("qwen-throughput-inactive.json"))
    }
  }

  func testQSAThroughputLegacyAndRangeValidation() throws {
    var settings = ModelAdvancedSettings.defaults(for: qwen)
    var old = try object(settings)
    old.removeValue(forKey: "qwenQSAQueryChunk")
    old.removeValue(forKey: "qwenQSAIndexed")
    let restored = try JSONDecoder().decode(ModelAdvancedSettings.self,
      from: JSONSerialization.data(withJSONObject: old)).normalized(for: qwen)
    XCTAssertEqual(restored.qwenQSAQueryChunk, 4)
    XCTAssertEqual(restored.qwenQSAIndexed, false)
    for invalid in [Int.min, -1, 0, 129, Int.max] {
      settings.qwenQSAQueryChunk = invalid
      XCTAssertThrowsError(try settings.validateQwenFlashSettings())
    }
    for valid in [1, 4, 16, 32, 128] {
      settings.qwenQSAQueryChunk = valid
      XCTAssertNoThrow(try settings.validateQwenFlashSettings())
    }
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
