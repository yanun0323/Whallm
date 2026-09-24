import Foundation
import DeepSeekRepack
import XCTest
@testable import DeepSeekV4SSDApp

final class ModelSettingsResetTests: XCTestCase {
  func testDefaultsAndMissingOptionalValuesForEveryModel() throws {
    for kind in ModelPackages.descriptors.compactMap({ ModelKind(rawValue: $0.kind) }) {
      let defaults = ModelAdvancedSettings.defaults(for: kind)
      XCTAssertEqual(defaults.anePrefillRatio, 0)
      XCTAssertEqual(defaults.layerMajorPrefill,
        kind.descriptor.supports("layerMajorPrefill") && kind != .mimoV26FlashRL)
      XCTAssertEqual(defaults.readyExpertDecode, kind.descriptor.supports("readyExpertDecode"))
      XCTAssertEqual(defaults.batchedExpertPrefill, kind.descriptor.supports("batchedExpertPrefill"))
      XCTAssertEqual(defaults.nextLayerPrefetch, kind.descriptor.supports("nextLayerPrefetch"))
      XCTAssertEqual(defaults.qwenGroupedExperts, kind.descriptor.supports("groupedExperts"))
      let budgets = [defaults.expertCacheGiB, defaults.mtpCacheGiB, defaults.dsparkCacheGiB].compactMap { $0 }
      XCTAssertFalse(budgets.isEmpty)
      for budget in budgets {
        XCTAssertEqual(budget, ExpertMemory.roundedGiB(budget))
        XCTAssertGreaterThan(budget, 0)
      }
      XCTAssertEqual(defaults.packedKVCache, false)
      XCTAssertEqual(defaults.packedIndexCache, false)
      XCTAssertEqual(defaults.approximationEnabled, false)
      var legacy = defaults
      legacy.readyExpertDecode = nil
      legacy.batchedExpertPrefill = nil
      legacy.nextLayerPrefetch = nil
      legacy.qwenGroupedExperts = nil
      legacy.anePrefillRatio = nil
      legacy.packedKVCache = nil
      legacy.packedIndexCache = nil
      legacy.approximationEnabled = nil
      let normalized = legacy.normalized(for: kind)
      XCTAssertEqual(normalized.readyExpertDecode, defaults.readyExpertDecode)
      XCTAssertEqual(normalized.batchedExpertPrefill, defaults.batchedExpertPrefill)
      XCTAssertEqual(normalized.nextLayerPrefetch, defaults.nextLayerPrefetch)
      XCTAssertEqual(normalized.qwenGroupedExperts, defaults.qwenGroupedExperts)
      XCTAssertEqual(normalized.anePrefillRatio, 0)
      XCTAssertEqual(normalized.packedKVCache, false)
      XCTAssertEqual(normalized.packedIndexCache, false)
      XCTAssertEqual(normalized.approximationEnabled, false)
    }
  }

  func testMiMoLayerMajorPrefillIsOptInAndPersists() throws {
    let suite = "MiMoPrefillTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    var settings = ModelAdvancedSettings.defaults(for: .mimoV26FlashRL)
    XCTAssertFalse(settings.layerMajorPrefill)
    XCTAssertTrue(ModelKind.mimoV26FlashRL.descriptor.supports("layerMajorPrefill"))
    settings.layerMajorPrefill = true
    settings.save(for: .mimoV26FlashRL, defaults: store)
    XCTAssertTrue(ModelAdvancedSettings.loadOrDefault(for: .mimoV26FlashRL, defaults: store).layerMajorPrefill)
  }

  func testExplicitAccelerationOptOutsSurviveSaveAndLoad() throws {
    let suite = "AccelerationDefaultsTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    for kind in ModelPackages.descriptors.compactMap({ ModelKind(rawValue: $0.kind) }) {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      settings.layerMajorPrefill = false
      settings.readyExpertDecode = false
      settings.batchedExpertPrefill = false
      settings.nextLayerPrefetch = false
      settings.qwenGroupedExperts = false
      settings.save(for: kind, defaults: store)
      let restored = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: store)
      XCTAssertFalse(restored.layerMajorPrefill)
      XCTAssertEqual(restored.readyExpertDecode, false)
      XCTAssertEqual(restored.batchedExpertPrefill, false)
      XCTAssertEqual(restored.nextLayerPrefetch, false)
      XCTAssertEqual(restored.qwenGroupedExperts, false)
    }
  }

  func testTwoConfirmationsRestoreModelSpecificDefaultsAndPersist() throws {
    let suite = "ModelSettingsResetTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    for kind in ModelPackages.descriptors.compactMap({ ModelKind(rawValue: $0.kind) }) {
      var custom = ModelAdvancedSettings.defaults(for: kind)
      custom.layerMajorPrefill = true
      custom.readyExpertDecode = true
      custom.batchedExpertPrefill = true
      custom.expertCacheGiB = 8.25 // Existing precision is preserved until edited.
      custom.anePrefillRatio = 0.5
      custom.packedKVCache = true
      custom.packedIndexCache = true
      custom.approximationEnabled = true
      custom.defaultMaxTokens = 256
      for feature in QwenOptimization.allCases { custom[keyPath: feature.keyPath] = true }
      custom.qwenMTPDraftTokens = 3
      custom.qwenMTPZeroAcceptanceLimit = 4
      custom.save(for: kind, defaults: store)
      let saved = try XCTUnwrap(ModelAdvancedSettings.load(for: kind, defaults: store))
      XCTAssertEqual(saved.layerMajorPrefill, kind.descriptor.supports("layerMajorPrefill"))
      XCTAssertEqual(saved.readyExpertDecode, kind.descriptor.supports("readyExpertDecode"))
      XCTAssertEqual(saved.batchedExpertPrefill, kind.descriptor.supports("batchedExpertPrefill"))
      XCTAssertEqual(saved.expertCacheGiB, 8.25)
      XCTAssertEqual(saved.anePrefillRatio, 0.5)
      XCTAssertEqual(saved.packedKVCache, true)
      XCTAssertEqual(saved.packedIndexCache, true)
      XCTAssertEqual(saved.approximationEnabled, kind.descriptor.supports("approximation"))
      var flow = ModelSettingsResetConfirmation()
      XCTAssertNil(flow.confirm(for: kind, locked: false))
      flow.begin(for: kind, locked: false)
      XCTAssertNil(flow.confirm(for: kind, locked: false))
      flow.begin(for: kind, locked: false)
      flow.advance()
      XCTAssertEqual(ModelAdvancedSettings.load(for: kind, defaults: store), saved)
      let restored = try XCTUnwrap(flow.confirm(for: kind, locked: false))
      XCTAssertEqual(restored, ModelAdvancedSettings.defaults(for: kind))
      restored.save(for: kind, defaults: store)
      XCTAssertEqual(ModelAdvancedSettings.load(for: kind, defaults: store), restored)
      XCTAssertEqual(flow.stage, .idle)
      XCTAssertNil(flow.confirm(for: kind, locked: false))
    }
  }

  func testCancellationLockAndModelChangePreventReset() {
    for advance in [false, true] {
      var flow = ModelSettingsResetConfirmation()
      flow.begin(for: .deepSeekV4, locked: false)
      if advance { flow.advance() }
      flow.cancel()
      XCTAssertNil(flow.confirm(for: .deepSeekV4, locked: false))
    }
    var flow = ModelSettingsResetConfirmation()
    flow.begin(for: .deepSeekV4, locked: true)
    XCTAssertEqual(flow.stage, .idle)
    flow.begin(for: .deepSeekV4, locked: false)
    flow.advance()
    XCTAssertNil(flow.confirm(for: .deepSeekV4, locked: true))
    flow.begin(for: .deepSeekV4, locked: false)
    flow.advance()
    XCTAssertNil(flow.confirm(for: .deepSeekV41, locked: false))
  }

  func testExpertMemorySliderBoundsAndStep() {
    for gib in [8, 24, 64, 192, 512] {
      let range = ExpertMemory.sliderRange(physicalMemory: UInt64(gib) * 1_073_741_824)
      XCTAssertEqual(range.lowerBound, 0.1)
      XCTAssertEqual(range.upperBound, Double(gib))
      XCTAssertEqual(ExpertMemory.sliderValue(-1, in: range), 0.1)
      XCTAssertEqual(ExpertMemory.sliderValue(Double(gib) + 1, in: range), Double(gib))
      XCTAssertEqual(ExpertMemory.sliderValue(1.24, in: range), 1.2)
      XCTAssertEqual(ExpertMemory.sliderValue(1.26, in: range), 1.3)
      XCTAssertEqual(ExpertMemory.sliderValue(.nan, in: range), 0.1)
    }
    let oddBytes: UInt64 = 8_000_000_000
    let range = ExpertMemory.sliderRange(physicalMemory: oddBytes)
    XCTAssertLessThanOrEqual(range.upperBound * ExpertMemory.gib, Double(oddBytes))
    XCTAssertEqual(range.upperBound, 7.4)
  }

  func testCacheBudgetRoundingAndStatusLabel() {
    XCTAssertEqual(ExpertMemory.defaultGiB(slots: 1152, blobBytes: ExpertMemory.blobBytes(for: .deepSeekV4)), 14.3)
    XCTAssertEqual(ExpertMemory.defaultGiB(slots: 3072, blobBytes: ExpertMemory.blobBytes(for: .qwen3_8FlashNext)), 7.5)
    XCTAssertEqual(ExpertMemory.defaultGiB(slots: 32, blobBytes: ExpertMemory.blobBytes(for: .qwen3_8FlashNext)), 0.1)
    XCTAssertEqual(ExpertMemory.defaultGiB(slots: 768, blobBytes: ExpertMemory.blobBytes(for: .deepSeekV4)), 9.6)
    XCTAssertEqual(ExpertMemory.roundedGiB(8.26), 8.3)
    XCTAssertEqual(ExpertMemory.roundedGiB(8.24), 8.2)
    XCTAssertEqual(L10n.string("Expert cache hit rate", language: .traditionalChinese), "專家快取命中率")
    XCTAssertEqual(L10n.string("Expert cache hit rate", language: .simplifiedChinese), "专家缓存命中率")
    XCTAssertEqual(L10n.string("Expert cache hit rate", language: .english), "Expert cache hit rate")
  }

  func testResetCopyIsLocalized() {
    XCTAssertEqual(L10n.string("Restore defaults", language: .traditionalChinese), "恢復成預設值")
    for language in [AppLanguage.english, .simplifiedChinese, .traditionalChinese] {
      for key in ["Default settings", "Restore defaults", "Restore this model's defaults?",
                  "Confirm restoring defaults", "Continue",
                  "Restore all advanced settings for this model, including generation settings. Alias and model files are kept.",
                  "Your custom advanced settings will be replaced and saved immediately. This cannot be undone."] {
        let value = L10n.string(key, language: language)
        XCTAssertFalse(value.isEmpty)
        if language != .english { XCTAssertNotEqual(value, key) }
      }
    }
  }
}
