import AppKit
import DeepSeekRepack
import Foundation
import SwiftUI
import XCTest
@testable import DeepSeekV4SSDApp

final class SwiftQwenModelTests: XCTestCase {
  private let swift = ModelKind.swift1_5Qwen3_8FlashNext
  private let fp8 = ModelKind.qwen3_8FlashNext

  func testShortDisplayNameKeepsInstallationAndAPIIdentity() {
    XCTAssertEqual(swift.displayName, "Swift1.5-Qwen3.8-Flash-Next")
    XCTAssertEqual(swift.modelKindLabel, "Swift1.5 Qwen3.8 Flash Next")
    XCTAssertEqual(swift.rawValue, "swift1.5-qwen3.8-flash-next")
    XCTAssertEqual(swift.apiModelID, "swift1.5-qwen3.8-flash-next-mxfp4")
    XCTAssertEqual(swift.descriptor.directoryName, "swift1.5-qwen3.8-flash-next-mxfp4.dsv4")
    XCTAssertEqual(SwiftQwenInstalledModelArtifact.repository,
      "Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4")
    XCTAssertEqual(SwiftQwenInstalledModelArtifact.installedBytes, 127_714_522_478)
  }

  func testSettingsPersistAndResetWithoutTouchingFP8() throws {
    let suite = "SwiftQwenModelTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    var original = ModelAdvancedSettings.defaults(for: fp8)
    original.defaultTemperature = 0.15
    original.readWorkers = 7
    original.qwenExpertWaveSlots = 16
    original.save(for: fp8, defaults: store)
    let before = store.data(forKey: "modelAdvancedSettings.\(fp8.rawValue)")

    var settings = ModelAdvancedSettings.loadOrDefault(for: swift, defaults: store)
    XCTAssertEqual(settings, .defaults(for: swift))
    settings.readWorkers = 9
    settings.defaultTemperature = 0.4
    settings.memoryLimitGiB = 24
    settings.promptCacheEntries = 3
    settings.save(for: swift, defaults: store)
    XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: swift, defaults: store), settings)
    XCTAssertEqual(store.data(forKey: "modelAdvancedSettings.\(fp8.rawValue)"), before)
    var reset = ModelSettingsResetConfirmation()
    reset.begin(for: swift, locked: false)
    reset.advance()
    try XCTUnwrap(reset.confirm(for: swift, locked: false)).save(for: swift, defaults: store)
    XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: swift, defaults: store), .defaults(for: swift))
    XCTAssertEqual(store.data(forKey: "modelAdvancedSettings.\(fp8.rawValue)"), before)
  }

  @MainActor
  func testLegacySettingsAndAliasAreNotImportedIntoSwift() throws {
    let suite = "SwiftQwenLegacyTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    let selectedKey = "selectedInstallModelKind"
    store.set(swift.rawValue, forKey: selectedKey)
    var old = try XCTUnwrap(JSONSerialization.jsonObject(
      with: JSONEncoder().encode(ModelAdvancedSettings.defaults(for: fp8))) as? [String: Any])
    old["publicModel"] = "old-fp8-alias"
    old["readWorkers"] = 7
    old["defaultTemperature"] = 0.15
    store.set(try JSONSerialization.data(withJSONObject: old), forKey: ServerConfiguration.preferenceKey)
    XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: swift, defaults: store), .defaults(for: swift))
    XCTAssertEqual(ModelLibrary(defaults: store).alias(for: swift), "")
    XCTAssertNil(store.object(forKey: "modelAdvancedSettingsLegacyModelKind"))
  }

  @MainActor
  func testBothModelsHaveSeparateCatalogEntriesAliasesAndRuntimeOptions() throws {
    let suite = "SwiftQwenCatalogTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: store)
    try library.saveAlias("Original", for: fp8)
    try library.saveAlias("Swift", for: swift)
    XCTAssertThrowsError(try library.saveAlias(fp8.apiModelID, for: swift))
    XCTAssertThrowsError(try library.saveAlias("Original", for: swift))
    var custom = ModelAdvancedSettings.defaults(for: swift)
    custom.readWorkers = 9
    custom.qwenAdaptiveSampling = false
    custom.defaultTemperature = 0.4
    custom.packedKVCache = true
    custom.packedIndexCache = true
    custom.qwenNgramIO = "pread"
    custom.qwenMTPPolicy = true
    custom.qwenMTPDraftTokens = 3
    custom.qwenMTPZeroAcceptanceLimit = 4
    custom.mtpEnabled = true
    let catalog = try ModelLibrary.makeServerCatalog(models: [installed(fp8), installed(swift)],
      aliases: library.aliases, settings: [swift: custom], powerSavingLimitGBps: nil)
    XCTAssertEqual(catalog.models.map(\.id), [fp8.apiModelID, swift.apiModelID])
    let original = catalog.models[0], added = catalog.models[1]
    XCTAssertNotEqual(original.path, added.path)
    XCTAssertEqual(added.modelKind, swift.rawValue)
    XCTAssertEqual(added.alias, "Swift")
    XCTAssertEqual(original.alias, "Original")
    XCTAssertEqual(added.runtime.readWorkers, 9)
    XCTAssertEqual(original.runtime.readWorkers, 16)
    // Swift fixes its hidden tuning (2026-10-09), so these saved choices do not apply.
    XCTAssertEqual(added.runtime.qwenNgramIO, "mmap")
    XCTAssertEqual(original.runtime.qwenNgramIO, "mmap")
    XCTAssertFalse(added.runtime.qwenQuantizedKV)
    XCTAssertFalse(added.runtime.qwenQuantizedIndex)
    XCTAssertTrue(added.runtime.qwenPooledIndexCache)
    XCTAssertTrue(added.runtime.qwenNgramLookupOptimized)
    XCTAssertTrue(added.runtime.qwenCompileTensorOps)
    XCTAssertTrue(added.runtime.qwenPhaseMemory)
    XCTAssertTrue(added.runtime.qwenNextLayerPrefetch)
    XCTAssertTrue(added.runtime.mtpEnabled)
    XCTAssertEqual(added.runtime.qwenMTPDraftTokens, 2)
    XCTAssertEqual(added.runtime.qwenMTPZeroAcceptanceLimit, 32)
    XCTAssertEqual(added.defaults.temperature, 0.4)
    XCTAssertEqual(added.defaults.qwenAdaptiveSampling, false)
    XCTAssertEqual(modelKind(withAPIModelID: swift.apiModelID), swift)
    XCTAssertEqual(ExpertMemory.blobBytes(for: swift), ExpertMemory.blobBytes(for: fp8))
    custom.qwenExpertWaveSlots = 513
    XCTAssertThrowsError(try custom.validate(for: swift))
  }

  func testPublishedManifestDiscoveryAndMemoryPlanningUseSwiftIdentity() throws {
    let sourceRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let data = try Data(contentsOf: sourceRoot.appending(path: "runtime/tests/fixtures/swift_qwen_manifest.json"))
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try data.write(to: directory.appending(path: "manifest.json"))
    // No full weights: discovery must still identify the model and report damage.
    let discovered = try XCTUnwrap(InstalledModelDiscovery.inspect(directory))
    XCTAssertEqual(discovered.modelKind, swift)
    XCTAssertEqual(discovered.size, 127_714_522_478)
    XCTAssertTrue(discovered.hasMTP)
    XCTAssertFalse(discovered.isUsable)
    XCTAssertEqual(discovered.quickIssues.count, 60)
    XCTAssertTrue(discovered.quickIssues.contains { $0.path == "vision/common.bin" })
    let manifest = try InstalledModel.loadManifest(at: directory)
    let profile = MemoryPlanningProfile(kind: swift, manifest: manifest, config: [:])
    XCTAssertNotNil(profile.estimate(.defaults(for: swift), mtpAvailable: true, dsparkAvailable: false))
  }

  @MainActor
  func testQwenModelsFixHiddenTuningAndKeepTheVisibleSettings() throws {
    let swift = ModelKind.swift1_5Qwen3_8FlashNext
    // Arbitrary saved values, including ones for controls Swift no longer shows.
    var saved = ModelAdvancedSettings.defaults(for: swift)
    saved.readWorkers = 9
    saved.prefetchReadWorkers = 3
    saved.memoryLimitGiB = 24
    saved.expertCacheGiB = 6
    saved.mtpEnabled = false
    saved.mtpCacheGiB = 0.5
    saved.promptCacheMode = .disk
    saved.promptCacheEntries = 3
    saved.promptCacheMemoryGiB = 5
    let warmup = FileManager.default.temporaryDirectory.appending(path: "warm-\(UUID().uuidString).txt")
    try Data("Hello".utf8).write(to: warmup)
    defer { try? FileManager.default.removeItem(at: warmup) }
    saved.warmupPromptPath = warmup.path
    saved.defaultMaxTokens = 2_048
    saved.qwenAdaptiveSampling = false
    saved.defaultTemperature = 0.3
    saved.recentExpertCache = false
    saved.prefillStepSize = 256
    saved.layerMajorPrefill = false
    saved.packedKVCache = true
    saved.packedIndexCache = true
    saved.qwenSortedExpertPrefill = false
    saved.qwenQSAMaskedPrefill = false
    saved.qwenQSAQueryChunk = 4
    saved.qwenExpertWaveSlots = 16
    saved.qwenNgramIO = "pread"
    saved.qwenPackedGDNPrefill = true
    saved.qwenMTPPolicy = false
    saved.qwenPhaseMemory = false
    let data = try JSONEncoder().encode(saved)
    let loaded = try JSONDecoder().decode(ModelAdvancedSettings.self, from: data).normalized(for: swift)

    // Visible settings keep the user's choices.
    XCTAssertEqual(loaded.readWorkers, 9)
    XCTAssertEqual(loaded.prefetchReadWorkers, 3)
    XCTAssertEqual(loaded.memoryLimitGiB, 24)
    XCTAssertEqual(loaded.expertCacheGiB, 6)
    XCTAssertEqual(loaded.mtpEnabled, false)
    XCTAssertEqual(loaded.mtpCacheGiB, 0.5)
    XCTAssertEqual(loaded.promptCacheMode, .disk)
    XCTAssertEqual(loaded.promptCacheEntries, 3)
    XCTAssertEqual(loaded.promptCacheMemoryGiB, 5)
    XCTAssertEqual(loaded.warmupPromptPath, warmup.path)
    XCTAssertEqual(loaded.defaultMaxTokens, 2_048)
    XCTAssertEqual(loaded.qwenAdaptiveSampling, false)
    XCTAssertEqual(loaded.defaultTemperature, 0.3)
    XCTAssertEqual(loaded.recentExpertCache, false)

    // Hidden controls use the fixed values whatever was saved.
    var fixed = loaded
    fixed.applyQwenFixedTuning()
    XCTAssertEqual(loaded, fixed)
    XCTAssertEqual(loaded.prefillStepSize, 1_024)
    XCTAssertTrue(loaded.layerMajorPrefill)
    XCTAssertEqual(loaded.packedKVCache, false)
    XCTAssertEqual(loaded.packedIndexCache, false)
    XCTAssertEqual(loaded.qwenSortedExpertPrefill, true)
    XCTAssertEqual(loaded.qwenQSAMaskedPrefill, true)
    XCTAssertEqual(loaded.qwenQSAQueryChunk, 16)
    XCTAssertEqual(loaded.qwenExpertWaveSlots, 0)
    XCTAssertEqual(loaded.qwenNgramIO, "mmap")
    XCTAssertEqual(loaded.qwenPackedGDNPrefill, false)
    XCTAssertEqual(loaded.qwenMTPPolicy, true)
    XCTAssertEqual(loaded.effectiveQwenMTPDraftTokens, 2)
    XCTAssertEqual(loaded.effectiveQwenMTPZeroAcceptanceLimit, 32)
    XCTAssertEqual(loaded.qwenPhaseMemory, true)
    XCTAssertNoThrow(try loaded.validate(for: swift))

    let defaults = ModelAdvancedSettings.defaults(for: swift)
    XCTAssertEqual(defaults.mtpEnabled, true)
    XCTAssertEqual(defaults.normalized(for: swift), defaults)
    let model = InstalledModelInfo(url: URL(fileURLWithPath: "/fixture/swift"), size: 0,
      quickIssues: [], hasMTP: false, hasDSpark: false, modelKind: swift, modelID: swift.descriptor.checkpointModelID)
    let runtime = try XCTUnwrap(ModelLibrary.makeServerCatalog(models: [model], aliases: [:],
      settings: [swift: saved], powerSavingLimitGBps: nil).models.first).runtime
    XCTAssertFalse(runtime.mtpEnabled, "no MTP files")
    XCTAssertEqual(runtime.prefillStepSize, 1_024)
    XCTAssertEqual(runtime.qwenQSAQueryChunk, 16)
    XCTAssertEqual(runtime.qwenSortedExpertPrefill, true)
    XCTAssertEqual(runtime.readWorkers, 9)

    // Qwen FP8 shows the same short list and uses the same fixed values.
    let fp8 = ModelKind.qwen3_8FlashNext
    var fp8Saved = saved
    fp8Saved.qwenPackedGDNPrefill = false
    let fp8Loaded = try JSONDecoder().decode(ModelAdvancedSettings.self,
      from: JSONEncoder().encode(fp8Saved)).normalized(for: fp8)
    XCTAssertEqual(fp8Loaded, loaded)
    XCTAssertEqual(ModelAdvancedSettings.defaults(for: fp8).mtpEnabled, true)
    var fp8Defaults = ModelAdvancedSettings.defaults(for: fp8)
    fp8Defaults.applyQwenFixedTuning()
    XCTAssertEqual(fp8Defaults, ModelAdvancedSettings.defaults(for: fp8))
  }

  @MainActor
  func testLowerMLXCommandBufferLimitsApplyOnlyToQwenOnlyCatalogs() throws {
    @MainActor func defaults(_ kinds: [ModelKind]) throws -> [String: String] {
      try ModelLibrary.makeServerCatalog(models: kinds.map(installed), aliases: [:], settings: [:],
        powerSavingLimitGBps: nil).runtimeEnvironmentDefaults
    }
    let lowered = ["MLX_MAX_OPS_PER_BUFFER": "10", "MLX_MAX_MB_PER_BUFFER": "10"]
    XCTAssertEqual(try defaults([.swift1_5Qwen3_8FlashNext]), lowered)
    XCTAssertEqual(try defaults([.qwen3_8FlashNext]), lowered)
    XCTAssertEqual(try defaults([.qwen3_8FlashNext, .swift1_5Qwen3_8FlashNext]), lowered)
    // One process serves the whole catalog, so any other model keeps MLX's own limits.
    XCTAssertEqual(try defaults([.swift1_5Qwen3_8FlashNext, .deepSeekV4]), [:])
    XCTAssertEqual(try defaults([.deepSeekV41]), [:])
    XCTAssertEqual(try defaults([.mimoV26FlashRL, .qwen3_8FlashNext]), [:])
    XCTAssertEqual(try defaults([]), [:])
  }

  private func installed(_ kind: ModelKind) -> InstalledModelInfo {
    InstalledModelInfo(url: URL(fileURLWithPath: "/fixture/\(kind.descriptor.directoryName)"),
      size: 0, quickIssues: [], hasMTP: true, hasDSpark: false,
      modelKind: kind, modelID: kind.descriptor.checkpointModelID)
  }
}
