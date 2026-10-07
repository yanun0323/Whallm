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
    settings.qwenSparseSDPA = true
    settings.qwenNgramIO = "pread"
    settings.qwenMTPPolicy = true
    settings.qwenMTPDraftTokens = 3
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
    custom.qwenSparseSDPA = true
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
    XCTAssertEqual(added.runtime.qwenNgramIO, "pread")
    XCTAssertEqual(original.runtime.qwenNgramIO, "mmap")
    XCTAssertTrue(added.runtime.qwenQuantizedKV)
    XCTAssertTrue(added.runtime.qwenQuantizedIndex)
    XCTAssertTrue(added.runtime.qwenPooledIndexCache)
    XCTAssertTrue(added.runtime.qwenNgramLookupOptimized)
    XCTAssertTrue(added.runtime.qwenCompileTensorOps)
    XCTAssertTrue(added.runtime.qwenPhaseMemory)
    XCTAssertTrue(added.runtime.qwenNextLayerPrefetch)
    XCTAssertTrue(added.runtime.mtpEnabled)
    XCTAssertEqual(added.runtime.qwenMTPDraftTokens, 3)
    XCTAssertEqual(added.runtime.qwenMTPZeroAcceptanceLimit, 4)
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
  func testSwiftShowsQwenSettingsInAllAppLanguages() {
    for language in [AppLanguage.english, .simplifiedChinese, .traditionalChinese] {
      let host = NSHostingView(rootView: QwenFlashSettingsSection(settings: .constant(.defaults(for: swift)),
        modelKind: swift, settingsLocked: false, language: language).frame(width: 760))
      XCTAssertGreaterThan(host.fittingSize.height, 200)
      XCTAssertLessThan(host.fittingSize.height, 1_400)
    }
  }

  private func installed(_ kind: ModelKind) -> InstalledModelInfo {
    InstalledModelInfo(url: URL(fileURLWithPath: "/fixture/\(kind.descriptor.directoryName)"),
      size: 0, quickIssues: [], hasMTP: true, hasDSpark: false,
      modelKind: kind, modelID: kind.descriptor.checkpointModelID)
  }
}
