import DeepSeekRepack
import Foundation
import XCTest
@testable import DeepSeekV4SSDApp

/// Existing Qwen identities now use the native engine without duplicate entries.
final class NativeQwenModelTests: XCTestCase {
  private let swift = ModelKind.swift1_5Qwen3_8FlashNext

  /// A Swift1.5 installation with every file at its manifest size (sparse, no data).
  private func makeSwiftInstallation(in root: URL, complete: Bool = true) throws -> URL {
    let sourceRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let data = try Data(contentsOf: sourceRoot.appending(path: "runtime/tests/fixtures/swift_qwen_manifest.json"))
    let model = root.appending(path: swift.descriptor.directoryName)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try data.write(to: model.appending(path: "manifest.json"))
    let manifest = try InstalledModel.loadManifest(at: model)
    for (index, file) in InstalledModel.verificationFiles(for: manifest).enumerated() {
      if !complete && index == 0 { continue }
      let url = model.appending(path: file.path)
      try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      FileManager.default.createFile(atPath: url.path, contents: nil)
      let handle = try FileHandle(forWritingTo: url)
      try handle.truncate(atOffset: file.size)
      try handle.close()
    }
    return model
  }

  func testIdentityAndUnchangedInstallation() {
    XCTAssertEqual(swift.apiModelID, "swift1.5-qwen3.8-flash-next-mxfp4")
    XCTAssertEqual(swift.displayName, "Swift1.5-Qwen3.8-Flash-Next")
    XCTAssertEqual(swift.descriptor.directoryName, "swift1.5-qwen3.8-flash-next-mxfp4.dsv4")
    XCTAssertEqual(ExpertMemory.blobBytes(for: swift), 2_611_200)
    XCTAssertEqual(ExpertMemory.totalExperts(for: .expert, kind: swift, manifest: nil), 24_576)
  }

  func testDiscoveryListsEachInstallationOnlyOnce() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try makeSwiftInstallation(in: root)
    let result = InstalledModelDiscovery.find(in: root)
    XCTAssertEqual(Set(result.models.map(\.modelKind)), [swift])
    XCTAssertTrue(result.models.allSatisfy { $0.isUsable && $0.url == model.standardizedFileURL.resolvingSymlinksInPath() })
    XCTAssertEqual(Set(result.models.map(\.id)).count, 1)

    // A damaged Swift installation is listed for repair as Swift only.
    let damagedRoot = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: damagedRoot) }
    _ = try makeSwiftInstallation(in: damagedRoot, complete: false)
    let damaged = InstalledModelDiscovery.find(in: damagedRoot)
    XCTAssertEqual(damaged.models.map(\.modelKind), [swift])
    XCTAssertFalse(damaged.models[0].isUsable)
  }

  func testDefaultsUseAutomaticCacheMTPAndEightBitWeights() throws {
    let defaults = ModelAdvancedSettings.defaults(for: swift)
    XCTAssertEqual(defaults.expertCacheAutomatic, true)
    XCTAssertEqual(defaults.mtpEnabled, true)
    XCTAssertEqual(defaults.qwen8BitCommonTensors, true)
    XCTAssertEqual(defaults.readWorkers, 8)
    XCTAssertEqual(defaults.promptCacheMode, .memory)
    XCTAssertNoThrow(try defaults.validate(for: swift))
    // A saved fixed size stays; settings of other kinds never carry the flag.
    var fixed = defaults
    fixed.expertCacheAutomatic = false
    XCTAssertEqual(fixed.normalized(for: swift).expertCacheAutomatic, false)
    var missing = defaults
    missing.expertCacheAutomatic = nil
    XCTAssertEqual(missing.normalized(for: swift).expertCacheAutomatic, true)
    XCTAssertNil(ModelAdvancedSettings.defaults(for: .deepSeekV4).normalized(for: .deepSeekV4).expertCacheAutomatic)
  }

  @MainActor
  func testCatalogSendsNativeOptionsAndMLXDefaults() throws {
    let installed = InstalledModelInfo(url: URL(fileURLWithPath: "/fixture/\(swift.descriptor.directoryName)"),
      size: 0, quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: swift,
      modelID: swift.descriptor.checkpointModelID)
    var settings = ModelAdvancedSettings.defaults(for: swift)
    let catalog = try ModelLibrary.makeServerCatalog(models: [installed],
      aliases: [:], settings: [swift: settings], powerSavingLimitGBps: nil)
    XCTAssertEqual(catalog.models.map(\.id), [swift.apiModelID])
    let entry = catalog.models[0]
    XCTAssertEqual(entry.modelKind, swift.rawValue)
    XCTAssertEqual(entry.path, installed.url.path)
    XCTAssertTrue(entry.runtime.nativeAutomaticExpertCache)
    XCTAssertTrue(entry.runtime.mtpEnabled)
    XCTAssertEqual(entry.runtime.qwen8BitCommonTensors, true)
    // MLX-only options keep the runtime defaults the native support requires.
    XCTAssertFalse(entry.runtime.qwenSortedExpertPrefill ?? false)
    XCTAssertFalse(entry.runtime.qwenQSAMaskedPrefill ?? false)
    XCTAssertFalse(entry.runtime.qwenQSASelectedKernel ?? false)
    XCTAssertFalse(entry.runtime.qwenPackedGDNPrefill ?? false)
    XCTAssertFalse(entry.runtime.qwenQuantizedKV)
    XCTAssertFalse(entry.runtime.dsparkEnabled)

    settings.expertCacheAutomatic = false
    settings.expertCacheGiB = 8
    let fixed = try ModelLibrary.makeServerCatalog(models: [installed],
      aliases: [:], settings: [swift: settings], powerSavingLimitGBps: nil)
    XCTAssertFalse(fixed.models[0].runtime.nativeAutomaticExpertCache)
    XCTAssertEqual(fixed.models[0].runtime.expertCacheBytes, 8 * 1_073_741_824)
  }

  @MainActor
  func testExistingSettingsAndAliasesSurviveTheEngineChange() throws {
    let suite = "NativeMigration.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: store)
    for kind in [ModelKind.qwen3_8FlashNext, .swift1_5Qwen3_8FlashNext] {
      try library.saveAlias("alias-" + kind.rawValue, for: kind)
      var old = ModelAdvancedSettings.defaults(for: kind)
      old.expertCacheAutomatic = nil
      old.readWorkers = 7
      old.defaultTemperature = 0.4
      old.mtpEnabled = false
      old.qwen8BitCommonTensors = false
      old.promptCacheMode = .disk
      old.applyQwenFixedTuning() // Values written by the previous MLX App.
      let data = try JSONEncoder().encode(old)
      store.set(data, forKey: "modelAdvancedSettings.\(kind.rawValue)")
      let loaded = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: store)
      XCTAssertEqual(loaded.readWorkers, 7)
      XCTAssertEqual(loaded.defaultTemperature, 0.4)
      XCTAssertEqual(loaded.mtpEnabled, false)
      XCTAssertEqual(loaded.qwen8BitCommonTensors, false)
      XCTAssertEqual(loaded.promptCacheMode, .disk)
      XCTAssertEqual(loaded.expertCacheAutomatic, true)
      XCTAssertEqual(library.alias(for: kind), "alias-" + kind.rawValue)
      let model = InstalledModelInfo(url: URL(fileURLWithPath: "/fixture/" + kind.descriptor.directoryName),
        size: 0, quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: kind,
        modelID: kind.descriptor.checkpointModelID)
      let catalog = try ModelLibrary.makeServerCatalog(models: [model], aliases: library.aliases,
        settings: [kind: loaded], powerSavingLimitGBps: nil)
      let entry = try XCTUnwrap(catalog.models.first)
      XCTAssertEqual(entry.id, kind.apiModelID)
      XCTAssertEqual(entry.alias, "alias-" + kind.rawValue)
      XCTAssertFalse(entry.runtime.qwenSortedExpertPrefill ?? true)
      XCTAssertFalse(entry.runtime.qwenQSASelectedKernel ?? true)
      XCTAssertFalse(entry.runtime.deepseekANEPrefill)
    }
  }

  func testMemoryEstimateFollowsTheCacheChoice() throws {
    let sourceRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let data = try Data(contentsOf: sourceRoot.appending(path: "runtime/tests/fixtures/swift_qwen_manifest.json"))
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try data.write(to: directory.appending(path: "manifest.json"))
    let profile = MemoryPlanningProfile(kind: swift, manifest: try InstalledModel.loadManifest(at: directory), config: [:])
    var settings = ModelAdvancedSettings.defaults(for: swift)
    settings.expertCacheAutomatic = false
    settings.expertCacheGiB = 2
    let small = try XCTUnwrap(profile.requestEstimate(settings, mtpAvailable: true, dsparkAvailable: false,
      contextTokens: 4_096, outputTokens: 128))
    settings.expertCacheGiB = 10
    let large = try XCTUnwrap(profile.requestEstimate(settings, mtpAvailable: true, dsparkAvailable: false,
      contextTokens: 4_096, outputTokens: 128))
    XCTAssertEqual(large.model - small.model, 8 * ExpertMemory.gib, accuracy: 3_000_000)
    // 8-bit resident weights hold less than the BF16 file.
    settings.qwen8BitCommonTensors = false
    let bf16 = try XCTUnwrap(profile.requestEstimate(settings, mtpAvailable: true, dsparkAvailable: false,
      contextTokens: 4_096, outputTokens: 128))
    XCTAssertGreaterThan(bf16.model, large.model)
    XCTAssertNotNil(profile.estimate(.defaults(for: swift), mtpAvailable: true, dsparkAvailable: false))
  }
}
