import DeepSeekRepack
import Foundation
import XCTest
@testable import DeepSeekV4SSDApp

/// The native-engine model kind: its own API ID and settings, the Swift1.5 installation.
final class NativeQwenModelTests: XCTestCase {
  private let native = ModelKind.swift1_5Qwen3_8FlashNextNative
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

  func testIdentityAndSharedInstallation() {
    XCTAssertEqual(native.apiModelID, "swift1.5-qwen3.8-flash-next-native")
    XCTAssertEqual(native.displayName, "Swift1.5-Qwen3.8-Flash-Next-Native")
    XCTAssertEqual(native.descriptor.directoryName, swift.descriptor.directoryName)
    XCTAssertEqual(native.installationKind, swift)
    XCTAssertEqual(ExpertMemory.blobBytes(for: native), ExpertMemory.blobBytes(for: swift))
    XCTAssertEqual(ExpertMemory.totalExperts(for: .expert, kind: native, manifest: nil), 24_576)
  }

  func testDiscoveryListsTheNativeKindWhereSwiftIsUsable() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try makeSwiftInstallation(in: root)
    let result = InstalledModelDiscovery.find(in: root)
    XCTAssertEqual(Set(result.models.map(\.modelKind)), [swift, native])
    XCTAssertTrue(result.models.allSatisfy { $0.isUsable && $0.url == model.standardizedFileURL.resolvingSymlinksInPath() })
    XCTAssertEqual(Set(result.models.map(\.id)).count, 2)

    // A damaged Swift installation is listed for repair as Swift only.
    let damagedRoot = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: damagedRoot) }
    _ = try makeSwiftInstallation(in: damagedRoot, complete: false)
    let damaged = InstalledModelDiscovery.find(in: damagedRoot)
    XCTAssertEqual(damaged.models.map(\.modelKind), [swift])
    XCTAssertFalse(damaged.models[0].isUsable)
  }

  /// Qwen3.8-Flash-Next-Native reads the Qwen3.8 FP8 installation the same way.
  @MainActor
  func testQwenNativeKindReadsTheQwenInstallation() throws {
    let qwenNative = ModelKind.qwen3_8FlashNextNative, qwen = ModelKind.qwen3_8FlashNext
    XCTAssertEqual(qwenNative.apiModelID, "qwen3.8-flash-next-native")
    XCTAssertEqual(qwenNative.displayName, "Qwen3.8-Flash-Next-Native")
    XCTAssertEqual(qwenNative.descriptor.directoryName, qwen.descriptor.directoryName)
    XCTAssertEqual(qwenNative.installationKind, qwen)
    XCTAssertEqual(qwen.sharingKinds, [qwenNative])
    XCTAssertEqual(ExpertMemory.blobBytes(for: qwenNative), ExpertMemory.blobBytes(for: qwen))

    let suite = "NativeQwenModelTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: defaults)
    XCTAssertEqual(library.plannedInstalledBytes(for: qwenNative), library.plannedInstalledBytes(for: qwen))

    let installed = InstalledModelInfo(url: URL(fileURLWithPath: "/fixture/\(qwen.descriptor.directoryName)"),
      size: 0, quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: qwen,
      modelID: qwen.descriptor.checkpointModelID)
    let catalog = try ModelLibrary.makeServerCatalog(models: [installed, installed.sharing(as: qwenNative)],
      aliases: [:], settings: [qwenNative: .defaults(for: qwenNative)], powerSavingLimitGBps: nil)
    XCTAssertEqual(catalog.models.map(\.id), [qwen.apiModelID, qwenNative.apiModelID])
    XCTAssertEqual(catalog.models[1].modelKind, qwenNative.rawValue)
    XCTAssertEqual(catalog.models[1].path, catalog.models[0].path)
    XCTAssertTrue(catalog.models[1].runtime.nativeAutomaticExpertCache)
    XCTAssertFalse(catalog.models[1].runtime.qwenSortedExpertPrefill ?? false)
  }

  @MainActor
  func testDownloadsAndPlanningUseTheSwiftInstallation() {
    let suite = "NativeQwenModelTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: defaults)
    XCTAssertEqual(library.plannedInstalledBytes(for: native), library.plannedInstalledBytes(for: swift))
    XCTAssertEqual(library.requiredStorageBytes(for: native), library.requiredStorageBytes(for: swift))
    XCTAssertFalse(library.isDownloading(native))
  }

  func testDefaultsUseAutomaticCacheMTPAndEightBitWeights() throws {
    let defaults = ModelAdvancedSettings.defaults(for: native)
    XCTAssertEqual(defaults.expertCacheAutomatic, true)
    XCTAssertEqual(defaults.mtpEnabled, true)
    XCTAssertEqual(defaults.qwen8BitCommonTensors, true)
    XCTAssertEqual(defaults.readWorkers, 8)
    XCTAssertEqual(defaults.promptCacheMode, .memory)
    XCTAssertNoThrow(try defaults.validate(for: native))
    // A saved fixed size stays; settings of other kinds never carry the flag.
    var fixed = defaults
    fixed.expertCacheAutomatic = false
    XCTAssertEqual(fixed.normalized(for: native).expertCacheAutomatic, false)
    var missing = defaults
    missing.expertCacheAutomatic = nil
    XCTAssertEqual(missing.normalized(for: native).expertCacheAutomatic, true)
    XCTAssertNil(ModelAdvancedSettings.defaults(for: swift).normalized(for: swift).expertCacheAutomatic)
  }

  @MainActor
  func testCatalogSendsNativeOptionsAndMLXDefaults() throws {
    let installed = InstalledModelInfo(url: URL(fileURLWithPath: "/fixture/\(swift.descriptor.directoryName)"),
      size: 0, quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: swift,
      modelID: swift.descriptor.checkpointModelID)
    var settings = ModelAdvancedSettings.defaults(for: native)
    let catalog = try ModelLibrary.makeServerCatalog(models: [installed, installed.sharing(as: native)],
      aliases: [:], settings: [native: settings], powerSavingLimitGBps: nil)
    XCTAssertEqual(catalog.models.map(\.id), [swift.apiModelID, native.apiModelID])
    let entry = catalog.models[1]
    XCTAssertEqual(entry.modelKind, native.rawValue)
    XCTAssertEqual(entry.path, catalog.models[0].path)
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
    let fixed = try ModelLibrary.makeServerCatalog(models: [installed.sharing(as: native)],
      aliases: [:], settings: [native: settings], powerSavingLimitGBps: nil)
    XCTAssertFalse(fixed.models[0].runtime.nativeAutomaticExpertCache)
    XCTAssertEqual(fixed.models[0].runtime.expertCacheBytes, 8 * 1_073_741_824)
  }

  func testMemoryEstimateFollowsTheCacheChoice() throws {
    let sourceRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let data = try Data(contentsOf: sourceRoot.appending(path: "runtime/tests/fixtures/swift_qwen_manifest.json"))
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try data.write(to: directory.appending(path: "manifest.json"))
    let profile = MemoryPlanningProfile(kind: native, manifest: try InstalledModel.loadManifest(at: directory), config: [:])
    var settings = ModelAdvancedSettings.defaults(for: native)
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
    XCTAssertNotNil(profile.estimate(.defaults(for: native), mtpAvailable: true, dsparkAvailable: false))
  }
}
