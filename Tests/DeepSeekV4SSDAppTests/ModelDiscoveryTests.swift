import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ModelDiscoveryTests: XCTestCase {
  private func availableInstalledModel() -> URL? {
    let fileManager = FileManager.default
    let project = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    let candidates = [
      fileManager.homeDirectoryForCurrentUser
        .appending(path: ".dsmodel", directoryHint: .isDirectory)
        .appending(path: "deepseek-v4-flash-0731.dsv4", directoryHint: .isDirectory),
      project.appending(path: "scratch/deepseek-v4-flash-0731.dsv4"),
    ]
    return candidates.first {
      fileManager.fileExists(atPath: $0.appending(path: "manifest.json").path)
    }
  }

  func testDiscoveryAcceptsCompleteInstalledModelAndRejectsMissingModel() throws {
    guard let model = availableInstalledModel() else {
      throw XCTSkip("The complete installed model is not available.")
    }

    XCTAssertEqual(InstalledModelDiscovery.inspect(model)?.url, model.standardizedFileURL)

    let empty = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }
    XCTAssertNil(InstalledModelDiscovery.inspect(empty))
  }

  func testDiscoveryKeepsModelWithMissingFilesForRepair() throws {
    guard let source = availableInstalledModel()?.appending(path: "manifest.json") else {
      throw XCTSkip("The installed model manifest is not available.")
    }
    let model = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: model) }
    try FileManager.default.copyItem(at: source, to: model.appending(path: "manifest.json"))

    let discovered = try XCTUnwrap(InstalledModelDiscovery.inspect(model))
    XCTAssertFalse(discovered.isUsable)
    XCTAssertFalse(discovered.quickIssues.isEmpty)
  }

  func testDiscoveryReportsInvalidManifest() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let model = root.appending(path: "broken.dsv4")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try Data("{}".utf8).write(to: model.appending(path: "manifest.json"))

    let result = InstalledModelDiscovery.find(in: root)
    XCTAssertTrue(result.models.isEmpty)
    XCTAssertEqual(result.invalidModelURLs.map(\.lastPathComponent), ["broken.dsv4"])
  }

  func testDiscoveryRecognizesQwenFormatTwoForRepair() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try JSONSerialization.data(withJSONObject: qwenManifestFixture()).write(
      to: root.appending(path: "manifest.json"))

    let model = try XCTUnwrap(InstalledModelDiscovery.inspect(root))

    XCTAssertEqual(model.modelKind, .qwen3_8FlashNext)
    XCTAssertEqual(model.modelID, "Qwen/Qwen3.8-Flash-Next-FP8")
    XCTAssertEqual(model.modelKindLabel, "Qwen3.8 Flash Next")
    XCTAssertFalse(model.hasMTP)
    XCTAssertFalse(model.hasDSpark)
    XCTAssertFalse(model.isUsable)
  }

  func testDiscoveryRejectsModelWithoutOfficialEncoder() throws {
    guard let source = availableInstalledModel()?.appending(path: "manifest.json"),
      let data = try? Data(contentsOf: source)
    else {
      throw XCTSkip("The installed model manifest is not available.")
    }
    var manifest = try XCTUnwrap(
      JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    let files = try XCTUnwrap(manifest["files"] as? [[String: Any]])
    manifest["files"] = files.filter { $0["path"] as? String != "encoding/encoding_dsv4.py" }

    let model = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: model) }
    try JSONSerialization.data(withJSONObject: manifest).write(
      to: model.appending(path: "manifest.json")
    )

    XCTAssertNil(InstalledModelDiscovery.inspect(model))
  }

  @MainActor
  func testModelLibraryDefaultsToHiddenHomeFolder() {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }

    XCTAssertEqual(
      ModelLibrary(defaults: defaults).rootURL.path,
      FileManager.default.homeDirectoryForCurrentUser.appending(path: ".dsmodel").path
    )
  }

  @MainActor
  func testModelLibraryRestoresSelectedQwenInstallKind() {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")

    XCTAssertEqual(ModelLibrary(defaults: defaults).selectedModelKind, .qwen3_8FlashNext)
  }

  @MainActor
  func testModelLibraryOffersSupportedKindsWithoutInstalledModels() {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: defaults)

    XCTAssertEqual(
      ModelLibrary.supportedModelKinds,
      [.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext, .mimoV26FlashRL]
    )
    XCTAssertNil(library.usableModel(for: .deepSeekV4))
    XCTAssertNil(library.usableModel(for: .deepSeekV41))
    XCTAssertNil(library.usableModel(for: .qwen3_8FlashNext))
    XCTAssertTrue(library.needsSelectedModelDownload)

    library.selectedModelKind = .qwen3_8FlashNext

    XCTAssertTrue(library.needsSelectedModelDownload)
  }

  @MainActor
  func testModelLibraryUsesPinnedInstalledModelSizes() {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: defaults)

    XCTAssertEqual(library.plannedInstalledBytes(for: .deepSeekV4), 166_878_580_480)
    XCTAssertEqual(library.plannedInstalledBytes(for: .deepSeekV41), 501_382_643_728)
    XCTAssertEqual(library.plannedInstalledBytes(for: .qwen3_8FlashNext), 125_291_490_955)
  }

  @MainActor
  func testStartingDownloadKeepsSelectedModel() {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    defaults.set("/dev/null/whallm-test", forKey: ModelLibrary.rootPreference)
    let library = ModelLibrary(defaults: defaults)

    library.startDownload(
      for: .deepSeekV4,
      to: URL(fileURLWithPath: "/dev/null/whallm-test/deepseek.dsv4")
    )

    XCTAssertEqual(library.selectedModelKind, .qwen3_8FlashNext)
  }

  @MainActor
  func testResumingDownloadKeepsSelectedModel() throws {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let destination = root.appending(path: "deepseek.dsv4")
    try FileManager.default.createDirectory(
      at: destination.appendingPathExtension("partial"),
      withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: root) }
    defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    defaults.set("/dev/null/whallm-test", forKey: ModelLibrary.rootPreference)
    defaults.set(true, forKey: "modelDownloadWasActive")
    defaults.set(destination.path, forKey: "modelDownloadDestination")
    defaults.set("deepseek-v4", forKey: "modelDownloadModelKind")
    let library = ModelLibrary(defaults: defaults)

    library.resumeDownloadIfNeeded()

    XCTAssertEqual(library.selectedModelKind, .qwen3_8FlashNext)
  }

  func testDownloadStorageBlockUsesRequiredAndAvailableBytes() {
    XCTAssertNil(
      ModelLibrary.storageDownloadBlock(requiredBytes: 100, availableBytes: 100)
    )
    XCTAssertEqual(
      ModelLibrary.storageDownloadBlock(requiredBytes: 101, availableBytes: 100),
      .insufficientStorage(requiredBytes: 101, availableBytes: 100)
    )
    XCTAssertEqual(
      ModelLibrary.storageDownloadBlock(requiredBytes: 0, availableBytes: nil),
      .storageUnavailable
    )
  }
}

private func qwenManifestFixture() -> [String: Any] {
  let checksum = String(repeating: "0", count: 64)
  var files: [[String: Any]] = [
    ["path": "common.bin", "size": 1, "sha256": checksum],
    ["path": "ngram.bin", "size": 51_200_245_760, "sha256": checksum],
  ]
  for path in [
    "config.json", "generation_config.json", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja",
    "tokenizer/vocab.json", "tokenizer/merges.txt",
  ] {
    files.append(["path": path, "size": 1, "sha256": checksum])
  }
  for layer in 0..<48 {
    files.append([
      "path": String(format: "experts/layer_%02d.bin", layer),
      "size": 1_336_934_400,
      "sha256": checksum,
    ])
  }
  return [
    "formatVersion": 2,
    "modelKind": "qwen3.8-flash-next",
    "modelID": "Qwen/Qwen3.8-Flash-Next-FP8",
    "revision": "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
    "layerCount": 48,
    "expertCount": 512,
    "selectedExpertCount": 10,
    "expertBlobSize": 2_611_200,
    "maximumContext": 262_144,
    "files": files,
    "commonTensors": [
      ["name": "fixture", "dtype": "U8", "shape": [1], "offset": 0, "length": 1]
    ],
    "expertRegions": [
      ["name": "gate_up.weight", "dtype": "U32", "shape": [1_280, 320], "offset": 0, "length": 1_638_400],
      ["name": "gate_up.scale", "dtype": "U8", "shape": [1_280, 80], "offset": 1_638_400, "length": 102_400],
      ["name": "down.weight", "dtype": "U32", "shape": [2_560, 80], "offset": 1_740_800, "length": 819_200],
      ["name": "down.scale", "dtype": "U8", "shape": [2_560, 20], "offset": 2_560_000, "length": 51_200],
    ],
    "expertQuantization": [
      "mode": "mxfp4", "bits": 4, "groupSize": 32, "conversionVersion": 2,
    ],
    "ngram": [
      "file": "ngram.bin", "dtype": "F8_E4M3", "rowBytes": 160,
      "shardCount": 128,
      "shardRowCount": 2_500_012,
      "headOffsets": [
        0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165,
        120_000_228, 140_000_297, 160_000_374, 180_000_455, 200_000_548,
        220_000_655, 240_000_802, 260_000_955, 280_001_114, 300_001_275,
      ],
      "headVocabSizes": [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063,
        20_000_069, 20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147,
        20_000_153, 20_000_159, 20_000_161, 20_000_171,
      ],
    ],
  ]
}
