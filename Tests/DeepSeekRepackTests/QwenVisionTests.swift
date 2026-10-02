import CryptoKit
import Foundation
import XCTest
@testable import DeepSeekRepack

final class QwenVisionTests: XCTestCase {
  func testBothModelsUseTheirOwnPinnedVisionSource() throws {
    let qwen = try XCTUnwrap(QwenVisionArtifact.definition(for: .qwen3_8FlashNext))
    let swift = try XCTUnwrap(QwenVisionArtifact.definition(for: .swift1_5Qwen3_8FlashNext))
    XCTAssertEqual(qwen.repository, QwenContract.modelID)
    XCTAssertEqual(qwen.revision, QwenContract.revision)
    XCTAssertEqual(qwen.sourceFile, "model-00001-of-00131.safetensors")
    XCTAssertEqual(qwen.copies.count, 333)
    XCTAssertEqual(swift.repository, SwiftQwenInstalledModelArtifact.repository)
    XCTAssertEqual(swift.revision, SwiftQwenInstalledModelArtifact.revision)
    XCTAssertEqual(swift.sourceFile, "preserved/vision.bin")
    XCTAssertEqual(swift.file.size, 897_864_704)
    for kind in [ModelKind.qwen3_8FlashNext, .swift1_5Qwen3_8FlashNext] {
      XCTAssertTrue(kind.descriptor.supports("imageInput"))
      XCTAssertFalse(kind.descriptor.supports("audioInput"))
      XCTAssertFalse(kind.descriptor.supports("documentInput"))
    }
  }

  func testRangeMappingIncludesPaddingWithoutReadingOtherWeights() async throws {
    let definition = fixture()
    let source = VisionFixtureSource(data: Data([9, 9, 1, 2, 8, 3, 4]))
    let mapped = QwenVisionSource(definition: definition, upstream: source)
    let whole = try await mapped.data(path: "vision/common.bin", range: 0..<6)
    XCTAssertEqual(whole, Data([1, 2, 0, 0, 3, 4]))
    let part = try await mapped.data(path: "vision/common.bin", range: 1..<5)
    XCTAssertEqual(part, Data([2, 0, 0, 3]))
    let calls = await source.calls
    XCTAssertEqual(calls.map(\.path), Array(repeating: "source", count: 4))
    XCTAssertEqual(calls.map(\.range), [2..<4, 5..<7, 3..<4, 5..<6])
    do { _ = try await mapped.data(path: "other", range: 0..<1); XCTFail("must reject unknown paths") } catch {}
    do { _ = try await mapped.data(path: "vision/common.bin"); XCTFail("must reject unbounded reads") } catch {}
  }

  func testSupplementInstallationIsResumableAndDoesNotTouchTextFiles() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let original = Data("unchanged text weights".utf8)
    try original.write(to: root.appending(path: "common.bin"))
    let definition = fixture()
    let source = VisionFixtureSource(data: Data([9, 9, 1, 2, 8, 3, 4]))
    let mapped = QwenVisionSource(definition: definition, upstream: source)
    try FileManager.default.createDirectory(at: root.appending(path: "vision"), withIntermediateDirectories: true)
    try Data([1, 2]).write(to: root.appending(path: "vision/common.bin.partial"))
    try await QwenVisionArtifact.install(at: root, definition: definition, source: mapped, progress: nil)
    XCTAssertEqual(try Data(contentsOf: root.appending(path: "vision/common.bin")), Data([1, 2, 0, 0, 3, 4]))
    XCTAssertEqual(try Data(contentsOf: root.appending(path: "common.bin")), original)
    let calls = await source.calls
    XCTAssertEqual(calls.map(\.range), [5..<7])
    try await QwenVisionArtifact.install(at: root, definition: definition, source: mapped, progress: nil)
    let after = await source.calls.count
    XCTAssertEqual(after, calls.count, "valid image weights must not be downloaded again")
  }

  func testFailedSupplementLeavesExistingFilesAndManifestInPlace() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let project = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let manifest = try Data(contentsOf: project.appending(path: "runtime/tests/fixtures/swift_qwen_manifest.json"))
    try manifest.write(to: root.appending(path: "manifest.json"))
    let text = Data("untouched".utf8)
    try text.write(to: root.appending(path: "common.bin"))
    let source = VisionFixtureSource(data: Data(), fails: true)
    let downloader = InstalledArtifactDownloader(repository: SwiftQwenInstalledModelArtifact.repository,
      revision: SwiftQwenInstalledModelArtifact.revision, source: source,
      expectedKind: .swift1_5Qwen3_8FlashNext, visionSource: source)
    do {
      _ = try await downloader.repair(at: root, invalidFiles: ["vision/common.bin"], progress: nil)
      XCTFail("fixture download must fail")
    } catch {}
    XCTAssertEqual(try Data(contentsOf: root.appending(path: "manifest.json")), manifest)
    XCTAssertEqual(try Data(contentsOf: root.appending(path: "common.bin")), text)
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathExtension("partial").path))
    let calls = await source.calls
    XCTAssertEqual(calls.map(\.path), ["preserved/vision.bin"], "no text manifest or weights may be requested")
  }

  func testBadChecksumDoesNotReplaceExistingVisionAndRejectsSymlink() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let directory = root.appending(path: "vision")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let old = Data([8, 8, 8, 8, 8, 8])
    try old.write(to: directory.appending(path: "common.bin"))
    let source = VisionFixtureSource(data: Data(repeating: 0, count: 7))
    let mapped = QwenVisionSource(definition: fixture(), upstream: source)
    do {
      try await QwenVisionArtifact.install(at: root, definition: fixture(), source: mapped, progress: nil)
      XCTFail("checksum must be checked")
    } catch {}
    XCTAssertEqual(try Data(contentsOf: directory.appending(path: "common.bin")), old)
    try FileManager.default.removeItem(at: directory)
    let outside = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: outside) }
    try FileManager.default.createSymbolicLink(at: directory, withDestinationURL: outside)
    do {
      try await QwenVisionArtifact.install(at: root, definition: fixture(), source: mapped, progress: nil)
      XCTFail("symlink must be rejected")
    } catch {}
    XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)
  }

  private func temporaryDirectory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
  }

  private func fixture() -> QwenVisionArtifact.Definition {
    let data = Data([1, 2, 0, 0, 3, 4])
    return .init(kind: "fixture", checkpointModelID: "fixture", checkpointRevision: "fixture",
      repository: "fixture", revision: "fixture", sourceFile: "source", sourceSize: 7,
      file: InstalledFile(path: "vision/common.bin", size: 6,
        sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()),
      copies: [.init(offset: 0, length: 2, sourceOffset: 2), .init(offset: 4, length: 2, sourceOffset: 5)])
  }
}

private actor VisionFixtureSource: CheckpointSource {
  let bytes: Data
  let fails: Bool
  var calls: [(path: String, range: Range<UInt64>)] = []
  init(data: Data, fails: Bool = false) { bytes = data; self.fails = fails }
  func data(path: String) async throws -> Data {
    calls.append((path, 0..<0))
    throw RepackError.badResponse("unbounded fixture read")
  }
  func data(path: String, range: Range<UInt64>) async throws -> Data {
    calls.append((path, range))
    if fails { throw RepackError.badResponse("fixture failure") }
    return bytes.subdata(in: Int(range.lowerBound)..<Int(range.upperBound))
  }
}
