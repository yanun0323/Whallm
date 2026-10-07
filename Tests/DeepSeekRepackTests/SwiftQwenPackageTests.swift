import CryptoKit
import Foundation
import XCTest
@testable import DeepSeekRepack

final class SwiftQwenPackageTests: XCTestCase {
  private func fixture() throws -> Data {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    return try Data(contentsOf: root.appending(path: "runtime/tests/fixtures/swift_qwen_manifest.json"))
  }

  func testPublishedManifestResolvesToIndependentModelWithoutChangingArtifact() throws {
    let data = try fixture()
    XCTAssertEqual(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
      "64ba0d5965085a6aedc0e81035b4d9fd7a6f5ad48581d2caf2034f39dc233b38")
    let raw = try JSONDecoder().decode(InstalledManifest.self, from: data)
    XCTAssertEqual(raw.modelKind, .qwen3_8FlashNext)
    let installed = try InstalledModel.decodeManifest(data)
    XCTAssertEqual(installed.modelKind, .swift1_5Qwen3_8FlashNext)
    XCTAssertEqual(installed.modelID, "ukisai/Swift1.5-Qwen3.8-Flash-Next")
    XCTAssertEqual(installed.files, raw.files)
    XCTAssertEqual(installed.commonTensors, raw.commonTensors)
    XCTAssertEqual(installed.mtp, raw.mtp)
    XCTAssertEqual(installed.files.count, 59)
    XCTAssertEqual(installed.files.reduce(UInt64(0)) { $0 + $1.size }, 126_816_657_774)
    XCTAssertEqual(InstalledModel.verificationFiles(for: installed).reduce(UInt64(0)) { $0 + $1.size },
      SwiftQwenInstalledModelArtifact.installedBytes)
    XCTAssertNoThrow(try InstalledModel.decodeManifest(JSONEncoder().encode(installed)))
    XCTAssertThrowsError(try QwenContract.validate(installed), "FP8 must not accept Swift weights")
  }

  func testRejectsUnpinnedSourceWrongKindAndMissingMTP() throws {
    let original = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture()) as? [String: Any])
    for (field, value) in [("modelID", "unknown/model"), ("revision", String(repeating: "0", count: 40)),
      ("modelKind", "deepseek-v4.1")] {
      var invalid = original
      invalid[field] = value
      XCTAssertThrowsError(try InstalledModel.decodeManifest(JSONSerialization.data(withJSONObject: invalid)), field)
    }
    var missingMTP = original
    missingMTP.removeValue(forKey: "mtp")
    missingMTP["files"] = (original["files"] as! [[String: Any]]).filter {
      !($0["path"] as! String).hasPrefix("mtp/")
    }
    XCTAssertThrowsError(try InstalledModel.decodeManifest(JSONSerialization.data(withJSONObject: missingMTP)))
  }

  func testArtifactDownloaderUsesOwnIdentityAndPreservesPublishedBytes() async throws {
    XCTAssertEqual(SwiftQwenInstalledModelArtifact.repository,
      "Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4")
    XCTAssertEqual(SwiftQwenInstalledModelArtifact.revision, "257cb509d72ecc07519be35a8b7558fde3b31b21")
    let data = try fixture()
    let source = SwiftManifestSource(bytes: data)
    let artifact = try await SwiftQwenInstalledModelArtifact.makeDownloader(source: source).manifest()
    XCTAssertEqual(artifact.data, data)
    XCTAssertEqual(artifact.manifest.modelKind, .swift1_5Qwen3_8FlashNext)
    do {
      _ = try await InstalledArtifactDownloader(repository: "FP8", revision: "fixture", source: source).manifest()
      XCTFail("The FP8 downloader must reject the Swift artifact")
    } catch { XCTAssertTrue(error is RepackError) }
    var fp8 = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    fp8["modelID"] = QwenContract.modelID
    fp8["revision"] = QwenContract.revision
    let fp8Source = SwiftManifestSource(bytes: try JSONSerialization.data(withJSONObject: fp8))
    do {
      _ = try await SwiftQwenInstalledModelArtifact.makeDownloader(source: fp8Source).manifest()
      XCTFail("The Swift downloader must reject FP8 weights")
    } catch { XCTAssertTrue(error is RepackError) }
  }

  func testPreparedArtifactCannotFallBackToFP8Repacking() async {
    do {
      _ = try await ModelPackages.package(for: .swift1_5Qwen3_8FlashNext).makeRepackPlan()
      XCTFail("Swift must never convert the FP8 checkpoint")
    } catch { XCTAssertTrue(error is RepackError) }
  }
}

private struct SwiftManifestSource: CheckpointSource {
  let bytes: Data
  func data(path: String) async throws -> Data {
    guard path == "manifest.json" else { throw RepackError.invalidPlan("unexpected path: \(path)") }
    return bytes
  }
  func data(path: String, range: Range<UInt64>) async throws -> Data {
    try await data(path: path).subdata(in: Int(range.lowerBound)..<Int(range.upperBound))
  }
}
