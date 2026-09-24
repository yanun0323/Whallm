import Foundation
import XCTest
@testable import DeepSeekRepack

final class MiMoPackageTests: XCTestCase {
  private func fixture(omitting: String? = nil) -> InstalledManifest {
    let tensors = (0..<331).map { i in
      InstalledTensor(name: "model.fixture.\(i)", dtype: "BF16", shape: [1],
        offset: UInt64(i * 256), length: 2)
    }
    let files = MiMoContract.requiredPaths.filter { $0 != omitting }.sorted().map { path in
      InstalledFile(path: path, size: path.hasPrefix("experts/") ? 256 * 13_369_344
        : path == "common.bin" ? 330 * 256 + 2 : 1, sha256: String(repeating: "0", count: 64))
    }
    return InstalledManifest(formatVersion: 4, modelID: MiMoContract.modelID,
      revision: MiMoContract.revision, layerCount: 47, expertCount: 256, selectedExpertCount: 8,
      expertBlobSize: 13_369_344, files: files, commonTensors: tensors,
      expertRegions: ModelContract.expertRegions, modelKind: .mimoV26FlashRL,
      maximumContext: 1_048_576,
      expertQuantization: ExpertQuantizationDescriptor(mode: "mxfp4", bits: 4, groupSize: 32, conversionVersion: 1))
  }

  func testRequiresAllCheckpointComponentsRatherThanTextOnlyWeights() throws {
    XCTAssertEqual(try MiMoContract.validate(fixture()).files.count, 151)
    for path in ["vision/common.bin", "audio/common.bin", "mtp/common.bin",
      "checkpoint/model_mtp.safetensors", "checkpoint/audio_tokenizer/model.safetensors",
      "checkpoint/dflash/dflash_draft_model.safetensors", "checkpoint/dflash/mask_embedding.pt",
      "checkpoint/headers/model_pp0_ep63_shard0.safetensors.header", "checkpoint/README.md"] {
      XCTAssertThrowsError(try MiMoContract.validate(fixture(omitting: path)), path)
    }
    XCTAssertFalse(ModelKind.mimoV26FlashRL.descriptor.supports("mtp"))
    XCTAssertFalse(ModelKind.mimoV26FlashRL.descriptor.supports("dspark"))
  }

  func testArtifactDownloaderChecksExpectedModelKind() async throws {
    let source = MiMoManifestSource(bytes: try JSONEncoder().encode(fixture()))
    let correct = InstalledArtifactDownloader(repository: "fixture", revision: "fixture", source: source,
      expectedKind: .mimoV26FlashRL)
    let artifact = try await correct.manifest()
    XCTAssertEqual(artifact.manifest.modelKind, .mimoV26FlashRL)
    let wrong = InstalledArtifactDownloader(repository: "fixture", revision: "fixture", source: source)
    do {
      _ = try await wrong.manifest()
      XCTFail("Qwen downloader accepted a MiMo artifact")
    } catch { XCTAssertTrue(error is RepackError) }
  }

  func testOptionalCompleteLocalManifest() throws {
    guard let path = ProcessInfo.processInfo.environment["WHALLM_MIMO_MODEL"] else {
      throw XCTSkip("Set WHALLM_MIMO_MODEL to validate a complete converted checkpoint")
    }
    let manifest = try InstalledModel.loadManifest(at: URL(fileURLWithPath: path))
    XCTAssertEqual(manifest.modelKind, .mimoV26FlashRL)
    XCTAssertEqual(manifest.files.count, 151)
    XCTAssertEqual(manifest.commonTensors.count, 331)
  }
}

private struct MiMoManifestSource: CheckpointSource {
  let bytes: Data
  func data(path: String) async throws -> Data { bytes }
  func data(path: String, range: Range<UInt64>) async throws -> Data {
    bytes.subdata(in: Int(range.lowerBound)..<Int(range.upperBound))
  }
}
