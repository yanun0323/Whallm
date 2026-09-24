import Foundation

struct MiMoPackage: ModelPackage {
  let verifiesInstallation = true
  let installationLabel: String? = "Downloading the complete MiMo installed model"
  var companions: [CompanionFile] {
    MiMoContract.preservedFiles.map { CompanionFile(source: $0, destination: "checkpoint/\($0)") }
  }
  func installedBytes() async throws -> UInt64 {
    let manifest = try await MiMoInstalledModelArtifact.makeDownloader().manifest().manifest
    return try manifest.files.reduce(UInt64(0)) { total, file in
      let sum = total.addingReportingOverflow(file.size)
      guard !sum.overflow else { throw RepackError.invalidPlan("installed file sizes overflow") }
      return sum.partialValue
    }
  }
  func makeRepackPlan() async throws -> RepackPlan {
    throw RepackError.invalidPlan("MiMo uses a prepared artifact; local conversion uses Scripts/convert_mimo_model.py")
  }
  func repack(plan: RepackPlan, to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    throw RepackError.invalidPlan("MiMo downloads require no conversion")
  }
  func install(to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    try await MiMoInstalledModelArtifact.makeDownloader().install(to: output, progress: progress)
  }
  func repair(at output: URL, invalidFiles: Set<String>, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    try await MiMoInstalledModelArtifact.makeDownloader().repair(at: output, invalidFiles: invalidFiles, progress: progress)
  }
  func validate(_ manifest: InstalledManifest) throws -> InstalledManifest { try MiMoContract.validate(manifest) }
}

/// Complete public artifact; inventory, sizes and remote digests verified.
/// This is the artifact commit, not the upstream checkpoint revision.
enum MiMoInstalledModelArtifact {
  static let repository = "Yanun/MiMo-V2.6-Flash-RL-MXFP4"
  static let revision = "a25b1711c5c9f97e0562f39c9cd9f043f6be3f09"

  static func makeDownloader() throws -> InstalledArtifactDownloader {
    return InstalledArtifactDownloader(repository: repository, revision: revision,
      source: HuggingFaceSource(modelID: repository, revision: revision), expectedKind: .mimoV26FlashRL)
  }
}

enum MiMoContract {
  static let modelID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
  static let revision = "5711b268169967567844e1e560e8a3966da959b1"
  static let preservedFiles = [
    ".gitattributes", "MiMo_V2_6_technical_report.pdf", "README.md", "assets/architecture.png",
    "audio_tokenizer/chat_template.jinja", "audio_tokenizer/config.json",
    "audio_tokenizer/generation_config.json", "audio_tokenizer/model.safetensors",
    "audio_tokenizer/tokenizer_config.json", "chat_template.jinja", "config.json",
    "configuration_mimo_v2.py", "dflash/config.json", "dflash/dflash.py",
    "dflash/dflash_draft_model.safetensors", "dflash/mask_embedding.pt",
    "dflash/model.safetensors.index.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "model_mtp.safetensors", "modeling_mimo_v2.py",
    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
  ]
  static let requiredPaths = Set(
    ["common.bin", "config.json", "generation_config.json", "checkpoint-map.json", "preservation.json",
     "checkpoint/common.bin", "vision/common.bin", "audio/common.bin", "mtp/common.bin"]
    + ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"].map { "tokenizer/\($0)" }
    + (0..<47).map { String(format: "experts/layer_%02d.bin", $0) }
    + (0..<64).map { "checkpoint/headers/model_pp0_ep\($0)_shard0.safetensors.header" }
    + preservedFiles.map { "checkpoint/\($0)" })

  static func validate(_ manifest: InstalledManifest) throws -> InstalledManifest {
    guard manifest.formatVersion == 4, manifest.modelKind == .mimoV26FlashRL,
      manifest.modelID == modelID, manifest.revision == revision,
      manifest.layerCount == 47, manifest.expertCount == 256, manifest.selectedExpertCount == 8,
      manifest.expertBlobSize == 13_369_344, manifest.maximumContext == 1_048_576,
      manifest.expertRegions == ModelContract.expertRegions,
      manifest.expertQuantization == ExpertQuantizationDescriptor(mode: "mxfp4", bits: 4, groupSize: 32, conversionVersion: 1),
      manifest.dspark == nil, manifest.mtp == nil, manifest.ngram == nil, manifest.engram == nil
    else { throw RepackError.incompatibleModel("installed manifest does not match the complete MiMo contract") }
    let paths = Set(manifest.files.map(\.path))
    guard paths == requiredPaths, manifest.files.count == paths.count,
      manifest.files.allSatisfy({ $0.size > 0 && $0.sha256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil })
    else { throw RepackError.invalidPlan("installed MiMo artifact is missing original checkpoint components") }
    for layer in 0..<47 {
      let path = String(format: "experts/layer_%02d.bin", layer)
      guard manifest.files.first(where: { $0.path == path })?.size == 256 * manifest.expertBlobSize
      else { throw RepackError.invalidPlan("installed MiMo expert layer has an invalid size") }
    }
    let commonSize = manifest.files.first { $0.path == "common.bin" }!.size
    var names = Set<String>()
    var end: UInt64 = 0
    guard manifest.commonTensors.count == 331 else { throw RepackError.invalidPlan("incomplete MiMo backbone") }
    for tensor in manifest.commonTensors.sorted(by: { $0.offset < $1.offset }) {
      guard names.insert(tensor.name).inserted, ["BF16", "F32"].contains(tensor.dtype),
        tensor.offset >= end, tensor.offset % 256 == 0, tensor.offset <= commonSize,
        tensor.length <= commonSize - tensor.offset
      else { throw RepackError.invalidPlan("invalid MiMo common tensor: \(tensor.name)") }
      end = tensor.offset + tensor.length
    }
    return manifest
  }
}
