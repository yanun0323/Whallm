import Foundation

struct QwenPackage: ModelPackage {
  let verifiesInstallation = true
  let installationLabel: String? = "Downloading the Qwen MXFP4 installed model"
  var companions: [CompanionFile] { QwenContract.companions }
  func installedBytes() async throws -> UInt64 { try await QwenInstalledModelArtifact().installedBytes() }
  func makeRepackPlan() async throws -> RepackPlan {
    try await QwenFlashNextCheckpoint().makeRepackPlan()
  }
  func repack(plan: RepackPlan, to output: URL, progress: RepackProgressHandler?) async throws
    -> InstalledManifest
  { try await QwenFlashNextCheckpoint().repack(plan: plan, to: output, progress: progress) }
  func install(to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    try await QwenInstalledModelArtifact().install(to: output, progress: progress)
  }
  func repair(at output: URL, invalidFiles: Set<String>, progress: RepackProgressHandler?)
    async throws -> InstalledManifest
  { try await QwenInstalledModelArtifact().repair(at: output, invalidFiles: invalidFiles, progress: progress) }
  func validate(_ manifest: InstalledManifest) throws -> InstalledManifest {
    try QwenContract.validate(manifest)
  }
}

extension QwenContract {
  static func validate(_ manifest: InstalledManifest, kind: ModelKind = .qwen3_8FlashNext) throws
    -> InstalledManifest
  {
    guard kind.usesQwenEngine, manifest.formatVersion == 2,
      manifest.modelKind == kind,
      manifest.modelID == kind.descriptor.checkpointModelID,
      manifest.revision == kind.descriptor.checkpointRevision,
      manifest.layerCount == QwenContract.layerCount,
      manifest.expertCount == QwenContract.expertCount,
      manifest.selectedExpertCount == QwenContract.selectedExpertCount,
      manifest.expertBlobSize == QwenContract.expertBlobSize,
      manifest.maximumContext == QwenContract.maximumContext,
      manifest.expertRegions == QwenContract.expertRegions,
      manifest.expertQuantization == QwenContract.quantization,
      manifest.ngram == QwenContract.ngram,
      manifest.dspark == nil
    else {
      throw RepackError.incompatibleModel(
        "installed manifest does not match the pinned Qwen model contract")
    }

    let mainPaths = Set(
      [
        "common.bin", "ngram.bin", "config.json", "generation_config.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja", "tokenizer/vocab.json", "tokenizer/merges.txt",
      ] + (0..<QwenContract.layerCount).map {
        String(format: "experts/layer_%02d.bin", $0)
      })
    let mtpPaths: Set<String> = ["mtp/common.bin", "mtp/experts/layer_00.bin"]
    let requiredPaths = mainPaths.union(manifest.mtp == nil ? [] : mtpPaths)
    let actualPaths = Set(manifest.files.map(\.path))
    guard actualPaths == requiredPaths, manifest.files.count == actualPaths.count else {
      throw RepackError.invalidPlan("installed Qwen manifest has an incomplete file set")
    }
    let layerSize = UInt64(QwenContract.expertCount) * QwenContract.expertBlobSize
    for layer in 0..<QwenContract.layerCount {
      let path = String(format: "experts/layer_%02d.bin", layer)
      guard manifest.files.first(where: { $0.path == path })?.size == layerSize else {
        throw RepackError.invalidPlan("installed Qwen expert layer has an invalid size")
      }
    }
    guard let commonSize = manifest.files.first(where: { $0.path == "common.bin" })?.size,
      let ngramSize = manifest.files.first(where: { $0.path == "ngram.bin" })?.size,
      ngramSize == UInt64(QwenContract.ngramShardCount * QwenContract.ngramShardRowCount
        * QwenContract.ngramRowBytes),
      !manifest.commonTensors.isEmpty
    else {
      throw RepackError.invalidPlan("installed Qwen tensor files have an invalid size")
    }
    var names = Set<String>()
    for tensor in manifest.commonTensors {
      guard !tensor.name.hasPrefix("model.visual."), !tensor.name.hasPrefix("mtp."),
        !tensor.name.contains("ngram_embedding.shard_"),
        names.insert(tensor.name).inserted,
        tensor.offset <= commonSize,
        tensor.length <= commonSize - tensor.offset
      else {
        throw RepackError.invalidPlan("invalid Qwen common tensor \(tensor.name)")
      }
    }
    if let mtp = manifest.mtp {
      guard mtp.layerCount == QwenContract.mtpLayerCount,
        mtp.useDedicatedEmbeddings == false,
        mtp.commonTensors.count == QwenContract.mtpCommonTensorCount,
        let mtpCommonSize = manifest.files.first(where: { $0.path == "mtp/common.bin" })?.size,
        manifest.files.first(where: { $0.path == "mtp/experts/layer_00.bin" })?.size
          == UInt64(QwenContract.expertCount) * QwenContract.expertBlobSize
      else {
        throw RepackError.invalidPlan("installed Qwen manifest has an invalid MTP contract")
      }
      var mtpNames = Set<String>()
      for tensor in mtp.commonTensors {
        guard tensor.name.hasPrefix("mtp."),
          !tensor.name.contains(".experts."),
          mtpNames.insert(tensor.name).inserted,
          tensor.offset <= mtpCommonSize,
          tensor.length <= mtpCommonSize - tensor.offset
        else {
          throw RepackError.invalidPlan("invalid Qwen MTP common tensor \(tensor.name)")
        }
      }
    }
    return manifest
  }

}
