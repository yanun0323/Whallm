import Foundation

/// Independent installed model using the existing Qwen inference contract.
struct SwiftQwenPackage: ModelPackage {
  let verifiesInstallation = true
  let installationLabel: String? = "Downloading the Qwen MXFP4 installed model"
  var companions: [CompanionFile] { QwenContract.companions }

  func installedBytes() async throws -> UInt64 {
    let manifest = try await SwiftQwenInstalledModelArtifact.makeDownloader().manifest().manifest
    return try InstalledModel.verificationFiles(for: manifest).reduce(UInt64(0)) { total, file in
      let sum = total.addingReportingOverflow(file.size)
      guard !sum.overflow else { throw RepackError.invalidPlan("installed file sizes overflow") }
      return sum.partialValue
    }
  }

  func makeRepackPlan() async throws -> RepackPlan {
    throw RepackError.invalidPlan("Swift Qwen uses a prepared artifact; no FP8 checkpoint conversion is needed")
  }

  func repack(plan: RepackPlan, to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    throw RepackError.invalidPlan("Swift Qwen downloads require no conversion")
  }

  func install(to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    try await SwiftQwenInstalledModelArtifact.makeDownloader().install(to: output, progress: progress)
  }

  func repair(at output: URL, invalidFiles: Set<String>, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    try await SwiftQwenInstalledModelArtifact.makeDownloader().repair(at: output, invalidFiles: invalidFiles, progress: progress)
  }

  func validate(_ manifest: InstalledManifest) throws -> InstalledManifest {
    guard manifest.mtp != nil else {
      throw RepackError.incompatibleModel("the Swift Qwen artifact must include its own MTP weights")
    }
    return try QwenContract.validate(manifest, kind: .swift1_5Qwen3_8FlashNext)
  }
}

public enum SwiftQwenInstalledModelArtifact {
  public static let repository = "Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4"
  // Artifact commit, distinct from the upstream checkpoint revision in the manifest.
  public static let revision = "257cb509d72ecc07519be35a8b7558fde3b31b21"
  // 59 unchanged manifest files (including MTP) plus the required vision blob.
  // Audit files are not installed; the vision layout is bundled with the App.
  public static let installedBytes: UInt64 = 126_816_657_774 + QwenVisionArtifact.installedBytes

  static func makeDownloader(source: (any CheckpointSource)? = nil) -> InstalledArtifactDownloader {
    InstalledArtifactDownloader(repository: repository, revision: revision,
      source: source ?? HuggingFaceSource(modelID: repository, revision: revision),
      expectedKind: .swift1_5Qwen3_8FlashNext)
  }
}
