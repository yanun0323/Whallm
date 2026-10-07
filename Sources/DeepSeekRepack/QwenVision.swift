import Foundation

/// Required image weights live beside the unchanged published text manifest.
/// Both models have their own pinned source, even when tensor bytes coincide.
public enum QwenVisionArtifact {
  public static let installedBytes: UInt64 = 897_864_704
  public static var modelKinds: [ModelKind] { definitions.compactMap { ModelKind(rawValue: $0.kind) } }

  struct Copy: Decodable, Sendable {
    let offset: UInt64
    let length: UInt64
    let sourceOffset: UInt64
  }

  struct Definition: Decodable, Sendable {
    let kind: String
    let checkpointModelID: String
    let checkpointRevision: String
    let repository: String
    let revision: String
    let sourceFile: String
    let sourceSize: UInt64
    let file: InstalledFile
    let copies: [Copy]
  }

  static let definitions: [Definition] = {
    let url = Bundle.main.url(forResource: "QwenVision", withExtension: "json")
      ?? Bundle.module.url(forResource: "QwenVision", withExtension: "json")
    struct Catalog: Decodable { let version: Int; let models: [Definition] }
    do {
      guard let url else { throw RepackError.invalidPlan("Qwen vision catalog is missing") }
      let catalog = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: url))
      guard catalog.version == 1, catalog.models.count == 2,
        Set(catalog.models.map(\.kind)).count == 2 else {
        throw RepackError.invalidPlan("invalid Qwen vision catalog")
      }
      for model in catalog.models {
        guard let kind = ModelKind(rawValue: model.kind), kind.usesQwenEngine,
          kind.descriptor.checkpointModelID == model.checkpointModelID,
          kind.descriptor.checkpointRevision == model.checkpointRevision,
          model.file.path == "vision/common.bin", model.file.size == installedBytes,
          model.file.sha256.count == 64, !model.copies.isEmpty
        else { throw RepackError.invalidPlan("invalid Qwen vision identity") }
        var end: UInt64 = 0
        for copy in model.copies {
          guard copy.offset >= end, copy.offset <= model.file.size,
            copy.length > 0, copy.length <= model.file.size - copy.offset,
            copy.sourceOffset <= model.sourceSize,
            copy.length <= model.sourceSize - copy.sourceOffset
          else { throw RepackError.invalidPlan("invalid Qwen vision copy range") }
          end = copy.offset + copy.length
        }
      }
      return catalog.models
    } catch { preconditionFailure("Cannot load Qwen vision packages: \(error)") }
  }()

  static func definition(for kind: ModelKind) -> Definition? {
    definitions.first { $0.kind == kind.rawValue }
  }

  public static func requiredFiles(for manifest: InstalledManifest) -> [InstalledFile] {
    guard let kind = manifest.modelKind, kind.usesQwenEngine,
      let definition = definition(for: kind),
      manifest.modelID == definition.checkpointModelID,
      manifest.revision == definition.checkpointRevision
    else { return [] }
    return [definition.file]
  }

  static func source(for kind: ModelKind, upstream: (any CheckpointSource)? = nil) -> any CheckpointSource {
    let definition = definition(for: kind)!
    return QwenVisionSource(definition: definition,
      upstream: upstream ?? HuggingFaceSource(modelID: definition.repository, revision: definition.revision))
  }

  /// Complete an older installation without moving or rewriting its text files.
  public static func install(at root: URL, progress: RepackProgressHandler? = nil) async throws {
    let manifest = try InstalledModel.loadManifest(at: root)
    guard let kind = manifest.modelKind, kind.usesQwenEngine else {
      throw RepackError.incompatibleModel("vision installation requires a Qwen model")
    }
    try await install(at: root, definition: definition(for: kind)!, source: source(for: kind), progress: progress)
  }

  static func install(at root: URL, definition: Definition, source: any CheckpointSource,
    progress: RepackProgressHandler?) async throws
  {
    let root = root.standardizedFileURL.resolvingSymlinksInPath()
    let destination = try safeFileURL(root: root, path: definition.file.path)
    let directory = destination.deletingLastPathComponent()
    guard directory.resolvingSymlinksInPath().path == directory.path,
      destination.resolvingSymlinksInPath().path == destination.path
    else { throw RepackError.invalidPlan("unsafe Qwen vision destination") }
    if (try? fileSize(destination)) == definition.file.size,
      try sha256(destination) == definition.file.sha256
    {
      progress?(RepackProgress(copiedBytes: definition.file.size, downloadedBytes: 0, totalBytes: definition.file.size))
      return
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let partial = destination.appendingPathExtension("partial")
    guard partial.resolvingSymlinksInPath().path == partial.path else {
      throw RepackError.invalidPlan("unsafe Qwen vision partial file")
    }
    if let available = try directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
      .volumeAvailableCapacityForImportantUsage, available >= 0,
      UInt64(available) < definition.file.size - min(definition.file.size, (try? fileSize(partial)) ?? 0)
    { throw RepackError.insufficientStorage(required: definition.file.size, available: available) }
    let state = VisionProgress(total: definition.file.size, progress: progress)
    _ = try await InstalledArtifactFileDownloader(source: source).run(file: definition.file, to: partial) {
      copied, downloaded in await state.advance(copied, downloaded)
    }
    try Task.checkCancellation()
    // POSIX rename replaces the old file atomically, retaining it on download failure.
    guard rename(partial.path, destination.path) == 0 else {
      throw RepackError.invalidPlan("cannot finish Qwen vision installation")
    }
  }
}

struct QwenVisionSource: CheckpointSource {
  let definition: QwenVisionArtifact.Definition
  let upstream: any CheckpointSource

  func data(path: String) async throws -> Data {
    throw RepackError.invalidPlan("Qwen vision downloads require bounded ranges")
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    guard path == definition.file.path, range.upperBound <= definition.file.size,
      range.count <= 8 * 1_024 * 1_024
    else { throw RepackError.invalidPlan("invalid Qwen vision download range") }
    var result = Data(repeating: 0, count: range.count)
    for copy in definition.copies {
      let start = max(copy.offset, range.lowerBound)
      let end = min(copy.offset + copy.length, range.upperBound)
      guard start < end else { continue }
      try Task.checkCancellation()
      let sourceStart = copy.sourceOffset + start - copy.offset
      let data = try await upstream.data(path: definition.sourceFile, range: sourceStart..<(sourceStart + end - start))
      guard data.count == Int(end - start) else { throw RepackError.badResponse("truncated Qwen vision range") }
      result.replaceSubrange(Int(start - range.lowerBound)..<Int(end - range.lowerBound), with: data)
    }
    return result
  }
}

private actor VisionProgress {
  let total: UInt64
  let progress: RepackProgressHandler?
  var copied: UInt64 = 0
  var downloaded: UInt64 = 0
  init(total: UInt64, progress: RepackProgressHandler?) { self.total = total; self.progress = progress }
  func advance(_ copied: UInt64, _ downloaded: UInt64) {
    self.copied += copied
    self.downloaded += downloaded
    progress?(RepackProgress(copiedBytes: self.copied, downloadedBytes: self.downloaded, totalBytes: total))
  }
}
