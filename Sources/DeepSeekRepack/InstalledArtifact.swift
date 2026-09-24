import CryptoKit
import Foundation

struct InstalledArtifactFileDownloader: Sendable {
  private let source: any CheckpointSource
  private let chunkSize: UInt64

  init(source: any CheckpointSource, chunkSize: UInt64 = 8 * 1_024 * 1_024) {
    self.source = source
    self.chunkSize = chunkSize
  }

  func run(
    file: InstalledFile,
    to destination: URL,
    progress: @Sendable (UInt64, UInt64) async -> Void
  ) async throws -> String {
    guard chunkSize > 0 else {
      throw RepackError.invalidPlan("installed artifact chunk size must be positive")
    }
    let fileManager = FileManager.default
    try fileManager.createDirectory(
      at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    if !fileManager.fileExists(atPath: destination.path) {
      guard fileManager.createFile(atPath: destination.path, contents: nil) else {
        throw RepackError.invalidPlan("cannot create \(file.path)")
      }
    }

    var existingBytes = try fileSize(destination)
    if existingBytes > file.size {
      try truncate(destination)
      existingBytes = 0
    } else if existingBytes == file.size {
      let digest = try sha256(destination)
      if digest == file.sha256 {
        await progress(existingBytes, 0)
        return digest
      }
      try truncate(destination)
      existingBytes = 0
    }

    var hasher = SHA256()
    if existingBytes > 0 {
      let input = try FileHandle(forReadingFrom: destination)
      defer { try? input.close() }
      var remaining = existingBytes
      var copiedSinceProgress: UInt64 = 0
      let progressIntervalBytes: UInt64 = 256 * 1_024 * 1_024
      while remaining > 0 {
        try Task.checkCancellation()
        let count = Int(min(UInt64(8 * 1_024 * 1_024), remaining))
        guard let data = try input.read(upToCount: count), !data.isEmpty else {
          throw RepackError.invalidPlan("cannot resume \(file.path)")
        }
        hasher.update(data: data)
        remaining -= UInt64(data.count)
        copiedSinceProgress += UInt64(data.count)
        if copiedSinceProgress >= progressIntervalBytes {
          await progress(copiedSinceProgress, 0)
          copiedSinceProgress = 0
        }
      }
      if copiedSinceProgress > 0 {
        await progress(copiedSinceProgress, 0)
      }
    }

    let output = try FileHandle(forWritingTo: destination)
    do {
      try output.seekToEnd()
      var position = existingBytes
      while position < file.size {
        try Task.checkCancellation()
        let end = min(position + chunkSize, file.size)
        let data = try await source.data(path: file.path, range: position..<end)
        try output.write(contentsOf: data)
        hasher.update(data: data)
        position = end
        await progress(UInt64(data.count), UInt64(data.count))
      }
      try output.synchronize()
      try output.close()
    } catch {
      try? output.close()
      throw error
    }

    let digest = hex(hasher.finalize())
    guard digest == file.sha256 else {
      try truncate(destination)
      throw RepackError.invalidPlan("installed artifact SHA-256 mismatch for \(file.path)")
    }
    return digest
  }

  private func truncate(_ url: URL) throws {
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    try handle.truncate(atOffset: 0)
  }
}

struct InstalledArtifactDownloader: Sendable {
  private let repository: String
  private let revision: String
  private let source: any CheckpointSource
  private let expectedKind: ModelKind
  private let fileConcurrency = 4

  init(repository: String, revision: String, source: any CheckpointSource,
    expectedKind: ModelKind = .qwen3_8FlashNext)
  {
    self.repository = repository
    self.revision = revision
    self.source = source
    self.expectedKind = expectedKind
  }

  func manifest() async throws -> (data: Data, manifest: InstalledManifest) {
    let data = try await source.data(path: "manifest.json")
    let manifest = try InstalledModel.decodeManifest(data)
    guard manifest.modelKind == expectedKind else {
      throw RepackError.incompatibleModel("installed artifact does not match \(expectedKind.rawValue)")
    }
    return (data, manifest)
  }

  func install(
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)?
  ) async throws -> InstalledManifest {
    try await run(artifact: manifest(), output: output, trustedFiles: [], progress: progress)
  }

  func repair(
    at output: URL,
    invalidFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)?
  ) async throws -> InstalledManifest {
    let output = output.standardizedFileURL
    let partial = output.appendingPathExtension("partial")
    let current = try InstalledModel.loadManifest(at: output)
    let artifact = try await manifest()
    guard current == artifact.manifest else {
      throw RepackError.incompatibleModel("installed model does not match the published artifact")
    }
    let paths = Set(current.files.map(\.path))
    guard invalidFiles.isSubset(of: paths) else {
      throw RepackError.invalidPlan("repair contains an unknown installed file")
    }
    guard !FileManager.default.fileExists(atPath: partial.path) else {
      throw RepackError.invalidPlan("partial repair already exists: \(partial.path)")
    }
    try FileManager.default.moveItem(at: output, to: partial)
    for path in invalidFiles {
      let url = try safeFileURL(root: partial, path: path)
      if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
      }
    }
    return try await run(
      artifact: artifact,
      output: output,
      trustedFiles: paths.subtracting(invalidFiles),
      progress: progress)
  }

  private func run(
    artifact: (data: Data, manifest: InstalledManifest),
    output: URL,
    trustedFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)?
  ) async throws -> InstalledManifest {
    let fileManager = FileManager.default
    let output = output.standardizedFileURL
    let partial = output.appendingPathExtension("partial")
    let receiptURL = partial.appendingPathComponent("installed-artifact-receipt.json")
    guard !fileManager.fileExists(atPath: output.path) else {
      throw RepackError.destinationExists(output.path)
    }

    let totalBytes = try installedBytes(artifact.manifest)
    let partialExists = fileManager.fileExists(atPath: partial.path)
    let parent = output.deletingLastPathComponent()
    try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
    if !partialExists,
      let available = try parent.resourceValues(
        forKeys: [.volumeAvailableCapacityForImportantUsageKey]
      ).volumeAvailableCapacityForImportantUsage,
      available >= 0,
      UInt64(available) < totalBytes
    {
      throw RepackError.insufficientStorage(required: totalBytes, available: available)
    }
    if !partialExists {
      try fileManager.createDirectory(at: partial, withIntermediateDirectories: false)
    }
    try artifact.data.write(
      to: partial.appendingPathComponent("manifest.json"), options: .atomic)

    let manifestDigest = sha256(artifact.data)
    var receipt: InstalledArtifactReceipt
    if fileManager.fileExists(atPath: receiptURL.path) {
      receipt = try JSONDecoder().decode(
        InstalledArtifactReceipt.self, from: Data(contentsOf: receiptURL))
      guard receipt.formatVersion == 1,
        receipt.repository == repository,
        receipt.revision == revision,
        receipt.manifestSHA256 == manifestDigest
      else {
        throw RepackError.invalidPlan("partial install does not match the published artifact")
      }
    } else {
      receipt = InstalledArtifactReceipt(
        formatVersion: 1,
        repository: repository,
        revision: revision,
        manifestSHA256: manifestDigest,
        completed: [:])
      for file in artifact.manifest.files where trustedFiles.contains(file.path) {
        receipt.completed[file.path] = file.sha256
      }
    }

    let filesByPath = Dictionary(uniqueKeysWithValues: artifact.manifest.files.map { ($0.path, $0) })
    receipt.completed = receipt.completed.filter { path, digest in
      guard let file = filesByPath[path], digest == file.sha256,
        let url = try? safeFileURL(root: partial, path: path),
        (try? fileSize(url)) == file.size
      else { return false }
      return true
    }
    try persist(receipt, to: receiptURL)
    let completedBytes = receipt.completed.keys.reduce(UInt64(0)) {
      $0 + (filesByPath[$1]?.size ?? 0)
    }
    let state = InstalledArtifactState(
      receipt: receipt,
      receiptURL: receiptURL,
      copiedBytes: completedBytes,
      totalBytes: totalBytes,
      progress: progress)
    await state.report()

    let pending = artifact.manifest.files.filter { receipt.completed[$0.path] == nil }
    let fileDownloader = InstalledArtifactFileDownloader(source: source)
    try await withThrowingTaskGroup(of: Void.self) { group in
      var iterator = pending.makeIterator()
      for _ in 0..<min(fileConcurrency, pending.count) {
        guard let file = iterator.next() else { break }
        group.addTask {
          try await download(
            file, partial: partial, downloader: fileDownloader, state: state)
        }
      }
      while try await group.next() != nil {
        if let file = iterator.next() {
          group.addTask {
            try await download(
              file, partial: partial, downloader: fileDownloader, state: state)
          }
        }
      }
    }

    let verification = try InstalledModel.audit(at: partial)
    guard verification.isValid else {
      let paths = Set(verification.issues.map(\.path))
      for path in paths {
        let url = try safeFileURL(root: partial, path: path)
        if fileManager.fileExists(atPath: url.path) {
          try fileManager.removeItem(at: url)
        }
      }
      try await state.remove(paths)
      throw RepackError.invalidPlan("downloaded installed model failed complete verification")
    }

    try fileManager.removeItem(at: receiptURL)
    let oldReceipt = partial.appendingPathComponent("repack-receipt.json")
    if fileManager.fileExists(atPath: oldReceipt.path) {
      try fileManager.removeItem(at: oldReceipt)
    }
    try fileManager.moveItem(at: partial, to: output)
    return verification.manifest
  }

  private func download(
    _ file: InstalledFile,
    partial: URL,
    downloader: InstalledArtifactFileDownloader,
    state: InstalledArtifactState
  ) async throws {
    let destination = try safeFileURL(root: partial, path: file.path)
    let digest = try await downloader.run(file: file, to: destination) {
      copiedBytes, downloadedBytes in
      await state.advance(copiedBytes: copiedBytes, downloadedBytes: downloadedBytes)
    }
    try await state.complete(path: file.path, digest: digest)
  }

  private func installedBytes(_ manifest: InstalledManifest) throws -> UInt64 {
    try manifest.files.reduce(UInt64(0)) { total, file in
      let result = total.addingReportingOverflow(file.size)
      guard !result.overflow else {
        throw RepackError.invalidPlan("installed file sizes overflow")
      }
      return result.partialValue
    }
  }
}

private struct InstalledArtifactReceipt: Codable, Sendable {
  let formatVersion: Int
  let repository: String
  let revision: String
  let manifestSHA256: String
  var completed: [String: String]
}

private actor InstalledArtifactState {
  private var receipt: InstalledArtifactReceipt
  private let receiptURL: URL
  private var copiedBytes: UInt64
  private var downloadedBytes: UInt64 = 0
  private let totalBytes: UInt64
  private let progress: (@Sendable (RepackProgress) -> Void)?

  init(
    receipt: InstalledArtifactReceipt,
    receiptURL: URL,
    copiedBytes: UInt64,
    totalBytes: UInt64,
    progress: (@Sendable (RepackProgress) -> Void)?
  ) {
    self.receipt = receipt
    self.receiptURL = receiptURL
    self.copiedBytes = copiedBytes
    self.totalBytes = totalBytes
    self.progress = progress
  }

  func report() {
    progress?(
      RepackProgress(
        copiedBytes: copiedBytes,
        downloadedBytes: downloadedBytes,
        totalBytes: totalBytes))
  }

  func advance(copiedBytes: UInt64, downloadedBytes: UInt64) {
    self.copiedBytes += copiedBytes
    self.downloadedBytes += downloadedBytes
    report()
  }

  func complete(path: String, digest: String) throws {
    receipt.completed[path] = digest
    try persist(receipt, to: receiptURL)
  }

  func remove(_ paths: Set<String>) throws {
    for path in paths { receipt.completed.removeValue(forKey: path) }
    try persist(receipt, to: receiptURL)
  }
}

private func persist(_ receipt: InstalledArtifactReceipt, to url: URL) throws {
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.sortedKeys]
  try encoder.encode(receipt).write(to: url, options: .atomic)
}
