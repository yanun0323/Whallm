import Foundation

/// Uses larger Xet transfers while Repacker keeps its 8 MiB receipt chunks.
struct XetCheckpointSource: StagedCheckpointSource {
  let base: HuggingFaceSource
  let runtime: XetRuntime
  private let stagedFiles = XetStagedFiles()

  var downloadBatchSize: UInt64 { 512 * 1_024 * 1_024 }
  var downloadConcurrency: Int { 3 }

  func data(path: String) async throws -> Data {
    try await base.data(path: path)
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    try await data(path: path, range: range, onProgress: { _ in })
  }

  func data(path: String, range: Range<UInt64>,
    onProgress: @escaping @Sendable (UInt64) -> Void) async throws -> Data
  {
    if let staged = await stagedFiles.url(for: path) {
      let handle = try FileHandle(forReadingFrom: staged)
      defer { try? handle.close() }
      try handle.seek(toOffset: range.lowerBound)
      guard let data = try handle.read(upToCount: range.count), data.count == range.count else {
        throw RepackError.badResponse("incomplete staged checkpoint range for \(path)")
      }
      return data
    }
    guard path.hasSuffix(".safetensors"), range.count >= 64 * 1_024 * 1_024 else {
      return try await base.data(path: path, range: range, onProgress: onProgress)
    }
    return try await runtime.download(
      modelID: base.modelID, revision: base.revision, path: path, range: range,
      onProgress: onProgress)
  }

  func stage(path: String, onProgress: @escaping @Sendable (UInt64) -> Void) async throws -> UInt64 {
    var lastError: Error?
    for attempt in 0..<3 {
      do {
        let file = try await runtime.stage(
          modelID: base.modelID, revision: base.revision, path: path,
          onProgress: onProgress)
        await stagedFiles.store(file.url, for: path)
        return file.size
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        lastError = error
        if attempt < 2 { try await Task.sleep(for: .milliseconds(500 * (attempt + 1))) }
      }
    }
    throw lastError ?? RepackError.badResponse("Xet checkpoint staging failed")
  }

  func release(path: String) async { await stagedFiles.remove(path) }
  func releaseAll() async { await stagedFiles.removeAll() }
}

private actor XetStagedFiles {
  private var files: [String: URL] = [:]

  func url(for path: String) -> URL? { files[path] }
  func store(_ url: URL, for path: String) { files[path] = url }
  func remove(_ path: String) {
    if let url = files.removeValue(forKey: path) { try? FileManager.default.removeItem(at: url) }
  }
  func removeAll() {
    for path in Array(files.keys) { remove(path) }
  }
}

struct StagedXetFile: Sendable {
  let url: URL
  let size: UInt64
}

struct XetRuntime: Sendable {
  let python: URL
  let helper: URL
  let runtimeDirectory: URL
  let pythonHome: URL?
  let sitePackages: URL?

  static func current() -> XetRuntime? {
    let fileManager = FileManager.default
    let bundle = Bundle.main.bundleURL
    let bundledPython = bundle.appending(path: "Contents/MacOS/python3")
    let resources = bundle.appending(path: "Contents/Resources")
    let bundledRuntime = resources.appending(path: "runtime")
    let bundledHelper = bundledRuntime.appending(path: "deepseek_v4_ssd/model_support/xet_range.py")
    let bundledFileHelper = bundledRuntime.appending(path: "deepseek_v4_ssd/model_support/xet_file.py")
    if fileManager.isExecutableFile(atPath: bundledPython.path),
      fileManager.fileExists(atPath: bundledHelper.path),
      fileManager.fileExists(atPath: bundledFileHelper.path)
    {
      return XetRuntime(
        python: bundledPython,
        helper: bundledHelper,
        runtimeDirectory: bundledRuntime,
        pythonHome: bundle.appending(path: "Contents/Frameworks/Python.framework/Versions/Current"),
        sitePackages: resources.appending(path: "python/site-packages"))
    }

    let project = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    let python = project.appending(path: ".venv/bin/python")
    let runtime = project.appending(path: "runtime")
    let helper = runtime.appending(path: "deepseek_v4_ssd/model_support/xet_range.py")
    let fileHelper = runtime.appending(path: "deepseek_v4_ssd/model_support/xet_file.py")
    guard fileManager.isExecutableFile(atPath: python.path),
      fileManager.fileExists(atPath: helper.path),
      fileManager.fileExists(atPath: fileHelper.path)
    else { return nil }
    return XetRuntime(
      python: python, helper: helper, runtimeDirectory: runtime,
      pythonHome: nil, sitePackages: nil)
  }

  static func hasReadToken() -> Bool {
    let environment = ProcessInfo.processInfo.environment
    if let token = environment["HF_TOKEN"], !token.isEmpty { return true }
    let home = FileManager.default.homeDirectoryForCurrentUser
    let cache = environment["XDG_CACHE_HOME"].map { URL(fileURLWithPath: $0) }
      ?? home.appending(path: ".cache")
    let hfHome = environment["HF_HOME"].map { URL(fileURLWithPath: $0) }
      ?? cache.appending(path: "huggingface")
    let tokenURL = environment["HF_TOKEN_PATH"].map { URL(fileURLWithPath: $0) }
      ?? hfHome.appending(path: "token")
    return FileManager.default.fileExists(atPath: tokenURL.path)
  }

  func download(modelID: String, revision: String, path: String, range: Range<UInt64>,
    onProgress: @escaping @Sendable (UInt64) -> Void = { _ in }) async throws -> Data
  {
    let cancellation = XetProcessCancellation()
    return try await withTaskCancellationHandler {
      let data = try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .utility).async {
          do {
            continuation.resume(returning: try downloadSync(
              modelID: modelID, revision: revision, path: path, range: range,
              cancellation: cancellation, onProgress: onProgress))
          } catch {
            continuation.resume(throwing: error)
          }
        }
      }
      try Task.checkCancellation()
      return data
    } onCancel: {
      cancellation.cancel()
    }
  }

  func stage(modelID: String, revision: String, path: String,
    onProgress: @escaping @Sendable (UInt64) -> Void) async throws -> StagedXetFile
  {
    let cancellation = XetProcessCancellation()
    return try await withTaskCancellationHandler {
      let file: StagedXetFile = try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .utility).async {
          do {
            continuation.resume(returning: try stageSync(
              modelID: modelID, revision: revision, path: path,
              cancellation: cancellation, onProgress: onProgress))
          } catch {
            continuation.resume(throwing: error)
          }
        }
      }
      do {
        try Task.checkCancellation()
        return file
      } catch {
        try? FileManager.default.removeItem(at: file.url)
        throw error
      }
    } onCancel: {
      cancellation.cancel()
    }
  }

  private func downloadSync(
    modelID: String, revision: String, path: String, range: Range<UInt64>,
    cancellation: XetProcessCancellation,
    onProgress: @escaping @Sendable (UInt64) -> Void
  ) throws -> Data {
    let fileManager = FileManager.default
    let output = fileManager.temporaryDirectory.appending(path: "whallm-xet-\(UUID().uuidString)")
    defer { try? fileManager.removeItem(at: output) }
    try runHelper(
      helper: helper,
      arguments: [modelID, revision, path, String(range.lowerBound), String(range.upperBound)],
      output: output, cancellation: cancellation, progressLimit: UInt64(range.count),
      onProgress: onProgress)
    let data = try Data(contentsOf: output, options: .mappedIfSafe)
    guard UInt64(data.count) == range.count else {
      throw RepackError.badResponse("Xet returned an incomplete checkpoint range")
    }
    return data
  }

  private func stageSync(
    modelID: String, revision: String, path: String,
    cancellation: XetProcessCancellation,
    onProgress: @escaping @Sendable (UInt64) -> Void
  ) throws -> StagedXetFile {
    let output = FileManager.default.temporaryDirectory.appending(
      path: "whallm-xet-shard-\(UUID().uuidString)")
    do {
      try runHelper(
        helper: helper.deletingLastPathComponent().appending(path: "xet_file.py"),
        arguments: [modelID, revision, path], output: output,
        cancellation: cancellation, progressLimit: nil, onProgress: onProgress)
      return StagedXetFile(url: output, size: try fileSize(output))
    } catch {
      try? FileManager.default.removeItem(at: output)
      throw error
    }
  }

  private func runHelper(
    helper: URL, arguments: [String], output: URL,
    cancellation: XetProcessCancellation, progressLimit: UInt64?,
    onProgress: @escaping @Sendable (UInt64) -> Void
  ) throws {
    let process = Process()
    process.executableURL = python
    process.arguments = [helper.path] + arguments + [output.path]
    let updates = Pipe()
    process.standardOutput = updates
    let errors = Pipe()
    process.standardError = errors
    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    if ProcessInfo.processInfo.physicalMemory >= 64 * 1_024 * 1_024 * 1_024 {
      environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    }
    environment["PYTHONPATH"] = [runtimeDirectory.path, sitePackages?.path]
      .compactMap { $0 }.joined(separator: ":")
    if let pythonHome { environment["PYTHONHOME"] = pythonHome.path }
    process.environment = environment
    try cancellation.register(process)
    try process.run()
    if cancellation.isCancelled, process.isRunning { process.terminate() }
    var progressBuffer = Data()
    while true {
      let bytes = updates.fileHandleForReading.availableData
      if bytes.isEmpty { break }
      progressBuffer.append(bytes)
      while let newline = progressBuffer.firstIndex(of: 10) {
        if let count = UInt64(String(decoding: progressBuffer[..<newline], as: UTF8.self)) {
          onProgress(min(count, progressLimit ?? count))
        }
        progressBuffer.removeSubrange(...newline)
      }
    }
    process.waitUntilExit()
    if cancellation.isCancelled { throw CancellationError() }
    let diagnostic = String(data: errors.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard process.terminationStatus == 0 else {
      throw RepackError.badResponse("Xet download failed: \(diagnostic)")
    }
  }
}

private final class XetProcessCancellation: @unchecked Sendable {
  private let lock = NSLock()
  private var process: Process?
  private var cancelled = false

  func register(_ process: Process) throws {
    let alreadyCancelled = lock.withLock {
      self.process = process
      return cancelled
    }
    if alreadyCancelled { throw CancellationError() }
  }

  var isCancelled: Bool { lock.withLock { cancelled } }

  func cancel() {
    let process = lock.withLock {
      cancelled = true
      return self.process
    }
    if let process, process.isRunning { process.terminate() }
  }
}
