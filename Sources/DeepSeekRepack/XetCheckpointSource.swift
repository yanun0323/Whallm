import Foundation

/// Uses larger Xet transfers while Repacker keeps its 8 MiB receipt chunks.
struct XetCheckpointSource: CheckpointSource {
  let base: HuggingFaceSource
  let runtime: XetRuntime

  var downloadBatchSize: UInt64 { 512 * 1_024 * 1_024 }
  var downloadConcurrency: Int { 1 }

  func data(path: String) async throws -> Data {
    try await base.data(path: path)
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    guard path.hasSuffix(".safetensors"), range.count >= 64 * 1_024 * 1_024 else {
      return try await base.data(path: path, range: range)
    }
    return try await runtime.download(
      modelID: base.modelID, revision: base.revision, path: path, range: range)
  }
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
    if fileManager.isExecutableFile(atPath: bundledPython.path),
      fileManager.fileExists(atPath: bundledHelper.path)
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
    guard fileManager.isExecutableFile(atPath: python.path),
      fileManager.fileExists(atPath: helper.path)
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

  func download(modelID: String, revision: String, path: String, range: Range<UInt64>) async throws
    -> Data
  {
    let cancellation = XetProcessCancellation()
    return try await withTaskCancellationHandler {
      let data = try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .utility).async {
          do {
            continuation.resume(returning: try downloadSync(
              modelID: modelID, revision: revision, path: path, range: range,
              cancellation: cancellation))
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

  private func downloadSync(
    modelID: String, revision: String, path: String, range: Range<UInt64>,
    cancellation: XetProcessCancellation
  ) throws -> Data {
    let fileManager = FileManager.default
    let output = fileManager.temporaryDirectory.appending(path: "whallm-xet-\(UUID().uuidString)")
    defer { try? fileManager.removeItem(at: output) }

    let process = Process()
    process.executableURL = python
    process.arguments = [helper.path, modelID, revision, path,
      String(range.lowerBound), String(range.upperBound), output.path]
    process.standardOutput = FileHandle.nullDevice
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
    process.waitUntilExit()
    if cancellation.isCancelled { throw CancellationError() }
    let diagnostic = String(data: errors.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard process.terminationStatus == 0 else {
      throw RepackError.badResponse("Xet range download failed: \(diagnostic)")
    }
    let data = try Data(contentsOf: output, options: .mappedIfSafe)
    guard UInt64(data.count) == range.count else {
      throw RepackError.badResponse("Xet returned an incomplete checkpoint range")
    }
    return data
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
