import Foundation

protocol CheckpointSource: Sendable {
  var downloadBatchSize: UInt64 { get }
  var downloadConcurrency: Int { get }
  func data(path: String) async throws -> Data
  func data(path: String, range: Range<UInt64>) async throws -> Data
  func data(path: String, range: Range<UInt64>,
    onProgress: @escaping @Sendable (UInt64) -> Void) async throws -> Data
}

/// Sources that stage one complete shard so later range reads are local.
protocol StagedCheckpointSource: CheckpointSource {
  func stage(path: String, onProgress: @escaping @Sendable (UInt64) -> Void) async throws -> UInt64
  func release(path: String) async
  func releaseAll() async
}

extension CheckpointSource {
  var downloadBatchSize: UInt64 { 8 * 1_024 * 1_024 }
  var downloadConcurrency: Int { 8 }

  func data(path: String, range: Range<UInt64>,
    onProgress: @escaping @Sendable (UInt64) -> Void) async throws -> Data
  {
    let result = try await data(path: path, range: range)
    onProgress(UInt64(result.count))
    return result
  }
}

struct HuggingFaceSource: CheckpointSource {
  let modelID: String
  let revision: String
  let session: URLSession

  init(modelID: String, revision: String, session: URLSession = .shared) {
    self.modelID = modelID
    self.revision = revision
    self.session = session
  }

  func data(path: String) async throws -> Data {
    var lastError = "unknown response"
    for attempt in 0..<10 {
      do {
        let (data, response) = try await session.data(for: request(path: path))
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
          let status = (response as? HTTPURLResponse)?.statusCode ?? 0
          lastError = "HTTP \(status)"
          if attempt < 9 {
            try await retryDelay(status: status, attempt: attempt)
          }
          continue
        }
        return data
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        lastError = String(describing: error)
        if attempt < 9 {
          try await retryDelay(status: 0, attempt: attempt)
        }
      }
    }
    throw RepackError.badResponse(
      "checkpoint request failed for \(path) after 10 attempts: \(lastError)")
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    guard !range.isEmpty else { return Data() }
    var lastError = "unknown response"
    for attempt in 0..<10 {
      do {
        var request = request(path: path)
        request.setValue(
          "bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 206 else {
          lastError = "HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)"
          if attempt < 9 {
            try await retryDelay(status: (response as? HTTPURLResponse)?.statusCode ?? 0,
              attempt: attempt)
          }
          continue
        }
        guard UInt64(data.count) == range.count else {
          lastError = "returned \(data.count) bytes; expected \(range.count)"
          if attempt < 9 {
            try await retryDelay(status: 0, attempt: attempt)
          }
          continue
        }
        return data
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        lastError = String(describing: error)
        if attempt < 9 {
          try await retryDelay(status: 0, attempt: attempt)
        }
      }
    }
    throw RepackError.badResponse(
      "checkpoint range request failed for \(path) after 10 attempts: \(lastError)")
  }

  private func retryDelay(status: Int, attempt: Int) async throws {
    let nanoseconds: UInt64
    if status == 429 {
      nanoseconds = UInt64(min(60, 5 << min(attempt, 4))) * 1_000_000_000
    } else {
      nanoseconds = UInt64(250_000_000 << min(attempt, 4))
    }
    try await Task.sleep(nanoseconds: nanoseconds)
  }

  private func request(path: String) -> URLRequest {
    let encodedPath = path.split(separator: "/").map(String.init).map {
      $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? $0
    }.joined(separator: "/")
    let url = URL(string: "https://huggingface.co/\(modelID)/resolve/\(revision)/\(encodedPath)")!
    var request = URLRequest(url: url)
    request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
    request.setValue("Whallm/1", forHTTPHeaderField: "User-Agent")
    return request
  }
}

public struct DeepSeekV4Checkpoint: Sendable {
  private let source: any CheckpointSource

  public init() {
    source = HuggingFaceSource(modelID: ModelContract.modelID, revision: ModelContract.revision)
  }

  init(source: any CheckpointSource) {
    self.source = source
  }

  public func makeRepackPlan(includeDSpark: Bool = true) async throws -> RepackPlan {
    let configData = try await source.data(path: "config.json")
    let config: ModelConfig
    do {
      config = try JSONDecoder().decode(ModelConfig.self, from: configData)
    } catch {
      throw RepackError.incompatibleModel("cannot decode model config: \(error)")
    }
    try ModelContract.validate(config)

    if includeDSpark {
      let dsparkData = try await source.data(path: "inference/config.json")
      let dsparkConfig: DSparkConfig
      do {
        dsparkConfig = try JSONDecoder().decode(DSparkConfig.self, from: dsparkData)
      } catch {
        throw RepackError.incompatibleModel("cannot decode DSpark config: \(error)")
      }
      try ModelContract.validate(dsparkConfig)
    }

    let indexData = try await source.data(path: "model.safetensors.index.json")
    let index = try CheckpointIndex.decode(indexData)
    let tensors = try await readTensors(index: index)
    return try RepackPlanner.makePlan(
      index: index,
      tensors: tensors,
      includeDSpark: includeDSpark
    )
  }

  public func repack(
    to output: URL,
    includeDSpark: Bool = true,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let plan = try await makeRepackPlan(includeDSpark: includeDSpark)
    return try await repack(plan: plan, to: output, progress: progress)
  }

  public func repack(
    plan: RepackPlan,
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let dsparkMatches =
      plan.dspark.map { descriptor in
        descriptor.layerCount == ModelContract.dsparkLayerCount
          && descriptor.blockSize == ModelContract.dsparkBlockSize
          && descriptor.noiseTokenID == ModelContract.dsparkNoiseTokenID
          && descriptor.targetLayerIDs == ModelContract.dsparkTargetLayerIDs
          && descriptor.markovRank == ModelContract.dsparkMarkovRank
          && !descriptor.commonTensors.isEmpty
      } ?? true
    guard plan.formatVersion == 1,
      plan.modelID == ModelContract.modelID,
      plan.revision == ModelContract.revision,
      plan.layerCount == ModelContract.layerCount,
      plan.expertCount == ModelContract.expertCount,
      plan.selectedExpertCount == ModelContract.selectedExpertCount,
      plan.expertBlobSize == ModelContract.expertBlobSize,
      plan.expertRegions == ModelContract.expertRegions,
      dsparkMatches
    else {
      throw RepackError.incompatibleModel("repack plan does not match the pinned model contract")
    }
    return try await Repacker(source: source).run(plan: plan, output: output, progress: progress)
  }

  public func repair(
    at output: URL,
    invalidFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let manifest = try InstalledModel.loadManifest(at: output)
    let plan = try await makeRepackPlan(includeDSpark: manifest.dspark != nil)
    return try await Repacker(source: source).repair(
      plan: plan,
      output: output,
      invalidFiles: invalidFiles,
      progress: progress
    )
  }

  public func installDSpark(
    at output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let manifest = try InstalledModel.loadManifest(at: output)
    try Self.validateDSparkInstallTarget(manifest)
    if manifest.dspark != nil {
      return try InstalledModel.verify(at: output)
    }
    let plan = try await makeRepackPlan(includeDSpark: true)
    let dsparkFiles = Set(
      plan.files.lazy.map(\.path).filter { $0.hasPrefix("dspark/") }
        + ["inference/config.json"]
    )
    return try await Repacker(source: source).repair(
      plan: plan,
      output: output,
      invalidFiles: dsparkFiles,
      progress: progress
    )
  }

  static func validateDSparkInstallTarget(_ manifest: InstalledManifest) throws {
    guard (manifest.modelKind ?? .deepSeekV4) == .deepSeekV4 else {
      throw RepackError.incompatibleModel("installed model is not DeepSeek-V4-Flash-0731")
    }
  }

  private func readTensors(index: CheckpointIndex) async throws -> [String: SafeTensor] {
    var headers: [String: (base: UInt64, header: SafeTensorsHeader)] = [:]
    // ponytail: Header reads are serial. Add bounded parallel reads only if inspect latency matters.
    for shard in Set(index.weightMap.values).sorted() {
      let prefix = try await source.data(path: shard, range: 0..<8)
      let headerLength = try littleEndianUInt64(prefix)
      guard headerLength > 1, headerLength <= 64 * 1_024 * 1_024 else {
        throw RepackError.invalidSafeTensors("invalid header length \(headerLength) in \(shard)")
      }
      let data = try await source.data(path: shard, range: 8..<(8 + headerLength))
      headers[shard] = (8 + headerLength, try SafeTensorsHeader.decode(data))
    }

    var tensors: [String: SafeTensor] = [:]
    tensors.reserveCapacity(index.weightMap.count)
    for (name, shard) in index.weightMap {
      guard let sourceHeader = headers[shard], let entry = sourceHeader.header.entries[name] else {
        throw RepackError.invalidIndex("index tensor \(name) is missing from \(shard)")
      }
      tensors[name] = SafeTensor(
        name: name,
        sourceFile: shard,
        dtype: entry.dtype,
        shape: entry.shape,
        sourceOffset: sourceHeader.base + entry.dataStart,
        length: entry.dataEnd - entry.dataStart
      )
    }
    return tensors
  }
}
