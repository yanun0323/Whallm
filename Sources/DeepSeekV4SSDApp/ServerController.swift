import Darwin
import DeepSeekRepack
import Foundation
import Security

struct ServerStatus: Decodable {
  struct AppMemory: Decodable, Equatable {
    let currentAppMemoryBytes: Double?
    let peakAppMemoryBytes: Double?
    let memoryScope: String
    let sampleIntervalSeconds: Double
    let activeRequests: Int
    let epoch: String
  }

  struct Runtime: Decodable {}

  struct Performance: Decodable {
    struct ActiveParametersCache: Decodable {
      let hitRate: Double
      let hits: Int
      let misses: Int
      let residentSlots: Int
      let capacitySlots: Int
    }

    let generating: Bool
    let runtimePromptTokens: Int
    let runtimeGenerationTokens: Int
    let accumulatedGenerationTokens: Int
    let completedRequestCount: Int
    let requestSeconds: Double
    let timeToFirstTokenSeconds: Double
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let requestSsdReadBytesPerSecond: Double
    let requestExpertCacheHitRate: Double
    let dsparkEnabled: Bool?
    let dsparkAcceptanceRate: Double?
    let dsparkAverageAcceptedLength: Double?
    let ssdBytesRead: UInt64
    let activeParametersCache: ActiveParametersCache
  }

  let model: String?
  let sourceModel: String?
  let modelPath: String?
  let runtime: Runtime?
  let loadedModel: String?
  let loadingModel: String?
  let performance: Performance
  let appMemory: AppMemory?

  static func decode(_ data: Data) throws -> ServerStatus {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(ServerStatus.self, from: data)
  }
}

enum PerformanceMetric: String, CaseIterable, Identifiable {
  case prefillTokensPerSecond
  case decodeTokensPerSecond
  case inputTokens
  case outputTokens
  case memoryUsage
  case ssdReadSpeed
  case cacheHitRate
  case firstTokenWaitTime
  case completionTime

  var id: String { rawValue }
}

struct PerformanceSnapshot: Equatable {
  var prefillTokensPerSecond = 0.0
  var decodeTokensPerSecond = 0.0
  var inputTokens = 0.0
  var outputTokens = 0.0
  var memoryUsage = 0.0
  var ssdReadSpeed = 0.0
  var cacheHitRate = 0.0
  var firstTokenWaitTime = 0.0
  var completionTime = 0.0

  subscript(metric: PerformanceMetric) -> Double {
    switch metric {
    case .prefillTokensPerSecond: prefillTokensPerSecond
    case .decodeTokensPerSecond: decodeTokensPerSecond
    case .inputTokens: inputTokens
    case .outputTokens: outputTokens
    case .memoryUsage: memoryUsage
    case .ssdReadSpeed: ssdReadSpeed
    case .cacheHitRate: cacheHitRate
    case .firstTokenWaitTime: firstTokenWaitTime
    case .completionTime: completionTime
    }
  }
}

struct MetricStatistics: Equatable {
  private(set) var count = 0
  private(set) var minimum = 0.0
  private(set) var total = 0.0
  private(set) var maximum = 0.0
  private var sortedValues: [Double] = []

  var average: Double { count == 0 ? 0 : total / Double(count) }
  var p95: Double {
    guard count > 0 else { return 0 }
    return sortedValues[Int(ceil(Double(count) * 0.95)) - 1]
  }

  mutating func record(_ value: Double) {
    guard value.isFinite else { return }
    if count == 0 {
      minimum = value
      maximum = value
    } else {
      minimum = Swift.min(minimum, value)
      maximum = Swift.max(maximum, value)
    }
    count += 1
    total += value
    var lowerBound = 0
    var upperBound = sortedValues.count
    while lowerBound < upperBound {
      let middle = lowerBound + (upperBound - lowerBound) / 2
      if sortedValues[middle] < value {
        lowerBound = middle + 1
      } else {
        upperBound = middle
      }
    }
    sortedValues.insert(value, at: lowerBound)
  }
}

struct PerformanceHistory: Equatable {
  private(set) var values: [PerformanceMetric: MetricStatistics] = [:]

  var isEmpty: Bool { values.isEmpty }

  subscript(metric: PerformanceMetric) -> MetricStatistics? { values[metric] }

  mutating func record(
    _ snapshot: PerformanceSnapshot,
    excluding excludedMetric: PerformanceMetric? = nil
  ) {
    for metric in PerformanceMetric.allCases where metric != excludedMetric {
      values[metric, default: MetricStatistics()].record(snapshot[metric])
    }
  }

  mutating func clear() {
    values.removeAll()
  }
}

struct LivePerformance: Equatable {
  var hasStatus = false
  var generating = false
  var completedRequestCount = 0
  var accumulatedOutputTokens = 0
  var snapshot = PerformanceSnapshot()
  var dsparkEnabled = false
  var dsparkAcceptanceRate = 0.0
  var dsparkAverageAcceptedLength = 0.0
  var loadedModel: String?
  var loadingModel: String?
  var appMemory: ServerStatus.AppMemory?
  var memoryResetFailed = false

  var memoryMaximum: Double? {
    memoryResetFailed ? nil : appMemory?.peakAppMemoryBytes
  }

  var liveFirstTokenWaitTime: Double {
    if generating && snapshot.outputTokens == 0 {
      return snapshot.completionTime
    }
    return snapshot.firstTokenWaitTime
  }
}

private struct RuntimeEnvironment {
  let runtimeDirectory: URL
  let pythonExecutable: URL
  let pythonHome: URL?
  let sitePackages: URL?

  static var current: RuntimeEnvironment {
    let bundle = Bundle.main.bundleURL
    let bundledPython = bundle.appending(path: "Contents/MacOS/python3")
    let resources = bundle.appending(path: "Contents/Resources")
    if FileManager.default.isExecutableFile(atPath: bundledPython.path) {
      return RuntimeEnvironment(
        runtimeDirectory: resources.appending(path: "runtime", directoryHint: .isDirectory),
        pythonExecutable: bundledPython,
        pythonHome: bundle.appending(
          path: "Contents/Frameworks/Python.framework/Versions/Current",
          directoryHint: .isDirectory
        ),
        sitePackages: resources.appending(
          path: "python/site-packages",
          directoryHint: .isDirectory
        )
      )
    }

    let project = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    return RuntimeEnvironment(
      runtimeDirectory: project.appending(path: "runtime", directoryHint: .isDirectory),
      pythonExecutable: project.appending(path: ".venv/bin/python"),
      pythonHome: nil,
      sitePackages: nil
    )
  }
}

enum PromptCacheMode: String, Codable, CaseIterable, Identifiable {
  case off, memory, disk

  var id: String { rawValue }
  var localizationKey: String {
    switch self {
    case .off: "Off"
    case .memory: "Memory"
    case .disk: "Disk"
    }
  }
}

struct ModelAdvancedSettings: Codable, Equatable, Sendable {
  private static let legacyModelPreference = "modelAdvancedSettingsLegacyModelKind"

  var slots = 1_152 // Legacy capacity, used until a memory budget is saved.
  var expertCacheGiB: Double?
  var mtpCacheGiB: Double?
  var dsparkCacheGiB: Double?
  var estimateInputTokens: Int?
  var readWorkers = 4
  var prefetchReadWorkers: Int? = 2
  var moePrefillStepSize: Int? = 0
  var readyExpertDecode: Bool? = true
  var batchedExpertPrefill: Bool? = true
  var nextLayerPrefetch: Bool? = true
  var packedKVCache: Bool? = false
  var packedIndexCache: Bool? = false
  var candidateIndex: Bool? = false
  var cedPrefill: Bool? = false
  var deepSeekANEPrefill: Bool? = false
  var approximationEnabled: Bool? = false
  var qwenAdaptiveSampling: Bool? = true
  var memoryLimitGiB = 0
  var prefillStepSize = 0
  var layerMajorPrefill = true
  var layerMajorPrefillThreshold: Int? = 1_024
  var promptCacheMode: PromptCacheMode?
  var promptCacheEntries = 2
  var promptCacheMemoryGiB = 8
  var warmupPromptPath = ""
  var bf16KVCache = false
  var anePrefillRatio: Double? = 0
  var qwenGroupedExperts: Bool?
  var qwenPooledIndexCache: Bool? = false
  var qwenNgramLookupOptimized: Bool? = false
  var qwenCompileTensorOps: Bool? = false
  var qwenPhaseMemory: Bool? = false
  var qwenExpertWaveSlots: Int? = 0
  var qwenNgramIO: String? = "mmap"
  var qwenNgramCacheMiB: Int? = 0
  var qwenSparseSDPA: Bool? = false
  var qwenQSAQueryChunk: Int? = 4
  var qwenQSAIndexed: Bool? = false
  var qwenPrefillReadExperts: Int? = 1
  var qwenPrefillSeedExperts: Int? = 0
  var qwenSharedExpertOverlap: Bool? = false
  var qwenMTPPolicy: Bool? = false
  var qwenMTPDraftTokens: Int? = 2
  var qwenMTPZeroAcceptanceLimit: Int? = 2
  var recentExpertCache: Bool?
  var routeAwareExpertCache: Bool?
  var mtpEnabled: Bool? = false
  var mtpSlots: Int? = 32
  var dsparkEnabled = false
  var dsparkSlots = 768
  var dsparkConfidenceThreshold = 0.6
  var defaultMaxTokens = 8_192
  var defaultTemperature = 0.2
  var defaultTopP = 0.98
  var defaultTopK = 0

  static func defaults(for modelKind: ModelKind) -> ModelAdvancedSettings {
    var settings = ModelAdvancedSettings()
    let descriptor = modelKind.descriptor
    settings.recentExpertCache = true
    settings.promptCacheMode = descriptor.supports("promptCache") ? .memory : .off
    settings.qwenGroupedExperts = descriptor.supports("groupedExperts")
    settings.slots = descriptor.defaults.slots
    let blob = ExpertMemory.blobBytes(for: modelKind)
    if blob > 0 {
      settings.expertCacheGiB = ExpertMemory.defaultGiB(slots: settings.slots, blobBytes: blob)
      if descriptor.supports("mtp") {
        settings.mtpCacheGiB = ExpertMemory.defaultGiB(slots: 32, blobBytes: blob)
      }
      if descriptor.supports("dspark") {
        settings.dsparkCacheGiB = ExpertMemory.defaultGiB(slots: 768, blobBytes: blob)
      }
    }
    settings.layerMajorPrefill = descriptor.supports("layerMajorPrefill")
    settings.readyExpertDecode = descriptor.supports("readyExpertDecode")
    settings.batchedExpertPrefill = descriptor.supports("batchedExpertPrefill")
    settings.nextLayerPrefetch = descriptor.supports("nextLayerPrefetch")
    settings.promptCacheEntries = descriptor.defaults.promptCacheEntries
    settings.bf16KVCache = descriptor.defaults.bf16KVCache
    settings.defaultMaxTokens = descriptor.defaults.maxTokens
    settings.defaultTemperature = descriptor.defaults.temperature
    settings.defaultTopP = descriptor.defaults.topP
    settings.defaultTopK = descriptor.defaults.topK
    if modelKind == .qwen3_8FlashNext {
      // Adopt the 4K input / 1024 output speed profile; keep cache budgets unchanged.
      settings.readWorkers = 16
      settings.prefillStepSize = 1_024
      settings.memoryLimitGiB = 30
      settings.qwenPooledIndexCache = true
      settings.qwenNgramLookupOptimized = true
      settings.qwenCompileTensorOps = true
      settings.qwenPhaseMemory = true
    }
    return settings
  }

  private init() {}

  func normalized(for modelKind: ModelKind) -> ModelAdvancedSettings {
    var settings = self
    let descriptor = modelKind.descriptor
    settings.promptCacheMode = descriptor.supports("promptCache")
      ? (settings.promptCacheMode ?? .memory) : .off
    settings.readyExpertDecode = descriptor.supports("readyExpertDecode")
      && (settings.readyExpertDecode ?? true)
    settings.batchedExpertPrefill = descriptor.supports("batchedExpertPrefill")
      && (settings.batchedExpertPrefill ?? true)
    settings.nextLayerPrefetch = descriptor.supports("nextLayerPrefetch")
      && (settings.nextLayerPrefetch ?? true)
    settings.packedKVCache = settings.packedKVCache ?? false
    settings.packedIndexCache = settings.packedIndexCache ?? false
    settings.prefetchReadWorkers = settings.prefetchReadWorkers ?? 2
    settings.moePrefillStepSize = settings.moePrefillStepSize ?? 0
    settings.approximationEnabled = descriptor.supports("approximation")
      && (settings.approximationEnabled ?? false)
    settings.qwenAdaptiveSampling = settings.qwenAdaptiveSampling ?? true
    settings.recentExpertCache = settings.recentExpertCache ?? true
    settings.layerMajorPrefillThreshold = settings.layerMajorPrefillThreshold ?? 1_024
    settings.anePrefillRatio = settings.anePrefillRatio ?? 0
    settings.layerMajorPrefill = descriptor.supports("layerMajorPrefill") && settings.layerMajorPrefill
    if !descriptor.editableSettings.contains("kvCachePrecision") {
      settings.bf16KVCache = descriptor.defaults.bf16KVCache
    }
    if !descriptor.supports("promptCache") {
      settings.promptCacheEntries = descriptor.defaults.promptCacheEntries
    }
    settings.qwenGroupedExperts = descriptor.supports("groupedExperts")
      ? (settings.qwenGroupedExperts ?? true) : false
    settings.mtpEnabled = descriptor.supports("mtp") ? (settings.mtpEnabled ?? false) : false
    settings.mtpSlots = settings.mtpSlots ?? 32
    let qwen = modelKind == .qwen3_8FlashNext
    settings.qwenExpertWaveSlots = qwen ? (settings.qwenExpertWaveSlots ?? 0) : 0
    settings.qwenNgramIO = qwen ? (settings.qwenNgramIO ?? "mmap") : "mmap"
    settings.qwenNgramCacheMiB = qwen ? (settings.qwenNgramCacheMiB ?? 0) : 0
    settings.qwenSparseSDPA = qwen ? (settings.qwenSparseSDPA ?? false) : false
    settings.qwenQSAQueryChunk = qwen ? (settings.qwenQSAQueryChunk ?? 4) : 4
    settings.qwenQSAIndexed = qwen ? (settings.qwenQSAIndexed ?? false) : false
    settings.qwenPrefillReadExperts = qwen ? qwenPrefillReadExperts ?? 1 : 1
    settings.qwenPrefillSeedExperts = qwen ? qwenPrefillSeedExperts ?? 0 : 0
    settings.qwenSharedExpertOverlap = qwen ? qwenSharedExpertOverlap ?? false : false
    settings.dsparkEnabled = descriptor.supports("dspark") && settings.dsparkEnabled
    return settings
  }

  static func load(
    for modelKind: ModelKind,
    defaults: UserDefaults
  ) -> ModelAdvancedSettings? {
    guard let data = defaults.data(forKey: preferenceKey(for: modelKind)) else { return nil }
    return try? JSONDecoder().decode(ModelAdvancedSettings.self, from: data)
  }

  static func loadOrDefault(
    for modelKind: ModelKind,
    defaults: UserDefaults = .standard
  ) -> ModelAdvancedSettings {
    let settings = load(for: modelKind, defaults: defaults)
      ?? legacySettings(for: modelKind, defaults: defaults)
      ?? ModelAdvancedSettings.defaults(for: modelKind)
    let normalized = settings.normalized(for: modelKind)
    normalized.save(for: modelKind, defaults: defaults)
    return normalized
  }

  func save(for modelKind: ModelKind, defaults: UserDefaults = .standard) {
    guard let data = try? JSONEncoder().encode(normalized(for: modelKind)) else { return }
    defaults.set(data, forKey: Self.preferenceKey(for: modelKind))
  }

  var effectiveQwenMTPDraftTokens: Int { qwenMTPPolicy == true ? (qwenMTPDraftTokens ?? 2) : 5 }
  var effectiveQwenMTPZeroAcceptanceLimit: Int { qwenMTPPolicy == true ? (qwenMTPZeroAcceptanceLimit ?? 2) : 1 }

  func validate(for modelKind: ModelKind) throws {
    if modelKind == .qwen3_8FlashNext { try validateQwenFlashSettings() }
    if modelKind == .qwen3_8FlashNext && qwenMTPPolicy == true {
      guard (1...5).contains(effectiveQwenMTPDraftTokens),
        (1...32).contains(effectiveQwenMTPZeroAcceptanceLimit) else {
        throw ConfigurationError(L10n.string("Choose 1–5 MTP draft tokens and 1–32 zero-acceptance rounds."))
      }
    }
    for value in [expertCacheGiB, mtpCacheGiB, dsparkCacheGiB].compactMap({ $0 }) {
      _ = try ExpertMemory.bytes(gib: value)
    }
    let blob = ExpertMemory.blobBytes(for: modelKind)
    if let expertCacheGiB {
      _ = try ExpertMemory.capacity(gib: expertCacheGiB, blobBytes: blob,
        minimum: modelKind == .qwen3_8FlashNext ? 10 : 6)
    }
    if mtpEnabled == true, let mtpCacheGiB {
      _ = try ExpertMemory.capacity(gib: mtpCacheGiB, blobBytes: blob, minimum: 10)
    }
    if dsparkEnabled, let dsparkCacheGiB {
      _ = try ExpertMemory.capacity(gib: dsparkCacheGiB, blobBytes: blob, minimum: 30)
    }
    guard slots >= 6 else {
      throw ConfigurationError(L10n.string("Slots must be at least 6."))
    }
    guard readWorkers >= 1, memoryLimitGiB >= 0, prefillStepSize >= 0 else {
      throw ConfigurationError(
        L10n.string(
          "Read workers must be greater than 0. Memory limit and prefill step size must be 0 or greater."
        ))
    }
    guard (prefetchReadWorkers ?? 2) >= 1, (moePrefillStepSize ?? 0) >= 0 else {
      throw ConfigurationError(L10n.string("Prefetch workers must be greater than 0. MoE prefill step size must be 0 or greater."))
    }
    guard (layerMajorPrefillThreshold ?? 1_024) >= 1 else {
      throw ConfigurationError(
        L10n.string("Layer-major prefill threshold must be greater than 0."))
    }
    guard promptCacheEntries >= 1, promptCacheMemoryGiB >= 1 else {
      throw ConfigurationError(
        L10n.string("Prompt cache entries and the memory limit must be greater than 0."))
    }
    guard (mtpSlots ?? 32) >= 10 else {
      throw ConfigurationError(L10n.string("MTP slots must be at least 10."))
    }
    guard (0...1).contains(anePrefillRatio ?? 0) else {
      throw ConfigurationError(L10n.string("ANE Prefill share must be from 0 through 1."))
    }
    guard dsparkSlots >= 30, (0...1).contains(dsparkConfidenceThreshold) else {
      throw ConfigurationError(L10n.string("Correct the default generation parameters."))
    }
    guard (1...272_000).contains(defaultMaxTokens),
      (0...2).contains(defaultTemperature),
      (0.000_001...1).contains(defaultTopP),
      (0...248_320).contains(defaultTopK)
    else {
      throw ConfigurationError(L10n.string("Correct the default generation parameters."))
    }
    if !warmupPromptPath.isEmpty,
      !FileManager.default.isReadableFile(atPath: warmupPromptPath)
    {
      throw ConfigurationError(L10n.string("The warmup prompt file cannot be read."))
    }
    if !modelKind.descriptor.supports("dspark"), dsparkEnabled {
      throw ConfigurationError(L10n.string("Qwen3.8-Flash-Next does not support DSpark."))
    }
  }

  private struct Legacy: Decodable {
    let publicModel: String?
    let slots: Int
    let readWorkers: Int
    let memoryLimitGiB: Int?
    let prefillStepSize: Int
    let layerMajorPrefill: Bool
    let promptCacheEntries: Int
    let promptCacheMemoryGiB: Int
    let warmupPromptPath: String
    let bf16KVCache: Bool
    let dsparkEnabled: Bool
    let dsparkSlots: Int
    let dsparkConfidenceThreshold: Double
    let defaultMaxTokens: Int
    let defaultTemperature: Double
    let defaultTopP: Double
    let defaultTopK: Int?
  }

  private static func legacySettings(
    for modelKind: ModelKind,
    defaults: UserDefaults
  ) -> ModelAdvancedSettings? {
    guard let data = defaults.data(forKey: ServerConfiguration.preferenceKey),
      let legacy = try? JSONDecoder().decode(Legacy.self, from: data)
    else { return nil }
    let identifiedKind: ModelKind
    if let savedKind = defaults.string(forKey: legacyModelPreference)
      .flatMap(ModelKind.init(rawValue:))
    {
      identifiedKind = savedKind
    } else {
      let selectedKind = defaults.string(forKey: "selectedInstallModelKind")
        .flatMap(ModelKind.init(rawValue:)) ?? .deepSeekV4
      switch legacy.publicModel {
      case "Qwen/Qwen3.8-Flash-Next-FP8", "qwen3.8-flash-next-fp8":
        identifiedKind = .qwen3_8FlashNext
      case "deepseek-v4.1-flash", "deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-flash":
        identifiedKind = .deepSeekV41
      case "deepseek-v4-flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731":
        identifiedKind = .deepSeekV4
      default:
        identifiedKind = selectedKind
      }
      defaults.set(identifiedKind.rawValue, forKey: legacyModelPreference)
    }
    guard identifiedKind == modelKind else { return nil }
    var settings = ModelAdvancedSettings.defaults(for: modelKind)
    // Preserve slot-based budgets when importing settings predating GiB fields.
    settings.expertCacheGiB = nil
    settings.mtpCacheGiB = nil
    settings.dsparkCacheGiB = nil
    settings.slots = legacy.slots
    settings.readWorkers = legacy.readWorkers
    settings.memoryLimitGiB = legacy.memoryLimitGiB ?? 0
    settings.prefillStepSize = legacy.prefillStepSize
    settings.layerMajorPrefill = legacy.layerMajorPrefill
    settings.promptCacheEntries = legacy.promptCacheEntries
    settings.promptCacheMemoryGiB = legacy.promptCacheMemoryGiB
    settings.warmupPromptPath = legacy.warmupPromptPath
    settings.bf16KVCache = legacy.bf16KVCache
    settings.dsparkEnabled = legacy.dsparkEnabled
    settings.dsparkSlots = legacy.dsparkSlots
    settings.dsparkConfidenceThreshold = legacy.dsparkConfidenceThreshold
    settings.defaultMaxTokens = legacy.defaultMaxTokens
    settings.defaultTemperature = legacy.defaultTemperature
    settings.defaultTopP = legacy.defaultTopP
    settings.defaultTopK = legacy.defaultTopK ?? 0
    return settings
  }

  private static func preferenceKey(for modelKind: ModelKind) -> String {
    "modelAdvancedSettings.\(modelKind.rawValue)"
  }
}

enum ServerLogLevel: String, Codable, CaseIterable, Identifiable {
  case debug
  case info
  case error

  var id: String { rawValue }

  var localizationKey: String {
    switch self {
    case .debug: "Debug"
    case .info: "Info"
    case .error: "Error"
    }
  }
}

struct ServerConfiguration: Codable, Equatable {
  static let preferenceKey = "serverConfiguration"
  static let powerSavingLimitOptionsGBps: [Double?] = [0.5, 1, 2, 3, 5, 10, 25, nil]

  var runtimeDirectory: String
  var pythonExecutable: String
  var pythonHome: String?
  var sitePackages: String?
  var host: String
  var port: Int
  var logLevel: ServerLogLevel
  var apiKey: String
  var powerSavingLimitGBps: Double?

  static var localDefault: ServerConfiguration {
    load(defaults: .standard, apiKey: AppKeychain.readAPIKey())
  }

  static func load(defaults: UserDefaults, apiKey: String) -> ServerConfiguration {
    let runtime = RuntimeEnvironment.current
    var configuration = ServerConfiguration(
      runtimeDirectory: runtime.runtimeDirectory.path,
      pythonExecutable: runtime.pythonExecutable.path,
      pythonHome: runtime.pythonHome?.path,
      sitePackages: runtime.sitePackages?.path,
      host: "127.0.0.1",
      port: 11_434,
      logLevel: .info,
      apiKey: "",
      powerSavingLimitGBps: nil
    )
    if let data = defaults.data(forKey: preferenceKey),
      var saved = decodeSavedConfiguration(data)
    {
      saved.runtimeDirectory = configuration.runtimeDirectory
      saved.pythonExecutable = configuration.pythonExecutable
      saved.pythonHome = configuration.pythonHome
      saved.sitePackages = configuration.sitePackages
      saved.apiKey = apiKey
      if !powerSavingLimitOptionsGBps.contains(where: {
        $0 == saved.powerSavingLimitGBps
      }) {
        saved.powerSavingLimitGBps = nil
      }
      configuration = saved
    } else {
      configuration.apiKey = apiKey
    }
    return configuration
  }

  private static func decodeSavedConfiguration(_ data: Data) -> ServerConfiguration? {
    try? JSONDecoder().decode(ServerConfiguration.self, from: data)
  }

  func save(defaults: UserDefaults = .standard) {
    var saved = self
    saved.runtimeDirectory = ""
    saved.pythonExecutable = ""
    saved.pythonHome = nil
    saved.sitePackages = nil
    saved.apiKey = ""
    guard let data = try? JSONEncoder().encode(saved) else { return }
    defaults.set(data, forKey: Self.preferenceKey)
  }

  var baseURL: URL? {
    let clientHost = ["0.0.0.0", "::"].contains(host) ? "127.0.0.1" : host
    let formattedHost = clientHost.contains(":") ? "[\(clientHost)]" : clientHost
    return URL(string: "http://\(formattedHost):\(port)")
  }

  func arguments(modelCatalogPath: String) -> [String] {
    [
      "-m", "deepseek_v4_ssd.server",
      "--model-catalog", modelCatalogPath,
      "--host", host,
      "--port", String(port),
      "--log-level", logLevel.rawValue,
    ]
  }

  func validate() throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: runtimeDirectory, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw ConfigurationError(L10n.string("The app runtime is missing. Install the app again."))
    }
    guard FileManager.default.isExecutableFile(atPath: pythonExecutable) else {
      throw ConfigurationError(
        L10n.string("Python is missing from the app. Install the app again."))
    }
    guard
      FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: runtimeDirectory)
          .appending(path: "deepseek_v4_ssd/server.py").path
      )
    else {
      throw ConfigurationError(L10n.string("The app runtime is incomplete. Install the app again."))
    }
    guard !host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw ConfigurationError(L10n.string("Host cannot be empty."))
    }
    guard (1...65_535).contains(port) else {
      throw ConfigurationError(L10n.string("Port must be from 1 through 65535."))
    }
    guard ["127.0.0.1", "::1", "localhost"].contains(host) || !apiKey.isEmpty else {
      throw ConfigurationError(L10n.string("An API key is required for a non-local host."))
    }
  }
}

extension ServerConfiguration {
  private enum CodingKeys: String, CodingKey {
    case runtimeDirectory
    case pythonExecutable
    case pythonHome
    case sitePackages
    case host
    case port
    case logLevel
    case apiKey
    case powerSavingLimitGBps
  }

  init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    runtimeDirectory = try values.decode(String.self, forKey: .runtimeDirectory)
    pythonExecutable = try values.decode(String.self, forKey: .pythonExecutable)
    pythonHome = try values.decodeIfPresent(String.self, forKey: .pythonHome)
    sitePackages = try values.decodeIfPresent(String.self, forKey: .sitePackages)
    host = try values.decode(String.self, forKey: .host)
    port = try values.decode(Int.self, forKey: .port)
    logLevel = try values.decodeIfPresent(ServerLogLevel.self, forKey: .logLevel) ?? .info
    apiKey = try values.decode(String.self, forKey: .apiKey)
    powerSavingLimitGBps = try values.decodeIfPresent(
      Double.self, forKey: .powerSavingLimitGBps)
  }
}

struct CatalogModel: Identifiable, Equatable, Sendable {
  let id: String
  let alias: String?

  var requestName: String { alias ?? id }
}

struct ModelCatalog: Codable, Equatable, Sendable {
  struct Entry: Codable, Equatable, Sendable {
    struct Runtime: Codable, Equatable, Sendable {
      let slots: Int
      var expertCacheBytes: UInt64? = nil
      var mtpCacheBytes: UInt64? = nil
      var dsparkCacheBytes: UInt64? = nil
      let readWorkers: Int
      let prefetchReadWorkers: Int
      let prefillStepSize: Int
      let fp8KVCache: Bool
      let memoryLimitGiB: Int
      let layerMajorPrefill: Bool
      let layerMajorPrefillThreshold: Int
      let promptCacheEntries: Int
      let promptCacheMemoryGiB: Int
      let persistentPromptCache: Bool
      let persistentPromptCacheEntries: Int
      let promptCacheDirectory: String?
      let moePrefillStepSize: Int
      let batchedExpertPrefill: Bool
      let qwenNextLayerPrefetch: Bool
      let qwenGroupedExperts: Bool
      let expertEvictionPolicy: String
      let anePrefill: Bool
      let anePrefillRatio: Double
      let fp4IndexCache: Bool
      let mtpEnabled: Bool
      let mtpSlots: Int
      let dsparkEnabled: Bool
      let dsparkPromptCache: Bool
      let dsparkConfidenceThreshold: Double
      let dsparkSlots: Int
      let dsparkFallbackEnabled: Bool
      let dsparkSequentialVerification: Bool
      let expertRouteTrace: String?
      let expertPageCacheProbe: Bool
      let separatePrefillIO: Bool
      let expertFileCachePolicy: String
      let readyExpertDecode: Bool
      let stagedExpertStreaming: Bool
      let powerSavingLimitGBps: Double?
      let qwenQuantizedKV: Bool
      let qwenQuantizedIndex: Bool
      let qwenPooledIndexCache: Bool
      let qwenNgramLookupOptimized: Bool
      let qwenCompileTensorOps: Bool
      let qwenPhaseMemory: Bool
      // Optional on decode so catalogs written before the App controls remain readable.
      var qwenExpertWaveSlots: Int? = nil
      var qwenNgramIO: String? = nil
      var qwenNgramCacheBytes: Int? = nil
      var qwenSparseSDPA: Bool? = nil
      var qwenQSAQueryChunk: Int? = nil
      var qwenQSAIndexed: Bool? = nil
      var qwenPrefillReadExperts: Int? = nil
      var qwenPrefillSeedExperts: Int? = nil
      var qwenSharedExpertOverlap: Bool? = nil
      let qwenMTPDraftTokens: Int
      let qwenMTPZeroAcceptanceLimit: Int
      let v41PackedKV: Bool
      let v41PackedIndex: Bool
      let v41CandidateIndex: Bool
      let v41CEDPrefill: Bool
      let v41NextLayerPrefetch: Bool
      let deepseekANEPrefill: Bool
      let v41LayerMajorPrefill: Bool

      enum CodingKeys: String, CodingKey {
        case slots
        case expertCacheBytes = "expert_cache_bytes"
        case mtpCacheBytes = "mtp_cache_bytes"
        case dsparkCacheBytes = "dspark_cache_bytes"
        case readWorkers = "read_workers"
        case prefetchReadWorkers = "prefetch_read_workers"
        case prefillStepSize = "prefill_step_size"
        case fp8KVCache = "fp8_kv_cache"
        case memoryLimitGiB = "memory_limit_gib"
        case layerMajorPrefill = "layer_major_prefill"
        case layerMajorPrefillThreshold = "layer_major_prefill_threshold"
        case promptCacheEntries = "prompt_cache_entries"
        case promptCacheMemoryGiB = "prompt_cache_memory_gib"
        case persistentPromptCache = "persistent_prompt_cache"
        case persistentPromptCacheEntries = "persistent_prompt_cache_entries"
        case promptCacheDirectory = "prompt_cache_directory"
        case moePrefillStepSize = "moe_prefill_step_size"
        case batchedExpertPrefill = "batched_expert_prefill"
        case qwenNextLayerPrefetch = "qwen_next_layer_prefetch"
        case qwenGroupedExperts = "qwen_grouped_experts"
        case expertEvictionPolicy = "expert_eviction_policy"
        case anePrefill = "ane_prefill"
        case anePrefillRatio = "ane_prefill_ratio"
        case fp4IndexCache = "fp4_index_cache"
        case mtpEnabled = "mtp_enabled"
        case mtpSlots = "mtp_slots"
        case dsparkEnabled = "dspark_enabled"
        case dsparkPromptCache = "dspark_prompt_cache"
        case dsparkConfidenceThreshold = "dspark_confidence_threshold"
        case dsparkSlots = "dspark_slots"
        case dsparkFallbackEnabled = "dspark_fallback_enabled"
        case dsparkSequentialVerification = "dspark_sequential_verification"
        case expertRouteTrace = "expert_route_trace"
        case expertPageCacheProbe = "expert_page_cache_probe"
        case separatePrefillIO = "separate_prefill_io"
        case expertFileCachePolicy = "expert_file_cache_policy"
        case readyExpertDecode = "ready_expert_decode"
        case stagedExpertStreaming = "staged_expert_streaming"
        case powerSavingLimitGBps = "power_saving_limit_gbps"
        case qwenQuantizedKV = "qwen_quantized_kv"
        case qwenQuantizedIndex = "qwen_quantized_index"
        case qwenPooledIndexCache = "qwen_pooled_index_cache"
        case qwenNgramLookupOptimized = "qwen_ngram_lookup_optimized"
        case qwenCompileTensorOps = "qwen_compile_tensor_ops"
        case qwenPhaseMemory = "qwen_phase_memory"
        case qwenExpertWaveSlots = "qwen_expert_wave_slots"
        case qwenNgramIO = "qwen_ngram_io"
        case qwenNgramCacheBytes = "qwen_ngram_cache_bytes"
        case qwenSparseSDPA = "qwen_sparse_sdpa"
        case qwenQSAQueryChunk = "qwen_qsa_query_chunk"
        case qwenQSAIndexed = "qwen_qsa_indexed"
        case qwenPrefillReadExperts = "qwen_prefill_read_experts"
        case qwenPrefillSeedExperts = "qwen_prefill_seed_experts"
        case qwenSharedExpertOverlap = "qwen_shared_expert_overlap"
        case qwenMTPDraftTokens = "qwen_mtp_draft_tokens"
        case qwenMTPZeroAcceptanceLimit = "qwen_mtp_zero_acceptance_limit"
        case v41PackedKV = "v41_packed_kv"
        case v41PackedIndex = "v41_packed_index"
        case v41CandidateIndex = "v41_candidate_index"
        case v41CEDPrefill = "v41_ced_prefill"
        case v41NextLayerPrefetch = "v41_next_layer_prefetch"
        case deepseekANEPrefill = "deepseek_ane_prefill"
        case v41LayerMajorPrefill = "v41_layer_major_prefill"
      }

      func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(slots, forKey: .slots)
        try values.encodeIfPresent(expertCacheBytes, forKey: .expertCacheBytes)
        try values.encodeIfPresent(mtpCacheBytes, forKey: .mtpCacheBytes)
        try values.encodeIfPresent(dsparkCacheBytes, forKey: .dsparkCacheBytes)
        try values.encode(qwenQuantizedKV, forKey: .qwenQuantizedKV)
        try values.encode(qwenQuantizedIndex, forKey: .qwenQuantizedIndex)
        try values.encode(qwenPooledIndexCache, forKey: .qwenPooledIndexCache)
        try values.encode(qwenNgramLookupOptimized, forKey: .qwenNgramLookupOptimized)
        try values.encode(qwenCompileTensorOps, forKey: .qwenCompileTensorOps)
        try values.encode(qwenPhaseMemory, forKey: .qwenPhaseMemory)
        try values.encode(qwenExpertWaveSlots ?? 0, forKey: .qwenExpertWaveSlots)
        try values.encode(qwenNgramIO ?? "mmap", forKey: .qwenNgramIO)
        try values.encode(qwenNgramCacheBytes ?? 0, forKey: .qwenNgramCacheBytes)
        try values.encode(qwenSparseSDPA ?? false, forKey: .qwenSparseSDPA)
        try values.encode(qwenQSAQueryChunk ?? 4, forKey: .qwenQSAQueryChunk)
        try values.encode(qwenQSAIndexed ?? false, forKey: .qwenQSAIndexed)
        try values.encode(qwenPrefillReadExperts ?? 1, forKey: .qwenPrefillReadExperts)
        try values.encode(qwenPrefillSeedExperts ?? 0, forKey: .qwenPrefillSeedExperts)
        try values.encode(qwenSharedExpertOverlap ?? false, forKey: .qwenSharedExpertOverlap)
        try values.encode(qwenMTPDraftTokens, forKey: .qwenMTPDraftTokens)
        try values.encode(qwenMTPZeroAcceptanceLimit, forKey: .qwenMTPZeroAcceptanceLimit)
        try values.encode(v41PackedKV, forKey: .v41PackedKV)
        try values.encode(v41PackedIndex, forKey: .v41PackedIndex)
        try values.encode(v41CandidateIndex, forKey: .v41CandidateIndex)
        try values.encode(v41CEDPrefill, forKey: .v41CEDPrefill)
        try values.encode(v41NextLayerPrefetch, forKey: .v41NextLayerPrefetch)
        try values.encode(deepseekANEPrefill, forKey: .deepseekANEPrefill)
        try values.encode(v41LayerMajorPrefill, forKey: .v41LayerMajorPrefill)
        try values.encode(readWorkers, forKey: .readWorkers)
        try values.encode(prefetchReadWorkers, forKey: .prefetchReadWorkers)
        try values.encode(prefillStepSize, forKey: .prefillStepSize)
        try values.encode(fp8KVCache, forKey: .fp8KVCache)
        try values.encode(memoryLimitGiB, forKey: .memoryLimitGiB)
        try values.encode(layerMajorPrefill, forKey: .layerMajorPrefill)
        try values.encode(
          layerMajorPrefillThreshold,
          forKey: .layerMajorPrefillThreshold
        )
        try values.encode(promptCacheEntries, forKey: .promptCacheEntries)
        try values.encode(promptCacheMemoryGiB, forKey: .promptCacheMemoryGiB)
        try values.encode(persistentPromptCache, forKey: .persistentPromptCache)
        try values.encode(persistentPromptCacheEntries, forKey: .persistentPromptCacheEntries)
        if let promptCacheDirectory {
          try values.encode(promptCacheDirectory, forKey: .promptCacheDirectory)
        } else {
          try values.encodeNil(forKey: .promptCacheDirectory)
        }
        try values.encode(moePrefillStepSize, forKey: .moePrefillStepSize)
        try values.encode(batchedExpertPrefill, forKey: .batchedExpertPrefill)
        try values.encode(qwenNextLayerPrefetch, forKey: .qwenNextLayerPrefetch)
        try values.encode(qwenGroupedExperts, forKey: .qwenGroupedExperts)
        try values.encode(expertEvictionPolicy, forKey: .expertEvictionPolicy)
        try values.encode(anePrefill, forKey: .anePrefill)
        try values.encode(anePrefillRatio, forKey: .anePrefillRatio)
        try values.encode(fp4IndexCache, forKey: .fp4IndexCache)
        try values.encode(mtpEnabled, forKey: .mtpEnabled)
        try values.encode(mtpSlots, forKey: .mtpSlots)
        try values.encode(dsparkEnabled, forKey: .dsparkEnabled)
        try values.encode(dsparkPromptCache, forKey: .dsparkPromptCache)
        try values.encode(dsparkConfidenceThreshold, forKey: .dsparkConfidenceThreshold)
        try values.encode(dsparkSlots, forKey: .dsparkSlots)
        try values.encode(dsparkFallbackEnabled, forKey: .dsparkFallbackEnabled)
        try values.encode(
          dsparkSequentialVerification, forKey: .dsparkSequentialVerification)
        if let expertRouteTrace {
          try values.encode(expertRouteTrace, forKey: .expertRouteTrace)
        } else {
          try values.encodeNil(forKey: .expertRouteTrace)
        }
        try values.encode(expertPageCacheProbe, forKey: .expertPageCacheProbe)
        try values.encode(separatePrefillIO, forKey: .separatePrefillIO)
        try values.encode(expertFileCachePolicy, forKey: .expertFileCachePolicy)
        try values.encode(readyExpertDecode, forKey: .readyExpertDecode)
        try values.encode(stagedExpertStreaming, forKey: .stagedExpertStreaming)
        if let powerSavingLimitGBps {
          try values.encode(powerSavingLimitGBps, forKey: .powerSavingLimitGBps)
        } else {
          try values.encodeNil(forKey: .powerSavingLimitGBps)
        }
      }
    }

    struct Defaults: Codable, Equatable, Sendable {
      let maxTokens: Int
      let temperature: Double
      let topP: Double
      let topK: Int
      let approximationMode: String?
      let qwenAdaptiveSampling: Bool?

      enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case approximationMode = "approximation_mode"
        case qwenAdaptiveSampling = "qwen_adaptive_sampling"
      }
    }

    let id: String
    let alias: String?
    let path: String
    let modelKind: String
    let runtime: Runtime
    let defaults: Defaults
    let warmupPromptPath: String?

    enum CodingKeys: String, CodingKey {
      case id, alias, path, runtime, defaults
      case modelKind = "model_kind"
      case warmupPromptPath = "warmup_prompt_path"
    }

    func encode(to encoder: Encoder) throws {
      var values = encoder.container(keyedBy: CodingKeys.self)
      try values.encode(id, forKey: .id)
      if let alias {
        try values.encode(alias, forKey: .alias)
      } else {
        try values.encodeNil(forKey: .alias)
      }
      try values.encode(path, forKey: .path)
      try values.encode(modelKind, forKey: .modelKind)
      try values.encode(runtime, forKey: .runtime)
      try values.encode(defaults, forKey: .defaults)
      if let warmupPromptPath {
        try values.encode(warmupPromptPath, forKey: .warmupPromptPath)
      } else {
        try values.encodeNil(forKey: .warmupPromptPath)
      }
    }

    func jsonObject() throws -> Any {
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(self))
    }
  }

  let version: Int
  let models: [Entry]

  init(models: [Entry]) {
    version = 1
    self.models = models
  }

  var availableModels: [CatalogModel] {
    models.map { CatalogModel(id: $0.id, alias: $0.alias) }
  }

  func encoded() throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try encoder.encode(self)
  }
}

struct TemporaryModelCatalog {
  let url: URL

  init(
    catalog: ModelCatalog,
    directory: URL = FileManager.default.temporaryDirectory
  ) throws {
    url = directory.appending(path: "whallm-model-catalog-\(UUID().uuidString).json")
    do {
      try catalog.encoded().write(to: url, options: .atomic)
      try FileManager.default.setAttributes(
        [.posixPermissions: 0o600],
        ofItemAtPath: url.path
      )
    } catch {
      try? FileManager.default.removeItem(at: url)
      throw error
    }
  }

  func remove() {
    try? FileManager.default.removeItem(at: url)
  }
}

enum AppKeychain {
  private static let service = "com.deepseekv4ssd.app"
  private static let account = "server-api-key"

  static func readAPIKey(service: String = service, account: String = account) -> String {
    var query = baseQuery(service: service, account: account)
    query[kSecReturnData as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var result: CFTypeRef?
    guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
      let data = result as? Data
    else { return "" }
    return String(data: data, encoding: .utf8) ?? ""
  }

  static func saveAPIKey(
    _ value: String,
    service: String = service,
    account: String = account
  ) {
    let query = baseQuery(service: service, account: account)
    if value.isEmpty {
      SecItemDelete(query as CFDictionary)
      return
    }
    guard readAPIKey(service: service, account: account) != value else { return }
    let data = Data(value.utf8)
    let status = SecItemUpdate(
      query as CFDictionary,
      [kSecValueData as String: data] as CFDictionary
    )
    if status == errSecItemNotFound {
      var item = query
      item[kSecValueData as String] = data
      SecItemAdd(item as CFDictionary, nil)
    }
  }

  private static func baseQuery(service: String, account: String) -> [String: Any] {
    [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
    ]
  }
}

struct ConfigurationError: LocalizedError {
  let message: String

  init(_ message: String) {
    self.message = message
  }

  var errorDescription: String? { message }
}

@MainActor
final class ServerController: ObservableObject {
  enum ModelAction: Equatable {
    case load(String)
    case unload(String)

    var modelID: String {
      switch self {
      case .load(let modelID), .unload(let modelID): modelID
      }
    }
  }

  enum State: Equatable {
    case stopped
    case starting
    case running
    case stopping
    case failed(String)

    var label: String {
      switch self {
      case .stopped: L10n.string("Stopped")
      case .starting: L10n.string("Starting")
      case .running: L10n.string("Running")
      case .stopping: L10n.string("Stopping")
      case .failed: L10n.string("Start failed")
      }
    }

    var symbol: String {
      switch self {
      case .running: "checkmark.circle.fill"
      case .starting, .stopping: "clock.fill"
      case .failed: "exclamationmark.triangle.fill"
      case .stopped: "stop.circle.fill"
      }
    }
  }

  @Published private(set) var state: State = .stopped
  @Published private(set) var log = ""
  @Published private(set) var performance = LivePerformance()
  @Published private(set) var performanceHistory = PerformanceHistory()
  @Published private(set) var catalogModels: [CatalogModel] = []
  @Published private(set) var modelAction: ModelAction?
  @Published private(set) var modelActionError: String?

  private var process: Process?
  private var outputTask: Task<Void, Never>?
  private var monitorTask: Task<Void, Never>?
  private var memoryResetTask: Task<Void, Never>?
  private var memoryResetRevision = 0
  private var memoryResetPending = false
  private var memoryResetFailed = false
  private var modelConfigurationTask: Task<Void, Never>?
  private var pendingModelConfigurations: [String: ModelCatalog.Entry] = [:]
  private var modelConfigurationErrors: [String: String] = [:]

  func waitForModelConfigurationUpdates(_ modelID: String) async throws {
    while let task = modelConfigurationTask {
      try Task.checkCancellation()
      await task.value
    }
    try Task.checkCancellation()
    if let message = modelConfigurationErrors[modelID] {
      throw ConfigurationError(message)
    }
  }
  private var monitorConfiguration: ServerConfiguration?
  private var temporaryModelCatalog: TemporaryModelCatalog?
  private var previousSSDBytes: UInt64?
  private var previousSSDTime: ContinuousClock.Instant?
  private var lastRecordedCompletedRequestCount = 0
  private var lastLoadedModel: String?

  var isActive: Bool {
    switch state {
    case .starting, .running, .stopping: true
    case .stopped, .failed: false
    }
  }

  var canManageModels: Bool { state == .running }

  func start(_ configuration: ServerConfiguration, catalog: ModelCatalog) {
    guard !isActive else { return }
    do {
      try configuration.validate()
      let temporaryModelCatalog = try TemporaryModelCatalog(catalog: catalog)
      self.temporaryModelCatalog = temporaryModelCatalog
      let process = Process()
      let output = Pipe()
      let runtimeURL = URL(fileURLWithPath: configuration.runtimeDirectory)
      process.executableURL = URL(fileURLWithPath: configuration.pythonExecutable)
      process.currentDirectoryURL = runtimeURL
      process.arguments = configuration.arguments(
        modelCatalogPath: temporaryModelCatalog.url.path)
      process.standardOutput = output
      process.standardError = output
      var environment = ProcessInfo.processInfo.environment
      environment["PYTHONPATH"] = [configuration.runtimeDirectory, configuration.sitePackages]
        .compactMap { $0 }
        .joined(separator: ":")
      environment["PYTHONDONTWRITEBYTECODE"] = "1"
      environment["WHALLM_APP_PID"] = String(ProcessInfo.processInfo.processIdentifier)
      if let pythonHome = configuration.pythonHome {
        environment["PYTHONHOME"] = pythonHome
      }
      if configuration.apiKey.isEmpty {
        environment.removeValue(forKey: "DEEPSEEK_API_KEY")
      } else {
        environment["DEEPSEEK_API_KEY"] = configuration.apiKey
      }
      process.environment = environment
      process.terminationHandler = { [weak self] finished in
        let status = finished.terminationStatus
        Task { @MainActor in self?.didTerminate(status: status) }
      }

      log = ""
      modelActionError = nil
      state = .starting
      try process.run()
      self.process = process
      catalogModels = catalog.availableModels
      monitorConfiguration = configuration
      readOutput(output.fileHandleForReading)
      startMonitoring()
    } catch {
      temporaryModelCatalog?.remove()
      temporaryModelCatalog = nil
      catalogModels = []
      state = .failed(error.localizedDescription)
      appendLog(L10n.string("Error: %@", error.localizedDescription) + "\n")
    }
  }

  func stop() {
    cancelModelConfigurationSync()
    guard let process, process.isRunning else {
      temporaryModelCatalog?.remove()
      temporaryModelCatalog = nil
      catalogModels = []
      state = .stopped
      return
    }
    state = .stopping
    process.interrupt()
  }

  func clearPerformanceHistory() {
    performanceHistory.clear()
    lastRecordedCompletedRequestCount = performance.completedRequestCount
    memoryResetTask?.cancel()
    memoryResetRevision += 1
    let revision = memoryResetRevision
    performance.appMemory = nil
    performance.snapshot.memoryUsage = .nan
    performance.memoryResetFailed = false
    memoryResetFailed = false
    memoryResetPending = false
    guard case .running = state, let configuration = monitorConfiguration,
      let baseURL = configuration.baseURL else { return }
    memoryResetPending = true
    memoryResetTask = Task { [weak self] in
      guard let self else { return }
      do {
        var request = URLRequest(url: baseURL.appending(path: "api/status/memory/reset"))
        request.httpMethod = "POST"
        request.httpBody = Data("{}".utf8)
        request.timeoutInterval = 2
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !configuration.apiKey.isEmpty {
          request.setValue("Bearer \(configuration.apiKey)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard !Task.isCancelled, revision == self.memoryResetRevision else { return }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
          let memory = try ServerStatus.decode(data).appMemory else {
          throw ConfigurationError("Memory reset failed")
        }
        self.memoryResetPending = false
        self.performance.appMemory = memory
        self.performance.snapshot.memoryUsage = memory.currentAppMemoryBytes ?? .nan
      } catch {
        guard !Task.isCancelled, revision == self.memoryResetRevision else { return }
        self.memoryResetPending = false
        self.memoryResetFailed = true
        self.performance.memoryResetFailed = true
      }
      self.memoryResetTask = nil
    }
  }

  func configureModel(_ configuration: ModelCatalog.Entry) {
    guard case .running = state else { return }
    pendingModelConfigurations[configuration.id] = configuration
    guard modelConfigurationTask == nil else { return }
    modelConfigurationTask = Task { [weak self] in
      await self?.flushModelConfigurations()
    }
  }

  func loadModel(_ modelID: String, catalog: () throws -> ModelCatalog) async {
    do {
      let configuration = try catalog().models.first { $0.id == modelID }
      guard let configuration else {
        throw ConfigurationError(L10n.string("The model is not available to this server."))
      }
      await changeLoadedModel(
        .load(modelID), path: "api/models/load", configuration: configuration)
    } catch {
      modelActionError = error.localizedDescription
    }
  }

  func unloadModel(_ modelID: String) async {
    await changeLoadedModel(.unload(modelID), path: "api/models/unload")
  }

  func unloadModelAfterBenchmark(_ modelID: String) async throws {
    // A stopped server has already released its model. The endpoint waits for
    // any cancelled generation to drain before releasing the selected model.
    guard case .running = state else { return }
    try await sendModelRequest(
      path: "api/models/unload", body: ["model": modelID], timeoutInterval: 1_800)
    await refreshPerformance()
  }

  private func changeLoadedModel(
    _ action: ModelAction,
    path: String,
    configuration: ModelCatalog.Entry? = nil
  ) async {
    guard modelAction == nil, case .running = state,
      monitorConfiguration?.baseURL != nil
    else { return }

    modelAction = action
    modelActionError = nil
    defer { modelAction = nil }

    do {
      if case .load = action, let modelConfigurationTask {
        await modelConfigurationTask.value
      }
      var body: [String: Any] = ["model": action.modelID]
      if let configuration {
        body["configuration"] = try configuration.jsonObject()
      }
      try await sendModelRequest(path: path, body: body, timeoutInterval: 1_800)
      if let configuration { updateCatalogModel(configuration) }
      await refreshPerformance()
    } catch {
      modelActionError = error.localizedDescription
    }
  }

  private func flushModelConfigurations() async {
    while !Task.isCancelled, let configuration = pendingModelConfigurations.values.first {
      pendingModelConfigurations.removeValue(forKey: configuration.id)
      do {
        try await sendModelRequest(
          path: "api/models/configure",
          body: ["configuration": try configuration.jsonObject()],
          timeoutInterval: 1_800
        )
        updateCatalogModel(configuration)
        modelConfigurationErrors.removeValue(forKey: configuration.id)
        modelActionError = nil
      } catch {
        modelConfigurationErrors[configuration.id] = error.localizedDescription
        if case .running = state {
          modelActionError = error.localizedDescription
        }
      }
    }
    modelConfigurationTask = nil
  }

  private func sendModelRequest(
    path: String,
    body: [String: Any],
    timeoutInterval: TimeInterval
  ) async throws {
    guard case .running = state,
      let configuration = monitorConfiguration,
      let baseURL = configuration.baseURL
    else { throw ConfigurationError(L10n.string("Start the server first")) }

    var request = URLRequest(url: baseURL.appending(path: path))
    request.httpMethod = "POST"
    request.timeoutInterval = timeoutInterval
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if !configuration.apiKey.isEmpty {
      request.setValue("Bearer \(configuration.apiKey)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let (_, response) = try await URLSession.shared.data(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw ConfigurationError(L10n.string("The server did not return an HTTP response."))
    }
    guard (200..<300).contains(http.statusCode) else {
      throw ConfigurationError(
        L10n.string("The server returned HTTP %lld.", Int64(http.statusCode)))
    }
  }

  private func updateCatalogModel(_ configuration: ModelCatalog.Entry) {
    guard let index = catalogModels.firstIndex(where: { $0.id == configuration.id })
    else { return }
    catalogModels[index] = CatalogModel(id: configuration.id, alias: configuration.alias)
  }

  private func cancelModelConfigurationSync() {
    modelConfigurationTask?.cancel()
    modelConfigurationTask = nil
    pendingModelConfigurations.removeAll()
    modelConfigurationErrors.removeAll()
  }

  private func readOutput(_ handle: FileHandle) {
    outputTask?.cancel()
    outputTask = Task.detached(priority: .utility) { [weak self] in
      while !Task.isCancelled {
        let data = handle.availableData
        guard !data.isEmpty else { break }
        let text = String(decoding: data, as: UTF8.self)
        await self?.received(text)
      }
    }
  }

  private func received(_ text: String) {
    appendLog(text)
    if case .starting = state, log.contains("Ready:") {
      state = .running
    }
  }

  private func startMonitoring() {
    monitorTask?.cancel()
    monitorTask = Task { [weak self] in
      while !Task.isCancelled {
        guard let self else { return }
        await self.refreshPerformance()
        do {
          try await Task.sleep(for: .seconds(1))
        } catch {
          return
        }
      }
    }
  }

  private func refreshPerformance() async {
    guard let process, process.isRunning else { return }
    let memoryRevision = memoryResetRevision
    let canAcceptMemory = !memoryResetPending
    guard case .running = state,
      let configuration = monitorConfiguration,
      let baseURL = configuration.baseURL
    else { return }

    var request = URLRequest(url: baseURL.appending(path: "api/status"))
    request.timeoutInterval = 2
    request.cachePolicy = .reloadIgnoringLocalCacheData
    if !configuration.apiKey.isEmpty {
      request.setValue("Bearer \(configuration.apiKey)", forHTTPHeaderField: "Authorization")
    }
    do {
      let (data, response) = try await URLSession.shared.data(for: request)
      guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
        performance.snapshot.memoryUsage = .nan
        performance.appMemory = nil
        return
      }
      let status = try ServerStatus.decode(data)
      // A response started before/during Clear must not restore the old peak.
      let memory = canAcceptMemory && !memoryResetPending && memoryRevision == memoryResetRevision
        ? status.appMemory : performance.appMemory
      updateLoadedModel(
        status.loadedModel,
        completedRequestCount: status.performance.completedRequestCount
      )
      let now = ContinuousClock.now
      let bytesPerSecond: Double
      if let previousSSDBytes, let previousSSDTime,
        status.performance.ssdBytesRead >= previousSSDBytes
      {
        let elapsed = seconds(from: previousSSDTime.duration(to: now))
        bytesPerSecond =
          elapsed > 0 ? Double(status.performance.ssdBytesRead - previousSSDBytes) / elapsed : 0
      } else {
        bytesPerSecond = 0
      }
      previousSSDBytes = status.performance.ssdBytesRead
      previousSSDTime = now
      let cache = status.performance.activeParametersCache
      let requestCompleted = status.performance.completedRequestCount > 0
      let cacheHitRate =
        status.performance.generating || !requestCompleted
        ? cache.hitRate : status.performance.requestExpertCacheHitRate
      let live = LivePerformance(
        hasStatus: true,
        generating: status.performance.generating,
        completedRequestCount: status.performance.completedRequestCount,
        accumulatedOutputTokens: status.performance.accumulatedGenerationTokens,
        snapshot: PerformanceSnapshot(
          prefillTokensPerSecond: status.performance.prefillTokensPerSecond,
          decodeTokensPerSecond: status.performance.decodeTokensPerSecond,
          inputTokens: Double(status.performance.runtimePromptTokens),
          outputTokens: Double(status.performance.runtimeGenerationTokens),
          memoryUsage: memory?.currentAppMemoryBytes ?? .nan,
          ssdReadSpeed: bytesPerSecond,
          cacheHitRate: cacheHitRate,
          firstTokenWaitTime: status.performance.timeToFirstTokenSeconds,
          completionTime: status.performance.requestSeconds
        ),
        dsparkEnabled: status.performance.dsparkEnabled ?? false,
        dsparkAcceptanceRate: status.performance.dsparkAcceptanceRate ?? 0,
        dsparkAverageAcceptedLength: status.performance.dsparkAverageAcceptedLength ?? 0,
        loadedModel: status.loadedModel,
        loadingModel: status.loadingModel,
        appMemory: memory,
        memoryResetFailed: memoryResetFailed
      )
      performance = live
      recordPerformanceSample(live)
    } catch {
      performance.snapshot.memoryUsage = .nan
      performance.appMemory = nil
      return
    }
  }

  func recordPerformanceSample(_ live: LivePerformance) {
    if live.completedRequestCount < lastRecordedCompletedRequestCount {
      lastRecordedCompletedRequestCount = live.completedRequestCount
    }

    let requestCompleted = live.completedRequestCount > lastRecordedCompletedRequestCount
    if live.generating || requestCompleted {
      performanceHistory.record(
        live.snapshot,
        excluding: requestCompleted ? nil : .firstTokenWaitTime
      )
    }
    if requestCompleted {
      lastRecordedCompletedRequestCount = live.completedRequestCount
    }
  }

  func updateLoadedModel(_ model: String?, completedRequestCount: Int) {
    guard model != lastLoadedModel else { return }
    lastLoadedModel = model
    guard model != nil else { return }
    performanceHistory.clear()
    lastRecordedCompletedRequestCount = completedRequestCount
    previousSSDBytes = nil
    previousSSDTime = nil
  }

  private func seconds(from duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }

  private func appendLog(_ text: String) {
    log += text
    if log.count > 40_000 {
      log.removeFirst(log.count - 40_000)
    }
  }

  private func didTerminate(status: Int32) {
    cancelModelConfigurationSync()
    outputTask?.cancel()
    outputTask = nil
    monitorTask?.cancel()
    monitorTask = nil
    memoryResetTask?.cancel()
    memoryResetTask = nil
    memoryResetRevision += 1
    memoryResetPending = false
    memoryResetFailed = false
    monitorConfiguration = nil
    temporaryModelCatalog?.remove()
    temporaryModelCatalog = nil
    catalogModels = []
    modelAction = nil
    modelActionError = nil
    previousSSDBytes = nil
    previousSSDTime = nil
    performance = LivePerformance()
    lastLoadedModel = nil
    process = nil
    if case .stopping = state {
      state = .stopped
    } else if status == 0 {
      state = .stopped
    } else {
      let message = L10n.string("The server stopped with exit code %d.", status)
      state = .failed(message)
      appendLog("\n\(message)\n")
    }
  }
}
