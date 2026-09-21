import AppKit
import Combine
import DeepSeekRepack
import Foundation

extension ModelKind {
  var displayName: String { descriptor.displayName }
  var modelKindLabel: String { descriptor.kindLabel }
  var assistantName: String { descriptor.assistantName }
  var apiModelID: String { descriptor.apiModelID }
}

enum ModelAliasError: LocalizedError, Equatable {
  case conflict(String)

  var errorDescription: String? {
    switch self {
    case .conflict(let name):
      L10n.string("Alias conflicts with %@.", name)
    }
  }
}

struct InstalledModelInfo: Identifiable, Equatable, Sendable {
  let url: URL
  let size: UInt64
  let quickIssues: [InstalledFileIssue]
  let hasMTP: Bool
  let hasDSpark: Bool
  let modelKind: ModelKind
  let modelID: String

  var id: String { url.path }
  var name: String { url.deletingPathExtension().lastPathComponent }
  var isUsable: Bool { quickIssues.isEmpty }
  var modelKindLabel: String { modelKind.modelKindLabel }
  var assistantName: String { modelKind.assistantName }
}

struct ModelDiscoveryResult: Sendable {
  let models: [InstalledModelInfo]
  let invalidModelURLs: [URL]
}

enum InstalledModelDiscovery {
  static func find(in root: URL) -> ModelDiscoveryResult {
    let root = root.standardizedFileURL
    let children =
      (try? FileManager.default.contentsOfDirectory(
        at: root,
        includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
        options: [.skipsHiddenFiles]
      )) ?? []
    let directories = children.filter {
      let values = try? $0.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
      return values?.isDirectory == true && values?.isSymbolicLink != true
        && $0.pathExtension != "partial"
    }
    var models: [InstalledModelInfo] = []
    var invalidModelURLs: [URL] = []
    for candidate in [root] + directories
    where FileManager.default.fileExists(atPath: candidate.appending(path: "manifest.json").path) {
      if let model = inspect(candidate) {
        models.append(model)
      } else {
        invalidModelURLs.append(candidate)
      }
    }
    return ModelDiscoveryResult(
      models: models.sorted {
        $0.name.localizedStandardCompare($1.name) == .orderedAscending
      },
      invalidModelURLs: invalidModelURLs.sorted {
        $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
      }
    )
  }

  static func inspect(_ root: URL) -> InstalledModelInfo? {
    let root = root.standardizedFileURL.resolvingSymlinksInPath()
    guard let manifest = try? InstalledModel.loadManifest(at: root) else { return nil }
    let paths = Set(manifest.files.map(\.path))
    let modelKind = manifest.modelKind ?? .deepSeekV4
    let requiredPaths = modelKind.descriptor.requiredPaths
    guard requiredPaths.allSatisfy(paths.contains) else { return nil }
    let expectedLayerSize = UInt64(manifest.expertCount) * manifest.expertBlobSize
    var totalSize: UInt64 = 0
    var issues: [InstalledFileIssue] = []
    for file in manifest.files {
      guard !file.path.hasPrefix("/"), !file.path.split(separator: "/").contains("..") else {
        return nil
      }
      let target = root.appending(path: file.path).standardizedFileURL.resolvingSymlinksInPath()
      guard target.path.hasPrefix(root.path + "/") else { return nil }
      let attributes = try? FileManager.default.attributesOfItem(atPath: target.path)
      if (attributes?[.type] as? FileAttributeType) != .typeRegular {
        issues.append(InstalledFileIssue(path: file.path, kind: .missing))
      } else if (attributes?[.size] as? NSNumber)?.uint64Value != file.size
        || (file.path.hasPrefix("experts/layer_") && file.size != expectedLayerSize)
      {
        issues.append(InstalledFileIssue(path: file.path, kind: .sizeMismatch))
      }
      let sum = totalSize.addingReportingOverflow(file.size)
      guard !sum.overflow else { return nil }
      totalSize = sum.partialValue
    }
    return InstalledModelInfo(
      url: root,
      size: totalSize,
      quickIssues: issues,
      hasMTP: manifest.mtp != nil,
      hasDSpark: manifest.dspark != nil,
      modelKind: modelKind,
      modelID: manifest.modelID
    )
  }
}

enum PreflightStatus: Sendable {
  case passed
  case warning
  case failed
}

struct PreflightCheck: Identifiable, Sendable {
  let id: String
  let title: String
  let detail: String
  let status: PreflightStatus
  let blocksDownload: Bool
}

enum ModelDownloadBlock: Equatable, Sendable {
  case loadingInstallationPlan
  case installationPlanUnavailable
  case unsupportedArchitecture
  case modelFolderNotWritable
  case storageUnavailable
  case insufficientStorage(requiredBytes: UInt64, availableBytes: UInt64)

  var message: String {
    switch self {
    case .loadingInstallationPlan:
      L10n.string("Loading model installation information.")
    case .installationPlanUnavailable:
      L10n.string(
        "The model installation information is not available. Check the network and try again.")
    case .unsupportedArchitecture:
      L10n.string("This runtime does not support Intel Mac.")
    case .modelFolderNotWritable:
      L10n.string("The app cannot write to this folder. Select another folder.")
    case .storageUnavailable:
      L10n.string(
        "The app cannot read the available space for this model folder. Select another folder.")
    case .insufficientStorage(let requiredBytes, let availableBytes):
      L10n.string(
        "Not enough space. The model needs %@. The model folder has %@ available.",
        Self.formattedBytes(requiredBytes),
        Self.formattedBytes(availableBytes)
      )
    }
  }

  private static func formattedBytes(_ bytes: UInt64) -> String {
    if bytes == 0 { return "0 KB" }
    return ByteCountFormatter.string(
      fromByteCount: Int64(clamping: bytes), countStyle: .file)
  }
}

enum ModelOperationPhase: Equatable {
  case idle
  case preparingDownload
  case downloading
  case installingMTP
  case cancelling
  case verifying
  case preparingRepair
  case repairing
  case installingDSpark
  case installingArtifact(String)

  var label: String {
    switch self {
    case .idle: ""
    case .preparingDownload: L10n.string("Preparing download")
    case .downloading: L10n.string("Downloading and installing the model")
    case .installingMTP: L10n.string("Downloading and installing MTP")
    case .cancelling: L10n.string("Stopping")
    case .verifying: L10n.string("Verifying the complete model")
    case .preparingRepair: L10n.string("Preparing repair")
    case .repairing: L10n.string("Downloading damaged data again")
    case .installingDSpark: L10n.string("Installing DSpark")
    case .installingArtifact(let label): L10n.string(label)
    }
  }
}

struct ModelOperationProgress: Equatable {
  let completedBytes: UInt64
  let totalBytes: UInt64
  let bytesPerSecond: Double?
  let estimatedSecondsRemaining: Double?

  var fraction: Double? {
    guard totalBytes > 0 else { return nil }
    return min(1, Double(completedBytes) / Double(totalBytes))
  }
}

@MainActor
final class ModelLibrary: ObservableObject {
  static let supportedModelKinds: [ModelKind] = ModelPackages.descriptors.compactMap {
    ModelKind(rawValue: $0.kind)
  }
  static let qwenMTPInstalledBytes: UInt64 = 1_518_071_296
  static let rootPreference = "modelLibraryRoot"
  private static let activeDownloadPreference = "modelDownloadWasActive"
  private static let activeDestinationPreference = "modelDownloadDestination"
  private static let activeDownloadModelKindPreference = "modelDownloadModelKind"
  private static let selectedModelKindPreference = "selectedInstallModelKind"
  private static let aliasMigrationPreference = "modelAliasMigrationVersion"
  nonisolated private static let recommendedMemoryBytes: UInt64 = 64 * 1_024 * 1_024 * 1_024

  @Published private(set) var rootURL: URL
  @Published private(set) var models: [InstalledModelInfo] = []
  @Published private(set) var invalidModelURLs: [URL] = []
  @Published private(set) var preflightChecks: [PreflightCheck] = []
  @Published private(set) var isScanning = false
  @Published private(set) var operationPhase = ModelOperationPhase.idle
  @Published private(set) var operationProgress: ModelOperationProgress?
  @Published private(set) var downloadModelKind: ModelKind?
  @Published private(set) var message: String?
  @Published private(set) var verificationModelPath: String?
  @Published private(set) var verificationIssues: [InstalledFileIssue]?
  @Published private(set) var installationBytesByModel: [String: UInt64] = [
    ModelKind.deepSeekV4.rawValue: 166_878_580_480,
    // Pinned V4.1 revision dba1be0: repack plan installed weight bytes.
    ModelKind.deepSeekV41.rawValue: 501_382_643_728,
    ModelKind.qwen3_8FlashNext.rawValue: 125_291_490_955,
  ]
  @Published private(set) var planningModelKinds: Set<String> = []
  @Published private(set) var installationPlanErrors: [String: String] = [:]
  @Published private(set) var modelFolderAvailableBytes: UInt64?
  @Published private(set) var modelFolderIsWritable = false
  @Published private(set) var aliases: [ModelKind: String] = [:]
  @Published var selectedModelKind: ModelKind {
    didSet {
      defaults.set(selectedModelKind.rawValue, forKey: Self.selectedModelKindPreference)
      refreshPreflight()
      Task { await refreshInstallationPlan(for: selectedModelKind) }
    }
  }

  private let defaults: UserDefaults
  private var operationTask: Task<Void, Never>?
  private var downloadStart: ContinuousClock.Instant?
  private var partialAllocatedBytesByModel: [String: UInt64] = [:]

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    selectedModelKind =
      defaults.string(forKey: Self.selectedModelKindPreference).flatMap(ModelKind.init(rawValue:))
      ?? .deepSeekV4
    if let savedPath = defaults.string(forKey: Self.rootPreference) {
      rootURL = URL(fileURLWithPath: savedPath, isDirectory: true)
    } else {
      rootURL = FileManager.default.homeDirectoryForCurrentUser.appending(
        path: ".dsmodel", directoryHint: .isDirectory)
    }
    Self.migrateLegacyAlias(defaults: defaults, selectedModelKind: selectedModelKind)
    aliases = Dictionary(
      uniqueKeysWithValues: Self.supportedModelKinds.compactMap { modelKind in
        defaults.string(forKey: Self.aliasPreferenceKey(for: modelKind)).map {
          (modelKind, $0)
        }
      }
    )
  }

  var usableModels: [InstalledModelInfo] { models.filter(\.isUsable) }
  var damagedModels: [InstalledModelInfo] { models.filter { !$0.isUsable } }
  var needsSelectedModelDownload: Bool { usableModel(for: selectedModelKind) == nil }
  var isBusy: Bool { operationPhase != .idle }
  var canDownload: Bool {
    canDownload(selectedModelKind)
  }
  var hasPartialDownload: Bool {
    Self.supportedModelKinds.contains(where: hasPartialDownload(for:))
  }

  func alias(for modelKind: ModelKind) -> String {
    aliases[modelKind] ?? ""
  }

  @discardableResult
  func saveAlias(_ value: String, for modelKind: ModelKind) throws -> String {
    let alias = value.trimmingCharacters(in: .whitespacesAndNewlines)
    if !alias.isEmpty {
      for otherKind in Self.supportedModelKinds
      where otherKind != modelKind && otherKind.apiModelID == alias {
        throw ModelAliasError.conflict(otherKind.apiModelID)
      }
      for otherKind in Self.supportedModelKinds
      where otherKind != modelKind && aliases[otherKind] == alias {
        throw ModelAliasError.conflict(alias)
      }
    }
    if alias.isEmpty {
      aliases.removeValue(forKey: modelKind)
      defaults.removeObject(forKey: Self.aliasPreferenceKey(for: modelKind))
    } else {
      aliases[modelKind] = alias
      defaults.set(alias, forKey: Self.aliasPreferenceKey(for: modelKind))
    }
    return alias
  }

  func makeServerCatalog(powerSavingLimitGBps: Double?) throws -> ModelCatalog {
    let settings = Dictionary(
      uniqueKeysWithValues: Self.supportedModelKinds.map { modelKind in
        (
          modelKind,
          ModelAdvancedSettings.loadOrDefault(for: modelKind, defaults: defaults)
        )
      }
    )
    return try Self.makeServerCatalog(
      models: models.filter { canUseModel(at: $0.url.path) },
      aliases: aliases,
      settings: settings,
      powerSavingLimitGBps: powerSavingLimitGBps
    )
  }

  static func makeServerCatalog(
    models: [InstalledModelInfo],
    aliases: [ModelKind: String],
    settings: [ModelKind: ModelAdvancedSettings],
    powerSavingLimitGBps: Double?
  ) throws -> ModelCatalog {
    let entries = try supportedModelKinds.compactMap { modelKind -> ModelCatalog.Entry? in
      guard let model = models.first(where: { $0.modelKind == modelKind && $0.isUsable })
      else { return nil }
      let settings = (settings[modelKind] ?? .defaults(for: modelKind))
        .normalized(for: modelKind)
      try settings.validate(for: modelKind)
      let flashWaves = modelKind == .qwen3_8FlashNext && settings.qwenFlashWavesEnabled
      let alias = aliases[modelKind] ?? ""
      return ModelCatalog.Entry(
        id: modelKind.apiModelID,
        alias: alias.isEmpty ? nil : alias,
        path: model.url.path,
        modelKind: modelKind.rawValue,
        runtime: ModelCatalog.Entry.Runtime(
          slots: settings.slots,
          expertCacheBytes: try settings.expertCacheGiB.map { try ExpertMemory.bytes(gib: $0) },
          mtpCacheBytes: try settings.mtpCacheGiB.map { try ExpertMemory.bytes(gib: $0) },
          dsparkCacheBytes: try settings.dsparkCacheGiB.map { try ExpertMemory.bytes(gib: $0) },
          readWorkers: settings.readWorkers,
          prefetchReadWorkers: settings.prefetchReadWorkers ?? 2,
          prefillStepSize: settings.prefillStepSize,
          fp8KVCache: !settings.bf16KVCache,
          memoryLimitGiB: settings.memoryLimitGiB,
          layerMajorPrefill: settings.layerMajorPrefill,
          layerMajorPrefillThreshold: settings.layerMajorPrefillThreshold ?? 1_024,
          promptCacheEntries: settings.promptCacheMode == .off ? 0 : settings.promptCacheEntries,
          promptCacheMemoryGiB: settings.promptCacheMemoryGiB,
          persistentPromptCache: settings.promptCacheMode == .disk,
          persistentPromptCacheEntries: 8,
          promptCacheDirectory: nil,
          moePrefillStepSize: settings.moePrefillStepSize ?? 0,
          batchedExpertPrefill: settings.batchedExpertPrefill == true && !flashWaves,
          qwenNextLayerPrefetch: modelKind == .qwen3_8FlashNext && settings.nextLayerPrefetch == true && settings.layerMajorPrefill && !flashWaves,
          qwenGroupedExperts: settings.qwenGroupedExperts == true && !flashWaves,
          expertEvictionPolicy: settings.routeAwareExpertCache == true ? "route" : (settings.recentExpertCache == true ? "lru" : "lfu"),
          anePrefill: modelKind.descriptor.supports("anePrefill"),
          anePrefillRatio: settings.anePrefillRatio ?? 0,
          fp4IndexCache: true,
          mtpEnabled: settings.mtpEnabled == true && model.hasMTP,
          mtpSlots: settings.mtpSlots ?? 32,
          dsparkEnabled: settings.dsparkEnabled && model.hasDSpark,
          dsparkPromptCache: false,
          dsparkConfidenceThreshold: settings.dsparkConfidenceThreshold,
          dsparkSlots: settings.dsparkSlots,
          dsparkFallbackEnabled: true,
          dsparkSequentialVerification: false,
          expertRouteTrace: nil,
          expertPageCacheProbe: false,
          separatePrefillIO: true,
          expertFileCachePolicy: "cached",
          readyExpertDecode: settings.readyExpertDecode == true,
          stagedExpertStreaming: false,
          powerSavingLimitGBps: powerSavingLimitGBps,
          qwenQuantizedKV: modelKind == .qwen3_8FlashNext && settings.packedKVCache == true,
          qwenQuantizedIndex: modelKind == .qwen3_8FlashNext && settings.packedIndexCache == true,
          qwenPooledIndexCache: modelKind == .qwen3_8FlashNext && settings.qwenPooledIndexCache == true,
          qwenNgramLookupOptimized: modelKind == .qwen3_8FlashNext && settings.qwenNgramLookupOptimized == true,
          qwenCompileTensorOps: modelKind == .qwen3_8FlashNext && settings.qwenCompileTensorOps == true,
          qwenPhaseMemory: modelKind == .qwen3_8FlashNext && settings.qwenPhaseMemory == true,
          qwenExpertWaveSlots: settings.qwenExpertWaveSlots ?? 0,
          qwenNgramIO: settings.qwenNgramIO ?? "mmap",
          qwenNgramCacheBytes: settings.effectiveQwenNgramCacheBytes,
          qwenSparseSDPA: settings.qwenSparseSDPA ?? false,
          qwenQSAQueryChunk: settings.qwenQSAQueryChunk ?? 4,
          qwenQSAIndexed: settings.qwenSparseSDPA == true && settings.qwenQSAIndexed == true,
          qwenMTPDraftTokens: modelKind == .qwen3_8FlashNext ? settings.effectiveQwenMTPDraftTokens : 5,
          qwenMTPZeroAcceptanceLimit: modelKind == .qwen3_8FlashNext ? settings.effectiveQwenMTPZeroAcceptanceLimit : 1,
          v41PackedKV: modelKind == .deepSeekV41 && settings.packedKVCache == true,
          v41PackedIndex: modelKind == .deepSeekV41 && settings.packedIndexCache == true,
          v41CandidateIndex: modelKind == .deepSeekV41 && settings.candidateIndex == true,
          v41CEDPrefill: modelKind == .deepSeekV41 && settings.cedPrefill == true && settings.layerMajorPrefill && !settings.dsparkEnabled,
          v41NextLayerPrefetch: modelKind == .deepSeekV41 && settings.nextLayerPrefetch == true && settings.layerMajorPrefill && settings.batchedExpertPrefill != false,
          deepseekANEPrefill: modelKind != .qwen3_8FlashNext && settings.deepSeekANEPrefill == true,
          v41LayerMajorPrefill: modelKind == .deepSeekV41 && settings.layerMajorPrefill
        ),
        defaults: ModelCatalog.Entry.Defaults(
          maxTokens: settings.defaultMaxTokens,
          temperature: settings.defaultTemperature,
          topP: settings.defaultTopP,
          topK: settings.defaultTopK,
          approximationMode: settings.approximationEnabled == true && !settings.dsparkEnabled && settings.mtpEnabled != true
            ? "learned-route-drop-lowest-1" : "exact",
          qwenAdaptiveSampling: settings.qwenAdaptiveSampling ?? true
        ),
        warmupPromptPath: settings.warmupPromptPath.isEmpty
          ? nil : settings.warmupPromptPath
      )
    }
    return ModelCatalog(models: entries)
  }

  static func aliasPreferenceKey(for modelKind: ModelKind) -> String {
    "modelAlias.\(modelKind.rawValue)"
  }

  private static func migrateLegacyAlias(
    defaults: UserDefaults,
    selectedModelKind: ModelKind
  ) {
    guard defaults.integer(forKey: aliasMigrationPreference) < 1 else { return }
    defer { defaults.set(1, forKey: aliasMigrationPreference) }
    guard defaults.string(forKey: aliasPreferenceKey(for: selectedModelKind)) == nil,
      let data = defaults.data(forKey: ServerConfiguration.preferenceKey),
      let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let rawName = payload["publicModel"] as? String
    else { return }
    let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
    let legacyDefaults: Set<String> = [
      "deepseek-v4-flash-0731",
      "deepseek-ai/DeepSeek-V4-Flash-0731",
      "deepseek-v4.1-flash",
      "deepseek-ai/DeepSeek-V4.1-Flash",
      "qwen3.8-flash-next-fp8",
      "Qwen/Qwen3.8-Flash-Next-FP8",
    ]
    guard !name.isEmpty, !legacyDefaults.contains(name),
      !supportedModelKinds.contains(where: {
        $0 != selectedModelKind && $0.apiModelID == name
      })
    else { return }
    defaults.set(name, forKey: aliasPreferenceKey(for: selectedModelKind))
  }

  func plannedInstalledBytes(for modelKind: ModelKind) -> UInt64? {
    installationBytesByModel[modelKind.rawValue]
  }

  func isPlanningInstallation(for modelKind: ModelKind) -> Bool {
    planningModelKinds.contains(modelKind.rawValue)
  }

  func hasPartialDownload(for modelKind: ModelKind) -> Bool {
    FileManager.default.fileExists(atPath: partialDownloadURL(for: modelKind).path)
  }

  func requiredStorageBytes(for modelKind: ModelKind) -> UInt64? {
    plannedInstalledBytes(for: modelKind).map {
      let allocated = partialAllocatedBytesByModel[modelKind.rawValue] ?? 0
      return $0 > allocated ? $0 - allocated : 0
    }
  }

  func downloadBlock(for modelKind: ModelKind) -> ModelDownloadBlock? {
    let key = modelKind.rawValue
    guard plannedInstalledBytes(for: modelKind) != nil else {
      return installationPlanErrors[key] == nil
        ? .loadingInstallationPlan : .installationPlanUnavailable
    }
    guard !preflightChecks.contains(where: { $0.blocksDownload && $0.status == .failed }) else {
      return .unsupportedArchitecture
    }
    guard modelFolderIsWritable else { return .modelFolderNotWritable }
    guard let requiredBytes = requiredStorageBytes(for: modelKind) else {
      return .loadingInstallationPlan
    }
    return Self.storageDownloadBlock(
      requiredBytes: requiredBytes,
      availableBytes: modelFolderAvailableBytes
    )
  }

  func canDownload(_ modelKind: ModelKind) -> Bool {
    downloadBlock(for: modelKind) == nil
  }

  func canStartDownload(_ modelKind: ModelKind) -> Bool {
    canDownload(modelKind)
      && !FileManager.default.fileExists(atPath: downloadDestination(for: modelKind).path)
  }

  func hasPartialMTPInstallation(for model: InstalledModelInfo) -> Bool {
    FileManager.default.fileExists(atPath: mtpInstallationPartialURL(for: model).path)
  }

  func mtpDownloadBlock(for model: InstalledModelInfo) -> ModelDownloadBlock? {
    guard model.modelKind.descriptor.supports("mtp"), !model.hasMTP else { return nil }
    guard !preflightChecks.contains(where: { $0.blocksDownload && $0.status == .failed }) else {
      return .unsupportedArchitecture
    }
    guard modelFolderIsWritable else { return .modelFolderNotWritable }
    let allocated = Self.allocatedBytes(at: mtpInstallationPartialURL(for: model))
    let requiredBytes =
      Self.qwenMTPInstalledBytes > allocated
      ? Self.qwenMTPInstalledBytes - allocated : 0
    return Self.storageDownloadBlock(
      requiredBytes: requiredBytes,
      availableBytes: modelFolderAvailableBytes
    )
  }

  func model(at path: String) -> InstalledModelInfo? {
    models.first { $0.url.path == path }
  }

  func usableModel(for kind: ModelKind) -> InstalledModelInfo? {
    usableModels.first { $0.modelKind == kind }
  }

  func canUseModel(at path: String) -> Bool {
    guard model(at: path)?.isUsable == true else { return false }
    return verificationModelPath != path || verificationIssues?.isEmpty != false
  }

  func setRoot(_ url: URL) async {
    let newRootURL = url.standardizedFileURL
    if newRootURL != rootURL {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      defaults.removeObject(forKey: Self.activeDestinationPreference)
      defaults.removeObject(forKey: Self.activeDownloadModelKindPreference)
    }
    rootURL = newRootURL
    defaults.set(rootURL.path, forKey: Self.rootPreference)
    verificationModelPath = nil
    verificationIssues = nil
    await scan()
  }

  func scan() async {
    isScanning = true
    message = nil
    do {
      try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
      let root = rootURL
      let result = await Task.detached(priority: .utility) {
        InstalledModelDiscovery.find(in: root)
      }.value
      models = result.models
      invalidModelURLs = result.invalidModelURLs
      refreshPreflight()
      await refreshSelectedPlan()
    } catch {
      models = []
      invalidModelURLs = []
      refreshPreflight()
      message = L10n.string("The model folder cannot be read. Select a writable folder.")
    }
    isScanning = false
  }

  func refreshPreflight() {
    let folderStatus = Self.inspectModelFolder(rootURL)
    modelFolderIsWritable = folderStatus.isWritable
    modelFolderAvailableBytes = folderStatus.availableBytes
    partialAllocatedBytesByModel = Dictionary(
      uniqueKeysWithValues: Self.supportedModelKinds.map {
        ($0.rawValue, Self.allocatedBytes(at: partialDownloadURL(for: $0)))
      }
    )
    preflightChecks = Self.makePreflightChecks(root: rootURL)
  }

  func refreshSelectedPlan() async {
    await refreshInstallationPlan(for: selectedModelKind)
  }

  func refreshInstallationPlan(for modelKind: ModelKind) async {
    let key = modelKind.rawValue
    guard installationBytesByModel[key] == nil, !planningModelKinds.contains(key) else { return }
    planningModelKinds.insert(key)
    installationPlanErrors.removeValue(forKey: key)
    do {
      let bytes = try await ModelPackages.package(for: modelKind).installedBytes()
      planningModelKinds.remove(key)
      installationBytesByModel[key] = bytes
    } catch {
      planningModelKinds.remove(key)
      installationPlanErrors[key] = String(describing: error)
      if selectedModelKind == modelKind {
        message = L10n.string(
          "The model installation information could not be loaded. Check the network and try again.\n%@",
          String(describing: error)
        )
      }
    }
    refreshPreflight()
  }

  func resumeDownloadIfNeeded() {
    guard defaults.bool(forKey: Self.activeDownloadPreference), !isBusy else { return }
    let modelKind =
      defaults.string(forKey: Self.activeDownloadModelKindPreference)
      .flatMap(ModelKind.init(rawValue:)) ?? selectedModelKind
    let saved = defaults.string(forKey: Self.activeDestinationPreference)
    let destination =
      saved.map { URL(fileURLWithPath: $0, isDirectory: true) }
      ?? defaultDownloadDestination(for: modelKind)
    guard FileManager.default.fileExists(atPath: destination.appendingPathExtension("partial").path)
    else {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      return
    }
    Task { [weak self] in
      guard let self else { return }
      await self.refreshInstallationPlan(for: modelKind)
      self.startDownload(for: modelKind, to: destination)
    }
  }

  func startDownload(to destination: URL? = nil) {
    startDownload(for: selectedModelKind, to: destination)
  }

  func startDownload(for modelKind: ModelKind, to destination: URL? = nil) {
    guard !isBusy else { return }
    refreshPreflight()
    if let block = downloadBlock(for: modelKind) {
      message = block.message
      return
    }
    let destination = destination ?? downloadDestination(for: modelKind)
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      message = L10n.string(
        "A model already exists in this location. Verify and repair the existing model first.")
      return
    }
    defaults.set(true, forKey: Self.activeDownloadPreference)
    defaults.set(destination.path, forKey: Self.activeDestinationPreference)
    defaults.set(modelKind.rawValue, forKey: Self.activeDownloadModelKindPreference)
    downloadModelKind = modelKind
    operationPhase = .preparingDownload
    operationProgress = nil
    downloadStart = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performDownload(
        to: destination,
        modelKind: modelKind
      )
    }
  }

  func startVerification(_ model: InstalledModelInfo) {
    guard !isBusy else { return }
    downloadModelKind = nil
    operationPhase = .verifying
    operationProgress = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performVerification(model.url)
    }
  }

  func startRepair(_ model: InstalledModelInfo) {
    guard !isBusy else { return }
    downloadModelKind = nil
    defaults.set(true, forKey: Self.activeDownloadPreference)
    defaults.set(model.url.path, forKey: Self.activeDestinationPreference)
    operationPhase = .verifying
    operationProgress = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performRepair(model.url)
    }
  }

  func startDSparkInstallation(_ model: InstalledModelInfo) {
    guard !isBusy, !model.hasDSpark, model.modelKind.descriptor.supports("dspark") else { return }
    downloadModelKind = nil
    operationPhase = .installingDSpark
    operationProgress = nil
    downloadStart = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performDSparkInstallation(model.url)
    }
  }

  func startMTPInstallation(_ model: InstalledModelInfo) {
    guard !isBusy, !model.hasMTP, model.modelKind.descriptor.supports("mtp") else { return }
    refreshPreflight()
    if let block = mtpDownloadBlock(for: model) {
      message = block.message
      return
    }
    downloadModelKind = model.modelKind
    operationPhase = .installingMTP
    operationProgress = nil
    downloadStart = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performMTPInstallation(model.url)
    }
  }

  func removeDSpark(_ model: InstalledModelInfo) {
    guard !isBusy, model.hasDSpark else { return }
    do {
      _ = try InstalledModel.removeDSpark(at: model.url)
      Task { [weak self] in
        await self?.scan()
        self?.message = L10n.string("DSpark was removed.")
      }
    } catch {
      message = L10n.string("DSpark could not be removed. %@", String(describing: error))
    }
  }

  func reinstall(_ url: URL) {
    guard !isBusy else { return }
    refreshPreflight()
    if let block = downloadBlock(for: selectedModelKind) {
      message = block.message
      return
    }
    do {
      var trashedURL: NSURL?
      try FileManager.default.trashItem(at: url, resultingItemURL: &trashedURL)
      startDownload(to: url)
    } catch {
      message = L10n.string(
        "The damaged model could not be moved to Trash. Check the folder permissions.")
    }
  }

  func cancelOperation() {
    guard isBusy else { return }
    defaults.set(false, forKey: Self.activeDownloadPreference)
    operationPhase = .cancelling
    operationTask?.cancel()
  }

  func reveal(_ url: URL) {
    NSWorkspace.shared.activateFileViewerSelecting([url])
  }

  private func defaultDownloadDestination(for modelKind: ModelKind) -> URL {
    let name = modelKind.descriptor.directoryName
    return rootURL.appending(path: name, directoryHint: .isDirectory)
  }

  private func partialDownloadURL(for modelKind: ModelKind) -> URL {
    let activeModelKind =
      defaults.string(forKey: Self.activeDownloadModelKindPreference)
      .flatMap(ModelKind.init(rawValue:)) ?? selectedModelKind
    let saved = defaults.string(forKey: Self.activeDestinationPreference)
    let destination: URL
    if activeModelKind == modelKind, let saved {
      destination = URL(fileURLWithPath: saved, isDirectory: true)
    } else {
      destination = defaultDownloadDestination(for: modelKind)
    }
    return destination.appendingPathExtension("partial")
  }

  private func downloadDestination(for modelKind: ModelKind) -> URL {
    hasPartialDownload(for: modelKind)
      ? partialDownloadURL(for: modelKind).deletingPathExtension()
      : defaultDownloadDestination(for: modelKind)
  }

  private func mtpInstallationPartialURL(for model: InstalledModelInfo) -> URL {
    model.url.deletingLastPathComponent()
      .appendingPathComponent(model.url.lastPathComponent + ".mtp-install")
      .appendingPathExtension("partial")
  }

  private func performDownload(
    to destination: URL,
    modelKind: ModelKind
  ) async {
    do {
      let package = ModelPackages.package(for: modelKind)
      let phase: ModelOperationPhase = package.installationLabel.map(ModelOperationPhase.installingArtifact) ?? .downloading
      _ = try await package.install(to: destination) { [weak self] progress in
        Task { @MainActor in self?.updateRepackProgress(progress, phase: phase) }
      }
      let needsAudit = !package.verifiesInstallation
      try Task.checkCancellation()
      let issues = needsAudit ? try await audit(destination).issues : []
      verificationModelPath = destination.path
      verificationIssues = issues
      let resultMessage =
        issues.isEmpty
        ? L10n.string("The model is installed and passed complete verification.")
        : L10n.string(
          "The model is installed, but %lld files failed verification.",
          Int64(issues.count))
      defaults.set(false, forKey: Self.activeDownloadPreference)
      await scan()
      message = resultMessage
    } catch is CancellationError {
      message = L10n.string("The download stopped. The app kept the progress.")
    } catch {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      message = L10n.string(
        "The model could not be downloaded. Check the network and try again.\n%@",
        String(describing: error))
    }
    finishOperation()
  }

  private func performVerification(_ url: URL) async {
    do {
      let verification = try await audit(url)
      verificationModelPath = url.path
      verificationIssues = verification.issues
      message =
        verification.isValid
        ? L10n.string("The model passed complete verification.")
        : L10n.string("%lld model files need repair.", Int64(verification.issues.count))
    } catch is CancellationError {
      message = L10n.string("Verification stopped.")
    } catch {
      message = L10n.string("The model could not be verified. %@", String(describing: error))
    }
    finishOperation()
  }

  private func performRepair(_ url: URL) async {
    do {
      let verification = try await audit(url)
      verificationModelPath = url.path
      verificationIssues = verification.issues
      guard !verification.isValid else {
        message = L10n.string("The model does not need repair.")
        defaults.set(false, forKey: Self.activeDownloadPreference)
        finishOperation()
        return
      }
      operationPhase = .preparingRepair
      operationProgress = nil
      downloadStart = nil
      let invalidFiles = Set(verification.issues.map(\.path))
      let modelKind = verification.manifest.modelKind ?? .deepSeekV4
      let package = ModelPackages.package(for: modelKind)
      let phase: ModelOperationPhase = package.installationLabel.map(ModelOperationPhase.installingArtifact) ?? .repairing
      _ = try await package.repair(at: url, invalidFiles: invalidFiles) { [weak self] progress in
        Task { @MainActor in self?.updateRepackProgress(progress, phase: phase) }
      }
      let needsAudit = !package.verifiesInstallation
      try Task.checkCancellation()
      let issues = needsAudit ? try await audit(url).issues : []
      verificationIssues = issues
      let resultMessage =
        issues.isEmpty
        ? L10n.string("The model was repaired and passed complete verification.")
        : L10n.string("%lld files still need repair.", Int64(issues.count))
      defaults.set(false, forKey: Self.activeDownloadPreference)
      await scan()
      message = resultMessage
    } catch is CancellationError {
      await scan()
      message = L10n.string("Repair stopped. The app kept the progress.")
    } catch {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      message = L10n.string(
        "The model could not be repaired. Check the network and storage.\n%@",
        String(describing: error))
      await scan()
    }
    finishOperation()
  }

  private func performDSparkInstallation(_ url: URL) async {
    do {
      let progress: @Sendable (RepackProgress) -> Void = { [weak self] progress in
        Task { @MainActor in self?.updateRepackProgress(progress, phase: .installingDSpark) }
      }
      let manifest = try InstalledModel.loadManifest(at: url)
      if manifest.modelKind == .deepSeekV41 {
        _ = try await DeepSeekV41Checkpoint().installDSpark(at: url, progress: progress)
      } else {
        _ = try await DeepSeekV4Checkpoint().installDSpark(at: url, progress: progress)
      }
      try Task.checkCancellation()
      let verification = try await audit(url)
      verificationModelPath = url.path
      verificationIssues = verification.issues
      await scan()
      message =
        verification.isValid
        ? L10n.string("DSpark is installed and ready.")
        : L10n.string("DSpark installation did not pass verification.")
    } catch is CancellationError {
      await scan()
      message = L10n.string("DSpark installation stopped. The app kept the progress.")
    } catch {
      await scan()
      message = L10n.string(
        "DSpark could not be installed. Check the network and storage.\n%@",
        String(describing: error)
      )
    }
    finishOperation()
  }

  private func performMTPInstallation(_ url: URL) async {
    do {
      _ = try await QwenFlashNextCheckpoint().installMTP(at: url) { [weak self] progress in
        Task { @MainActor in
          self?.updateRepackProgress(progress, phase: .installingMTP)
        }
      }
      try Task.checkCancellation()
      await scan()
      message = L10n.string("MTP is installed and ready.")
    } catch is CancellationError {
      await scan()
      message = L10n.string("MTP installation stopped. The app kept the progress.")
    } catch {
      await scan()
      message = L10n.string(
        "MTP could not be installed. Check the network and storage.\n%@",
        String(describing: error)
      )
    }
    finishOperation()
  }

  private func audit(_ url: URL) async throws -> InstalledModelVerification {
    operationPhase = .verifying
    operationProgress = nil
    let worker = Task.detached(priority: .utility) { [weak self] in
      try InstalledModel.audit(at: url) { progress in
        Task { @MainActor in
          self?.operationProgress = ModelOperationProgress(
            completedBytes: progress.checkedBytes,
            totalBytes: progress.totalBytes,
            bytesPerSecond: nil,
            estimatedSecondsRemaining: nil
          )
        }
      }
    }
    return try await withTaskCancellationHandler {
      try await worker.value
    } onCancel: {
      worker.cancel()
    }
  }

  private func updateRepackProgress(_ progress: RepackProgress, phase: ModelOperationPhase) {
    if operationPhase != phase { operationPhase = phase }
    let now = ContinuousClock.now
    if progress.downloadedBytes > 0, downloadStart == nil { downloadStart = now }
    let speed = downloadStart.map {
      let elapsed = Self.seconds(from: $0.duration(to: now))
      return elapsed > 0 ? Double(progress.downloadedBytes) / elapsed : 0
    }
    let remaining =
      progress.totalBytes > progress.copiedBytes
      ? progress.totalBytes - progress.copiedBytes : 0
    operationProgress = ModelOperationProgress(
      completedBytes: progress.copiedBytes,
      totalBytes: progress.totalBytes,
      bytesPerSecond: speed,
      estimatedSecondsRemaining: speed.flatMap { $0 > 0 ? Double(remaining) / $0 : nil }
    )
  }

  private func finishOperation() {
    operationTask = nil
    operationPhase = .idle
    operationProgress = nil
    downloadModelKind = nil
    downloadStart = nil
    refreshPreflight()
  }

  nonisolated private static func makePreflightChecks(root: URL) -> [PreflightCheck] {
    #if arch(arm64)
      let architecture = PreflightCheck(
        id: "architecture", title: L10n.string("Apple Silicon"),
        detail: L10n.string("This Mac uses Apple Silicon."),
        status: .passed, blocksDownload: true)
    #else
      let architecture = PreflightCheck(
        id: "architecture", title: L10n.string("Apple Silicon"),
        detail: L10n.string("This runtime does not support Intel Mac."),
        status: .failed, blocksDownload: true)
    #endif

    let memory = ProcessInfo.processInfo.physicalMemory
    let hasRecommendedMemory = memory >= recommendedMemoryBytes
    let memoryCheck = PreflightCheck(
      id: "memory",
      title: L10n.string("Memory"),
      detail: hasRecommendedMemory
        ? L10n.string("This Mac has at least 64 GiB of memory.")
        : L10n.string(
          "This Mac has less than 64 GiB of memory. Performance or stability may be reduced."),
      status: hasRecommendedMemory ? .passed : .warning,
      blocksDownload: false
    )

    let isInternal =
      (try? root.resourceValues(forKeys: [.volumeIsInternalKey]).volumeIsInternal)
      ?? nil
    let storageTypeCheck = PreflightCheck(
      id: "ssd",
      title: L10n.string("High-speed SSD"),
      detail: isInternal == false
        ? L10n.string(
          "Make sure that the external disk is a high-speed SSD. A slow disk reduces generation speed."
        )
        : L10n.string("Use a high-speed SSD."),
      status: isInternal == false ? .warning : .passed,
      blocksDownload: false
    )
    return [architecture, memoryCheck, storageTypeCheck]
  }

  nonisolated private static func inspectModelFolder(_ root: URL) -> (
    isWritable: Bool, availableBytes: UInt64?
  ) {
    let fileManager = FileManager.default
    let probe = root.appending(path: ".write-check-\(UUID().uuidString)")
    let isWritable: Bool
    do {
      try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
      try Data().write(to: probe, options: .atomic)
      try fileManager.removeItem(at: probe)
      isWritable = true
    } catch {
      try? fileManager.removeItem(at: probe)
      isWritable = false
    }
    let available = try? root.resourceValues(
      forKeys: [.volumeAvailableCapacityForImportantUsageKey]
    ).volumeAvailableCapacityForImportantUsage
    let availableBytes = available.flatMap { $0 >= 0 ? UInt64($0) : nil }
    return (isWritable, availableBytes)
  }

  nonisolated static func storageDownloadBlock(
    requiredBytes: UInt64,
    availableBytes: UInt64?
  ) -> ModelDownloadBlock? {
    guard let availableBytes else { return .storageUnavailable }
    guard availableBytes >= requiredBytes else {
      return .insufficientStorage(
        requiredBytes: requiredBytes,
        availableBytes: availableBytes
      )
    }
    return nil
  }

  nonisolated private static func allocatedBytes(at root: URL) -> UInt64 {
    guard
      let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey],
        options: [.skipsHiddenFiles]
      )
    else { return 0 }
    var total: UInt64 = 0
    for case let url as URL in enumerator {
      let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey])
      guard values?.isRegularFile == true, let size = values?.totalFileAllocatedSize else {
        continue
      }
      let sum = total.addingReportingOverflow(UInt64(size))
      if sum.overflow { return UInt64.max }
      total = sum.partialValue
    }
    return total
  }

  nonisolated private static func seconds(from duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }
}
