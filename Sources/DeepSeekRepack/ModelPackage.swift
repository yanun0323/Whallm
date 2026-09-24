import Foundation

public struct ModelPackageDescriptor: Decodable, Sendable {
  public struct Defaults: Decodable, Sendable {
    public let slots: Int
    public let promptCacheEntries: Int
    public let bf16KVCache: Bool
    public let maxTokens: Int
    public let temperature: Double
    public let topP: Double
    public let topK: Int
  }

  public let kind: String
  public let displayName: String
  public let kindLabel: String
  public let assistantName: String
  public let apiModelID: String
  public let owner: String
  public let checkpointModelID: String
  public let checkpointRevision: String
  public let manifestVersion: Int
  public let directoryName: String
  public let requiredPaths: [String]
  public let features: Set<String>
  public let editableSettings: Set<String>
  public let defaults: Defaults
  public let automaticMemoryGiB: Int
  public let prefillThreshold: Int?

  public func supports(_ feature: String) -> Bool { features.contains(feature) }
}

public enum ModelPackages {
  public static let descriptors: [ModelPackageDescriptor] = {
    // The packaged App must not depend on the build machine's module bundle.
    let url = Bundle.main.url(forResource: "ModelPackages", withExtension: "json")
      ?? Bundle.module.url(forResource: "ModelPackages", withExtension: "json")
    do {
      guard let url else { throw RepackError.invalidPlan("model package catalog is missing") }
      return try decodeDescriptors(Data(contentsOf: url))
    } catch {
      preconditionFailure("Cannot load built-in model packages: \(error)")
    }
  }()

  static func decodeDescriptors(_ data: Data) throws -> [ModelPackageDescriptor] {
    struct Catalog: Decodable {
      let version: Int
      let models: [ModelPackageDescriptor]
    }
    let catalog = try JSONDecoder().decode(Catalog.self, from: data)
    let models = catalog.models
    guard catalog.version == 1, !models.isEmpty,
      Set(models.map(\.kind)).count == models.count,
      Set(models.map(\.apiModelID)).count == models.count
    else { throw RepackError.invalidPlan("invalid model package catalog") }
    for model in models {
      guard !model.kind.isEmpty, !model.apiModelID.isEmpty,
        model.manifestVersion > 0, model.automaticMemoryGiB >= 0,
        model.prefillThreshold.map({ $0 > 0 }) ?? true,
        model.checkpointRevision.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil,
        model.defaults.slots >= 6, model.defaults.promptCacheEntries >= 1,
        (1...272_000).contains(model.defaults.maxTokens),
        (0...248_320).contains(model.defaults.topK),
        (0...2).contains(model.defaults.temperature),
        (0.000001...1).contains(model.defaults.topP),
        ([model.directoryName] + model.requiredPaths).allSatisfy({ path in
          !path.isEmpty && !path.hasPrefix("/") && !path.split(separator: "/").contains("..")
        })
      else { throw RepackError.invalidPlan("invalid model package descriptor") }
    }
    return models
  }

  public static func package(for kind: ModelKind) -> any ModelPackage {
    guard let package = builtins[kind.rawValue] else {
      preconditionFailure("Model package implementation is missing: \(kind.rawValue)")
    }
    return package
  }

  static func package(for manifest: InstalledManifest) throws -> any ModelPackage {
    let legacy: [Int: ModelKind] = [1: .deepSeekV4, 2: .qwen3_8FlashNext, 3: .deepSeekV41]
    guard let kind = manifest.modelKind ?? legacy[manifest.formatVersion],
      kind.descriptor.manifestVersion == manifest.formatVersion
    else { throw RepackError.incompatibleModel("installed manifest has an unsupported model contract") }
    return package(for: kind)
  }

  private static let builtins: [String: any ModelPackage] = [
    "deepseek-v4": DeepSeekV4Package(),
    "deepseek-v4.1": DeepSeekV41Package(),
    "qwen3.8-flash-next": QwenPackage(),
    "mimo-v2.6-flash-rl": MiMoPackage(),
  ]

  static func companions(for kind: ModelKind) -> [CompanionFile] {
    package(for: kind).companions
  }

}

public protocol ModelPackage: Sendable {
  var verifiesInstallation: Bool { get }
  var installationLabel: String? { get }
  var companions: [CompanionFile] { get }
  func installedBytes() async throws -> UInt64
  func makeRepackPlan() async throws -> RepackPlan
  func repack(plan: RepackPlan, to output: URL, progress: RepackProgressHandler?) async throws
    -> InstalledManifest
  func install(to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest
  func repair(at output: URL, invalidFiles: Set<String>, progress: RepackProgressHandler?)
    async throws -> InstalledManifest
  func validate(_ manifest: InstalledManifest) throws -> InstalledManifest
}

public typealias RepackProgressHandler = @Sendable (RepackProgress) -> Void
