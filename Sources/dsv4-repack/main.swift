import DeepSeekRepack
import Foundation

@main
struct CLI {
  static func main() async {
    do {
      try await run(Array(CommandLine.arguments.dropFirst()))
    } catch {
      FileHandle.standardError.write(Data("error: \(error)\n".utf8))
      Foundation.exit(1)
    }
  }

  private static func run(_ arguments: [String]) async throws {
    guard let command = arguments.first else {
      printUsage()
      return
    }
    switch command {
    case "inspect":
      let plan = try await makePlan(model: modelSelection(in: arguments))
      printSummary(plan)
    case "plan":
      let output = try value(after: "--output", in: arguments)
      let plan = try await makePlan(model: modelSelection(in: arguments))
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let url = URL(fileURLWithPath: output).standardizedFileURL
      guard !FileManager.default.fileExists(atPath: url.path) else {
        throw RepackError.destinationExists(url.path)
      }
      try encoder.encode(plan).write(to: url, options: .atomic)
      printSummary(plan)
      print("plan: \(url.path)")
    case "repack":
      let output = try value(after: "--output", in: arguments)
      let printer = ProgressPrinter()
      let outputURL = URL(fileURLWithPath: output)
      let progress: @Sendable (RepackProgress) -> Void = { printer.update($0) }
      let plan: RepackPlan
      if let planPath = optionalValue(after: "--plan", in: arguments) {
        let data = try Data(contentsOf: URL(fileURLWithPath: planPath))
        plan = try JSONDecoder().decode(RepackPlan.self, from: data)
      } else {
        plan = try await makePlan(model: modelSelection(in: arguments))
      }
      let manifest = try await ModelPackages.package(for: plan.modelKind ?? .deepSeekV4)
        .repack(plan: plan, to: outputURL, progress: progress)
      print("installed: \(output)")
      print("files: \(manifest.files.count)")
    case "verify":
      let model = try value(after: "--model", in: arguments)
      let manifest = try InstalledModel.verify(at: URL(fileURLWithPath: model))
      print("verified: \(model)")
      print("files: \(manifest.files.count)")
    case "install-dspark":
      let model = try value(after: "--model", in: arguments)
      let printer = ProgressPrinter()
      let url = URL(fileURLWithPath: model)
      let installed = try InstalledModel.loadManifest(at: url)
      let manifest: InstalledManifest
      if let kind = installed.modelKind, kind.isDeepSeekV41 {
        manifest = try await DeepSeekV41Checkpoint(kind: kind).installDSpark(at: url) { printer.update($0) }
      } else {
        manifest = try await DeepSeekV4Checkpoint().installDSpark(at: url) { printer.update($0) }
      }
      print("DSpark installed: \(model)")
      print("files: \(manifest.files.count)")
    case "install-mtp":
      let model = try value(after: "--model", in: arguments)
      let printer = ProgressPrinter()
      let manifest = try await QwenFlashNextCheckpoint().installMTP(
        at: URL(fileURLWithPath: model)
      ) { printer.update($0) }
      print("MTP installed: \(model)")
      print("files: \(manifest.files.count)")
    case "benchmark":
      let model = try value(after: "--model", in: arguments)
      let samples = try integer(after: "--samples", in: arguments, default: 32)
      let results = try ExpertIOBenchmark.run(
        at: URL(fileURLWithPath: model),
        samples: samples
      )
      printBenchmark(results)
    case "help", "--help", "-h":
      printUsage()
    default:
      throw RepackError.invalidPlan("unknown command \(command)")
    }
  }

  private static func value(after option: String, in arguments: [String]) throws -> String {
    guard let index = arguments.firstIndex(of: option), arguments.indices.contains(index + 1) else {
      throw RepackError.invalidPlan("missing \(option)")
    }
    return arguments[index + 1]
  }

  private static func modelSelection(in arguments: [String]) throws -> ModelKind {
    guard let value = optionalValue(after: "--model", in: arguments) else {
      return .deepSeekV4
    }
    switch value {
    case "deepseek-v4", "deepseek-v4-flash-0731": return .deepSeekV4
    case "deepseek-v4.1", "deepseek-v4.1-flash", "deepseek-flash": return .deepSeekV41
    case "deepseek-v4.1-1017", "deepseek-v4.1-flash-1017": return .deepSeekV41_1017
    case "qwen3.8-flash-next": return .qwen3_8FlashNext
    default: throw RepackError.invalidPlan("unknown model \(value)")
    }
  }

  private static func makePlan(model: ModelKind) async throws -> RepackPlan {
    try await ModelPackages.package(for: model).makeRepackPlan()
  }

  private static func optionalValue(after option: String, in arguments: [String]) -> String? {
    guard let index = arguments.firstIndex(of: option), arguments.indices.contains(index + 1)
    else {
      return nil
    }
    return arguments[index + 1]
  }

  private static func integer(
    after option: String,
    in arguments: [String],
    default defaultValue: Int
  ) throws -> Int {
    guard let index = arguments.firstIndex(of: option) else { return defaultValue }
    guard arguments.indices.contains(index + 1), let value = Int(arguments[index + 1]) else {
      throw RepackError.invalidPlan("invalid \(option)")
    }
    return value
  }

  private static func printBenchmark(_ results: [ExpertIOBenchmarkResult]) {
    print("mode\tworkers\treads\tGiB/s\tmean ms")
    for result in results {
      print(
        "\(result.cacheMode.rawValue)\t\(result.concurrency)\t\(result.reads)\t"
          + String(format: "%.2f\t%.2f", result.gibibytesPerSecond, result.meanMilliseconds)
      )
    }
  }

  private static func printSummary(_ plan: RepackPlan) {
    print("model: \(plan.modelID)")
    print("revision: \(plan.revision)")
    print("layers: \(plan.layerCount)")
    print("experts per layer: \(plan.expertCount)")
    print("expert blob: \(plan.expertBlobSize) bytes")
    print("checkpoint tensor bytes: \(plan.checkpointTensorBytes)")
    print("installed weight bytes: \(plan.installedBytes)")
    print("copy operations: \(plan.copies.count)")
  }

  private static func printUsage() {
    print(
      """
      Usage:
        dsv4-repack inspect [--model deepseek-v4.1|deepseek-v4.1-1017|qwen3.8-flash-next]
        dsv4-repack plan [--model deepseek-v4.1|deepseek-v4.1-1017|qwen3.8-flash-next] --output plan.json
        dsv4-repack repack [--model deepseek-v4.1|deepseek-v4.1-1017|qwen3.8-flash-next] --output MODEL.dsv4 [--plan plan.json]
        dsv4-repack verify --model deepseek-v4-flash-0731.dsv4
        dsv4-repack install-dspark --model deepseek-v4-flash-0731.dsv4
        dsv4-repack install-mtp --model qwen3.8-flash-next.dsv4
        dsv4-repack benchmark --model deepseek-v4-flash-0731.dsv4 [--samples 32]
      """)
  }
}

private final class ProgressPrinter: @unchecked Sendable {
  private let lock = NSLock()
  private var lastPercent = -1

  func update(_ progress: RepackProgress) {
    let percent =
      progress.totalBytes == 0 ? 100 : Int(progress.copiedBytes * 100 / progress.totalBytes)
    lock.lock()
    defer { lock.unlock() }
    guard percent != lastPercent else { return }
    lastPercent = percent
    FileHandle.standardOutput.write(Data("repack: \(percent)%\n".utf8))
  }
}
