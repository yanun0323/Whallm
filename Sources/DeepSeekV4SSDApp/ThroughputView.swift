import Foundation
import SwiftUI
import AppKit

enum BenchmarkContext: String, CaseIterable, Codable, Sendable {
  case code, novel

  var title: String { self == .code ? "Code" : "Novel" }
  var detail: String {
    self == .code ? "Whallm source code · bundled, without repetition"
      : "Moby-Dick · bundled English novel, without repetition"
  }
}

struct ThroughputResult: Codable, Identifiable, Sendable {
  var id: Int { contextTokens }
  let model: String
  let slots: Int?
  let benchmarkContext: BenchmarkContext
  let corpusSha256: String
  let contextTokens: Int
  let generationTokens: Int
  let generationLimit: Int
  let ttftMs: Double
  let tpotMs: Double?
  let prefillTps: Double
  let decodeTps: Double
  let elapsedSeconds: Double
  let throughputTps: Double
  let peakAppMemoryBytes: Double?
  let memoryScope: String?
  let outputTokenSha256: String
  let promptCacheReusedTokens: Int
  let finishReason: String?
  // Older servers did not report sampling settings; keep those values unknown.
  var temperature: Double? = nil
  var seed: UInt32? = nil
  var topP: Double? = nil
  var topK: Int? = nil
  var minP: Double? = nil
  var presencePenalty: Double? = nil
  var repetitionPenalty: Double? = nil
  var diagnostics: ThroughputDiagnostics? = nil

  static let columns = ["Input / Output", "TTFT (ms)", "TPOT (ms)", "PP tok/s", "TG tok/s", "Total (s)", "Throughput", "Peak Memory"]

  var cells: [String] {
    ["\(contextTokens) / \(generationTokens)", Self.number(ttftMs), Self.number(tpotMs),
     Self.number(prefillTps), Self.number(decodeTps), Self.number(elapsedSeconds),
     Self.number(throughputTps), peakAppMemoryBytes.map {
       String(format: "%.2f GiB", locale: Locale(identifier: "en_US_POSIX"), $0 / 1_073_741_824)
     } ?? "—"]
  }

  private static func number(_ value: Double?) -> String {
    value.map { String(format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), $0) } ?? "—"
  }
}

enum ThroughputOutputFormat: String, CaseIterable {
  case plainText = "Plain text", json = "JSON", markdown = "Markdown table"

  func render(_ results: [ThroughputResult]) throws -> String {
    if self == .json {
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
      return String(decoding: try encoder.encode(results), as: UTF8.self)
    }
    let headers = ["Model", "Context", "Slots", "Output limit"] + ThroughputResult.columns + ["Temperature", "Seed"]
    let rows: [[String]] = results.map { result in
      let slots = result.slots.map { String($0) } ?? "—"
      let temperature = result.temperature.map { String($0) } ?? "—"
      let seed = result.seed.map { String($0) } ?? "—"
      let prefix: [String] = [result.model, result.benchmarkContext.title, slots, String(result.generationLimit)]
      return prefix + result.cells + [temperature, seed]
    }
    if self == .plainText {
      let table = [headers] + rows
      let widths = headers.indices.map { column in table.map { $0[column].count }.max() ?? 0 }
      return table.map { cells in
        cells.enumerated().map { column, cell in
          cell.padding(toLength: widths[column], withPad: " ", startingAt: 0)
        }.joined(separator: "  ").trimmingCharacters(in: CharacterSet.whitespaces)
      }.joined(separator: "\n")
    }
    func row(_ cells: [String]) -> String {
      "| " + cells.map {
        $0.replacingOccurrences(of: "\\", with: "\\\\")
          .replacingOccurrences(of: "|", with: "\\|")
          .replacingOccurrences(of: "\n", with: "<br>")
      }.joined(separator: " | ") + " |"
    }
    return ([row(headers), row(headers.map { _ in "---" })] + rows.map(row)).joined(separator: "\n")
  }
}

private struct ThroughputOutputText: NSViewRepresentable {
  let text: String

  func makeNSView(context: Context) -> NSScrollView {
    let scroll = NSScrollView()
    scroll.hasHorizontalScroller = true
    scroll.hasVerticalScroller = true
    scroll.drawsBackground = false
    let view = NSTextView()
    view.isEditable = false
    view.isSelectable = true
    view.isRichText = false
    view.drawsBackground = false
    view.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
    view.textColor = .labelColor
    view.textContainerInset = NSSize(width: 16, height: 16)
    view.isHorizontallyResizable = true
    view.isVerticallyResizable = true
    view.maxSize = NSSize(width: 1_000_000, height: 1_000_000)
    view.textContainer?.widthTracksTextView = false
    view.textContainer?.lineFragmentPadding = 0
    view.textContainer?.containerSize = view.maxSize
    scroll.documentView = view
    return scroll
  }

  func updateNSView(_ scroll: NSScrollView, context: Context) {
    guard let view = scroll.documentView as? NSTextView, view.string != text else { return }
    view.string = text
    view.sizeToFit()
  }
}

struct ThroughputEvent: Decodable, Sendable {
  struct Failure: Decodable, Sendable { let message: String }
  let phase: String?
  let generated: Int?
  let result: ThroughputResult?
  let error: Failure?

  static func decode(_ line: String) throws -> ThroughputEvent? {
    guard line.hasPrefix("data:") else { return nil }
    let data = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
    guard data != "[DONE]" else { return nil }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(Self.self, from: Data(data.utf8))
  }
}

private struct ThroughputError: LocalizedError {
  let message: String
  var errorDescription: String? { message }
}

enum ThroughputClient {
  static func run(
    configuration: ServerConfiguration, model: String, context: Int, generation: Int,
    benchmarkContext: BenchmarkContext,
    receive: @MainActor @escaping (ThroughputEvent) -> Void
  ) async throws -> ThroughputResult {
    guard let baseURL = configuration.baseURL else {
      throw ThroughputError(message: L10n.string("Invalid Base URL"))
    }
    var request = URLRequest(url: baseURL.appending(path: "api/benchmark/throughput"))
    request.httpMethod = "POST"
    request.timeoutInterval = 3_600
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if !configuration.apiKey.isEmpty {
      request.setValue("Bearer \(configuration.apiKey)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONSerialization.data(withJSONObject: [
      "model": model, "context_length": context, "generation_length": generation,
      "benchmark_context": benchmarkContext.rawValue,
    ])
    // A dedicated session closes the connection promptly when Cancel is pressed.
    let session = URLSession(configuration: .ephemeral)
    defer { session.invalidateAndCancel() }
    let (bytes, response) = try await session.bytes(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw ThroughputError(message: L10n.string("The server did not return an HTTP response."))
    }
    guard (200..<300).contains(http.statusCode) else {
      var data = Data()
      for try await byte in bytes { data.append(byte) }
      let detail = try? JSONDecoder().decode(ThroughputEvent.self, from: data)
      throw ThroughputError(message: detail?.error?.message
        ?? L10n.string("The server returned HTTP %lld.", Int64(http.statusCode)))
    }
    var result: ThroughputResult?
    for try await line in bytes.lines {
      try Task.checkCancellation()
      if line.trimmingCharacters(in: .whitespaces) == "data: [DONE]" {
        guard let result else {
          throw ThroughputError(message: L10n.string("The benchmark returned no result. Try again."))
        }
        return result
      }
      guard let event = try ThroughputEvent.decode(line) else { continue }
      if let error = event.error { throw ThroughputError(message: error.message) }
      if let value = event.result { result = value }
      await receive(event)
    }
    throw ThroughputError(message: L10n.string("The streaming response ended before [DONE]."))
  }
}

@MainActor
final class ThroughputSession: ObservableObject {
  #if WHALLM_LOCAL_BUILD
  static let dryRunAvailable = true
  #else
  static let dryRunAvailable = false
  #endif
  static let dryRunModel = "dry-run"
  static let generationLengths = [128, 512, 1024, 4096]
  static let contextLengths = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 204800]
  @Published var model = ""
  @Published var contextLengths: Set<Int> = [4096, 8192, 16384]
  @Published var generationLength = 128
  @Published var benchmarkContext: BenchmarkContext = .code
  @Published private(set) var results: [ThroughputResult] = []
  @Published private(set) var isRunning = false
  @Published private(set) var isUnloading = false
  @Published private(set) var phase = ""
  @Published private(set) var currentContext = 0
  @Published private(set) var generated = 0
  @Published private(set) var error: String?
  @Published private(set) var resultModel = ""
  private var task: Task<Void, Never>?

  func runDryRun() {
    #if WHALLM_LOCAL_BUILD
    guard !isRunning, model == Self.dryRunModel else { return }
    guard !contextLengths.isEmpty else {
      error = L10n.string("Select at least one context length.")
      return
    }
    error = nil
    resultModel = "Dry run (simulated)"
    currentContext = 0
    generated = 0
    results = contextLengths.sorted().map { length in
      let prefill = Double(length) / 2400
      let decode = Double(generationLength - 1) / 75
      let elapsed = prefill + decode
      return ThroughputResult(
        model: Self.dryRunModel, slots: 2304, benchmarkContext: benchmarkContext,
        corpusSha256: "dry-run", contextTokens: length,
        generationTokens: generationLength, generationLimit: generationLength,
        ttftMs: prefill * 1000, tpotMs: 1000 / 75, prefillTps: 2400, decodeTps: 75,
        elapsedSeconds: elapsed, throughputTps: Double(length + generationLength) / elapsed,
        peakAppMemoryBytes: 4_294_967_296 + Double(length) * 8192, memoryScope: "simulated",
        outputTokenSha256: "dry-run", promptCacheReusedTokens: 0, finishReason: "dry_run",
        temperature: 0, seed: 42
      )
    }
    phase = "Dry run complete · simulated results"
    #endif
  }

  func start(configuration: ServerConfiguration, server: ServerController, catalog: ModelCatalog) {
    guard !isRunning, !model.isEmpty, !contextLengths.isEmpty else { return }
    if model == Self.dryRunModel { runDryRun(); return }
    let model = model
    let generation = generationLength
    let context = benchmarkContext
    let modelID = catalog.models.first { $0.id == model || $0.alias == model }?.id ?? model
    start(prepare: {
      if !server.isActive { server.start(configuration, catalog: catalog) }
      let deadline = ContinuousClock.now.advanced(by: .seconds(60))
      while server.state == .starting, ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(100))
      }
      try Task.checkCancellation()
      guard server.state == .running else {
        if case .failed(let message) = server.state { throw ThroughputError(message: message) }
        throw ThroughputError(message: L10n.string("The server is not ready. Check Logs and try again."))
      }
      try await server.waitForModelConfigurationUpdates(modelID)
    }, run: { length, receive in
      try await ThroughputClient.run(
        configuration: configuration, model: model, context: length, generation: generation,
        benchmarkContext: context, receive: receive)
    }, unload: {
      try await server.unloadModelAfterBenchmark(modelID)
    })
  }

  // The lifecycle is shared by the real client and tests; cleanup outlives cancellation.
  func start(
    prepare: @MainActor @escaping () async throws -> Void,
    run: @MainActor @escaping (Int, @MainActor @escaping (ThroughputEvent) -> Void) async throws -> ThroughputResult,
    unload: @MainActor @escaping () async throws -> Void
  ) {
    guard !isRunning, !model.isEmpty, !contextLengths.isEmpty else { return }
    if model == Self.dryRunModel { runDryRun(); return }
    let lengths = contextLengths.sorted()
    results = []
    resultModel = model
    error = nil
    isRunning = true
    phase = "Starting server…"
    generated = 0
    currentContext = lengths[0]
    task = Task {
      defer { isRunning = false; task = nil }
      var requestedModel = false
      var finalPhase = "Benchmark complete"
      do {
        try await prepare()
        for length in lengths {
          try Task.checkCancellation()
          currentContext = length
          generated = 0
          phase = "Loading model…"
          requestedModel = true
          let result = try await run(length) { [weak self] event in
            guard let self else { return }
            if event.phase == "loading" { phase = "Loading model…" }
            if event.phase == "running" { phase = "Running benchmark…" }
            if let count = event.generated { generated = count }
          }
          try Task.checkCancellation()
          results.append(result)
        }
      } catch {
        if Task.isCancelled {
          finalPhase = "Benchmark cancelled"
        } else {
          self.error = error.localizedDescription
          finalPhase = "Benchmark failed"
        }
      }
      if requestedModel {
        isUnloading = true
        phase = "Unloading model…"
        // An unstructured task does not inherit the cancelled benchmark's cancellation.
        let cleanup = Task { @MainActor in try await unload() }
        do {
          try await cleanup.value
        } catch {
          let message = L10n.string("Could not unload the benchmark model: %@", error.localizedDescription)
          self.error = [self.error, message].compactMap { $0 }.joined(separator: "\n")
          finalPhase = "Benchmark failed"
        }
        isUnloading = false
      }
      phase = finalPhase
    }
  }

  func cancel() {
    guard !isUnloading else { return }
    task?.cancel()
  }

  func reportFailure(_ error: Error) {
    self.error = error.localizedDescription
    phase = "Benchmark failed"
  }
}

struct ThroughputView: View {
  let configuration: ServerConfiguration
  @ObservedObject var server: ServerController
  @ObservedObject var modelLibrary: ModelLibrary
  @ObservedObject var session: ThroughputSession
  let language: AppLanguage
  @State private var outputFormat: ThroughputOutputFormat = .plainText

  private func label(_ key: String) -> String { L10n.string(key, language: language) }
  private var models: [CatalogModel] {
    (try? modelLibrary.makeServerCatalog(powerSavingLimitGBps: configuration.powerSavingLimitGBps))?
      .availableModels ?? []
  }

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        VStack(alignment: .leading, spacing: 6) {
          Text(label("Configuration"))
            .font(.caption.weight(.semibold))
            .textCase(.uppercase)
            .tracking(0.4)
            .foregroundStyle(.secondary)

          VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
              HStack(spacing: 16) {
                settingLabel("Model", detail: session.model == ThroughputSession.dryRunModel
                  ? "Simulated results for UI debugging. No model is loaded."
                  : "Run automatically loads the selected model.")
                Spacer(minLength: 8)
                Picker(label("Model"), selection: $session.model) {
                  Text(label("Select a model…")).tag("")
                  if ThroughputSession.dryRunAvailable {
                    Text("Dry run").tag(ThroughputSession.dryRunModel)
                  }
                  ForEach(models) { model in Text(model.requestName).tag(model.id) }
                }
                .labelsHidden().fixedSize().frame(maxWidth: 280, alignment: .trailing)
              }
              .padding(.vertical, 12)
              Divider()
              HStack(spacing: 16) {
                settingLabel("Benchmark Context", detail: session.benchmarkContext.detail)
                Spacer(minLength: 8)
                Picker(label("Benchmark Context"), selection: $session.benchmarkContext) {
                  ForEach(BenchmarkContext.allCases, id: \.self) { context in
                    Text(label(context.title)).tag(context)
                  }
                }
                .labelsHidden().fixedSize().frame(width: 210, alignment: .trailing)
              }
              .padding(.vertical, 12)
              Divider()
              VStack(alignment: .leading, spacing: 8) {
                settingLabel("Context lengths", detail: "Input tokens per test. Select one or more lengths.")
                ViewThatFits(in: .horizontal) {
                  HStack(spacing: 6) { contextToggles }
                    .fixedSize()
                  LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 48, maximum: 64), spacing: 6)],
                    alignment: .leading, spacing: 6
                  ) { contextToggles }
                }
              }
              .padding(.vertical, 12)
              Divider()
              HStack(spacing: 16) {
                settingLabel("Generation length", detail: "Maximum output tokens per test.")
                Spacer(minLength: 8)
                Picker(label("Generation length"), selection: $session.generationLength) {
                  ForEach(ThroughputSession.generationLengths, id: \.self) { Text(String($0)).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize().frame(width: 210, alignment: .trailing)
              }
              .padding(.vertical, 12)
            }
            .disabled(session.isRunning)
            Divider()
            HStack(spacing: 16) {
              Text(label(session.model == ThroughputSession.dryRunModel
                ? "Simulated results for UI debugging. No model is loaded."
                : models.isEmpty
                ? "Install a model on the Model page to run a benchmark."
                : "One request at a time. Prompt reuse is off; model settings apply."))
                .font(.system(size: 13)).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
              Spacer(minLength: 8)
              if session.isRunning {
                Button(label("Cancel"), role: .cancel) { session.cancel() }
                  .disabled(session.isUnloading)
                  .keyboardShortcut(".", modifiers: .command)
              } else {
                Button { run() } label: {
                  Label(label(session.model == ThroughputSession.dryRunModel ? "Generate results" : "Run Benchmark"), systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(session.contextLengths.isEmpty || (session.model != ThroughputSession.dryRunModel
                  && (!models.contains { $0.id == session.model } || server.performance.generating
                    || server.modelAction != nil || server.state == .stopping || modelLibrary.isBusy)))
              }
            }
            .padding(.vertical, 12)
          }
          .padding(.horizontal, 16)
          .padding(.vertical, 4)
          .background(AppTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
        }

        if !session.phase.isEmpty {
          HStack(spacing: 12) {
            if session.isRunning { ProgressView().controlSize(.small) }
            Text(label(session.phase))
            Spacer()
            if session.isRunning {
              Text("\(session.currentContext / 1024)K · \(session.generated) / \(session.generationLength)")
                .monospacedDigit().foregroundStyle(.secondary)
            }
          }
          .padding(16)
          .background(AppTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
        }
        if let error = session.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
        if !session.results.isEmpty {
          results
          resultOutput
        }
      }
      .font(.system(size: 14))
      .controlSize(.regular)
      .buttonBorderShape(.roundedRectangle(radius: 5))
      .frame(maxWidth: 760)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 32)
      .padding(.vertical, 32)
    }
    .onAppear { selectAvailableModel() }
    .onChange(of: models) { selectAvailableModel() }
    .onChange(of: session.model) {
      if session.model == ThroughputSession.dryRunModel { session.runDryRun() }
    }
  }

  private func settingLabel(_ title: String, detail: String) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(label(title)).fontWeight(.semibold)
      Text(label(detail))
        .font(.system(size: 13)).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
  }

  private var contextToggles: some View {
    ForEach(ThroughputSession.contextLengths, id: \.self) { length in
      Toggle("\(length / 1024)K", isOn: Binding(
        get: { session.contextLengths.contains(length) },
        set: { selected in
          if selected { session.contextLengths.insert(length) }
          else { session.contextLengths.remove(length) }
        }
      ))
      .toggleStyle(.button)
      .font(.system(size: 13, weight: .medium))
      .accessibilityLabel(label("Context lengths") + ": \(length) tokens")
    }
  }

  private var results: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text(label("Single request results")).font(.headline)
      HStack {
        Text(label(session.resultModel))
        if let result = session.results.first {
          Text("· " + label(result.benchmarkContext.title))
          Text("· " + slotsSummary + " slots")
        }
        Spacer()
        if let result = session.results.first {
          Text(L10n.string("Output limit: %lld tokens", language: language, Int64(result.generationLimit)))
        }
      }
      .font(.callout).foregroundStyle(.secondary)
      GeometryReader { geometry in
        ScrollView(.horizontal) {
          Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 10) {
            GridRow {
              ForEach(ThroughputResult.columns, id: \.self) {
                Text(label($0)).font(.caption).foregroundStyle(.secondary)
                  .fixedSize().frame(maxWidth: .infinity, alignment: .leading).frame(height: 18)
              }
            }
            Divider().gridCellUnsizedAxes(.horizontal)
            ForEach(session.results) { result in
              GridRow {
                ForEach(Array(result.cells.enumerated()), id: \.offset) { _, value in
                  Text(value).fixedSize()
                    .frame(maxWidth: .infinity, alignment: .leading).frame(height: 24)
                }
              }
            }
          }
          .frame(width: max(700, geometry.size.width - 32))
          .monospacedDigit().textSelection(.enabled).padding(16)
        }
      }
      .frame(height: 61 + CGFloat(session.results.count) * 34 + 12)
      .background(AppTheme.cardBackground, in: RoundedRectangle(cornerRadius: AppTheme.cardRadius))
      ForEach(session.results) { result in
        if let diagnostics = result.diagnostics {
          VStack(alignment: .leading, spacing: 4) {
            Text("\(result.contextTokens) tokens").font(.caption).foregroundStyle(.secondary)
            ThroughputDiagnosticsView(diagnostics: diagnostics, language: language)
          }
        }
      }
      Text(label("Throughput uses temperature 0 and seed 42. Other model settings still apply. Fixed sampling does not guarantee identical output across acceleration settings."))
        .font(.caption).foregroundStyle(.secondary)
      Text(label("TTFT: first token · TPOT: time per output token · PP: input speed · TG: output speed. Throughput counts input + output tokens per second. Peak Memory samples macOS physical footprint about every 10 ms during loading and generation: Whallm + its inference process, or the inference process alone for a standalone server. Brief peaks may be missed; — means unavailable."))
        .font(.caption).foregroundStyle(.secondary)
      Text(label("Output may end early. Loading time is excluded; no extra warm-up is performed."))
        .font(.caption).foregroundStyle(.secondary)
    }
  }

  private var slotsSummary: String {
    Set(session.results.map { $0.slots.map(String.init) ?? "—" }).sorted().joined(separator: " / ")
  }

  private var resultOutput: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text(label("Result output")).font(.headline)
        Spacer()
        Picker(label("Output format"), selection: $outputFormat) {
          ForEach(ThroughputOutputFormat.allCases, id: \.self) { format in
            Text(label(format.rawValue)).tag(format)
          }
        }
        .labelsHidden().pickerStyle(.segmented).fixedSize()
      }
      ThroughputOutputText(text: resultOutputText)
      .frame(height: 240)
      .background(AppTheme.cardBackground, in: RoundedRectangle(cornerRadius: AppTheme.cardRadius))
    }
  }

  private var resultOutputText: String {
    do { return try outputFormat.render(session.results) }
    catch { return error.localizedDescription }
  }

  private func selectAvailableModel() {
    guard !session.isRunning,
      !(ThroughputSession.dryRunAvailable && session.model == ThroughputSession.dryRunModel),
      !models.contains(where: { $0.id == session.model }) else { return }
    session.model = models.first?.id ?? ""
  }

  private func run() {
    if session.model == ThroughputSession.dryRunModel {
      session.runDryRun()
      return
    }
    do {
      let catalog = try modelLibrary.makeServerCatalog(powerSavingLimitGBps: configuration.powerSavingLimitGBps)
      session.start(configuration: configuration, server: server, catalog: catalog)
    } catch {
      session.reportFailure(error)
    }
  }
}
