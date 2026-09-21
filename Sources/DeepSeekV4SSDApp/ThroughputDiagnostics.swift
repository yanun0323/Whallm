import Foundation
import SwiftUI

struct ThroughputDiagnostics: Codable, Sendable {
  let schemaVersion: Int
  let runtimeConfigJson: String
  let configSha256: String
  let sourceFilesSha256: String
  let throughFirstToken: [String: Double]?
  let afterFirstToken: [String: Double]?
  let requestTotal: [String: Double]?
  let initialResidentExperts: Int?
  let finalResidentExperts: Int?
  let processDiskBytesRead: Double?
  let decodeCacheEvalSeconds: Double?
  let notes: String
}

struct ThroughputDiagnosticsView: View {
  let diagnostics: ThroughputDiagnostics
  let language: AppLanguage

  private func text(_ key: String) -> String { L10n.string(key, language: language) }
  private func value(_ counters: [String: Double]?, _ key: String, scale: Double = 1) -> String {
    guard let number = counters?[key] else { return "—" }
    return String(format: "%.3f", locale: Locale(identifier: "en_US_POSIX"), number / scale)
  }

  var body: some View {
    DisclosureGroup(text("Runtime configuration and expert I/O")) {
      VStack(alignment: .leading, spacing: 8) {
        Text("\(text("Configuration hash")): \(diagnostics.configSha256.prefix(12)) · \(text("Source files hash")): \(diagnostics.sourceFilesSha256.prefix(12))")
          .font(.caption.monospaced())
        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 4) {
          GridRow {
            Text(text("Counter"))
            Text(text("Through first token"))
            Text(text("After first token"))
          }
          counter("Expert payload GiB", key: "bytes_read", scale: 1_073_741_824)
          counter("Expert wait seconds", key: "wait_seconds")
          counter("Expert cache hits", key: "hits")
          counter("Expert cache misses", key: "misses")
          counter("Prefill seeded experts", key: "prefill_seeded_experts")
          counter("Seeded experts used", key: "prefill_seed_hits")
          counter("Prefill read batches", key: "prefill_read_batches")
        }
        .font(.caption).monospacedDigit()
        Text(text("Expert bytes are logical payload, not SSD traffic. Wait counts consumer blocking, not total I/O service time. Asynchronous work may cross the first-token boundary."))
          .font(.caption).foregroundStyle(.secondary)
        Text(diagnostics.runtimeConfigJson)
          .font(.caption.monospaced()).textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
      }.frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
    }
    .accessibilityIdentifier("throughput-runtime-diagnostics")
  }

  private func counter(_ title: String, key: String, scale: Double = 1) -> some View {
    GridRow {
      Text(text(title))
      Text(value(diagnostics.throughFirstToken, key, scale: scale))
      Text(value(diagnostics.afterFirstToken, key, scale: scale))
    }
  }
}
