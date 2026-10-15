import AppKit
import SwiftUI
import XCTest

@testable import DeepSeekV4SSDApp

final class ServerLogFileTests: XCTestCase {
  private func temporaryURL() -> URL {
    FileManager.default.temporaryDirectory.appending(path: "ServerLogFileTests-\(UUID())")
      .appending(path: "server.log")
  }

  func testAppendPersistsAcrossInstancesAndRestrictsPermissions() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    try ServerLogFile(url: url).append("first\n")
    try ServerLogFile(url: url).append("第二筆 🐋\n")
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "first\n第二筆 🐋\n")
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
  }

  func testOverflowDropsOldestLinesAndKeepsNewestInOrder() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    let file = ServerLogFile(url: url, maximumBytes: 12)
    try file.append("aaaa\nbbbb\n")
    try file.append("cccc\n")
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "bbbb\ncccc\n")
    try file.append("dddd\n")
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "cccc\ndddd\n")
  }

  func testOversizedSingleLinePreservesUTF8AndExactByteLimit() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    let file = ServerLogFile(url: url, maximumBytes: 10)
    try file.append("old\n")
    try file.append("中文🐋中文🐋\n")
    let data = try Data(contentsOf: url)
    XCTAssertLessThanOrEqual(data.count, 10)
    XCTAssertEqual(String(data: data, encoding: .utf8), "文🐋\n")
    try file.append("1234567890")
    XCTAssertEqual(try Data(contentsOf: url).count, 10)
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "1234567890")
  }

  func testProductionLimitAndOversizedExistingFile() throws {
    XCTAssertEqual(ServerLogFile.maximumBytes, 10_000_000)
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    let file = ServerLogFile(url: url)
    try file.append(String(repeating: "x", count: 10_000_000))
    XCTAssertEqual(try Data(contentsOf: url).count, 10_000_000)
    try file.append("newest\n")
    let data = try Data(contentsOf: url)
    XCTAssertEqual(data.count, 10_000_000)
    XCTAssertTrue(data.suffix(7).elementsEqual(Data("newest\n".utf8)))
    try Data((String(repeating: "x", count: 10_000_100) + "\nlast\n").utf8).write(to: url)
    try file.append("")
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "last\n")
  }

  @MainActor
  func testControllerToggleWritesOnlyNewLogsAndKeepsScreenLimit() async throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    let controller = ServerController(logFileURL: url)
    controller.appendLog("not saved\n")
    XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    controller.setLogFileEnabled(true)
    controller.appendLog("saved\n")
    controller.setLogFileEnabled(false)
    controller.appendLog("also not saved\n")
    await drain(controller)
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "saved\n")
    controller.setLogFileEnabled(true)
    controller.appendLog(String(repeating: "a", count: 50_000))
    await drain(controller)
    XCTAssertEqual(controller.log.count, 40_000)
    XCTAssertEqual(try Data(contentsOf: url).count, 50_006)
    XCTAssertNil(controller.logFileError)
  }

  @MainActor
  func testWriteFailureIsVisibleWithoutStoppingServerAndCanRetry() async throws {
    let url = temporaryURL()
    let folder = url.deletingLastPathComponent()
    defer { try? FileManager.default.removeItem(at: folder) }
    try Data("not a directory".utf8).write(to: folder)
    let controller = ServerController(logFileURL: url)
    controller.setLogFileEnabled(true)
    await drain(controller)
    for _ in 0..<100 where controller.logFileError == nil {
      try await Task.sleep(for: .milliseconds(10))
    }
    XCTAssertNotNil(controller.logFileError)
    XCTAssertTrue(controller.log.contains("[ERROR]"))
    XCTAssertFalse(controller.isActive)
    try FileManager.default.removeItem(at: folder)
    controller.setLogFileEnabled(false)
    controller.setLogFileEnabled(true)
    controller.appendLog("retry\n")
    await drain(controller)
    XCTAssertNil(controller.logFileError)
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "retry\n")
  }

  @MainActor
  func testStartupFailureUsesSavedPreferenceAndFlushesToDisk() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    let controller = ServerController(logFileURL: url)
    let configuration = ServerConfiguration(runtimeDirectory: "/missing-whallm-runtime",
      pythonExecutable: "/missing-python", host: "127.0.0.1", port: 11434,
      logLevel: .error, saveLogsToFile: true, apiKey: "")
    controller.start(configuration, catalog: ModelCatalog(models: []))
    controller.flushLogFile()
    XCTAssertFalse(controller.log.isEmpty)
    XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), controller.log)
    guard case .failed = controller.state else { return XCTFail("Expected startup failure") }
  }

  @MainActor
  func testEventLevelsSafeErrorsAndSplitReadinessMessage() {
    let controller = ServerController(logFileURL: temporaryURL())
    controller.eventLogLevel = .error
    controller.recordEvent(.debug, "hidden debug")
    controller.recordEvent(.info, "hidden info")
    controller.recordFailure("Chat", error: NSError(domain: NSURLErrorDomain, code: -1001,
      userInfo: [NSLocalizedDescriptionKey: "private prompt and key"]), requestID: "test-request")
    XCTAssertFalse(controller.log.contains("hidden"))
    XCTAssertFalse(controller.log.contains("private prompt"))
    XCTAssertTrue(controller.log.contains("[ERROR] request=test-request Chat failed"))
    XCTAssertTrue(controller.log.contains("code=-1001"))
    let before = controller.log
    controller.received(Data("WHALLM_SERVER_".utf8))
    controller.received(Data("READY\n".utf8))
    XCTAssertEqual(controller.log, before, "Readiness control message is not a log")
    controller.recordStatusFailure("offline")
    controller.recordStatusFailure("offline")
    XCTAssertEqual(controller.log.components(separatedBy: "Status polling failed").count - 1, 1)
  }

  @MainActor
  func testClientAndVerificationFailuresReachSharedLog() async throws {
    let suite = "ServerLogFileTests-\(UUID())"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let controller = ServerController(logFileURL: temporaryURL())
    let chat = ChatSession(defaults: defaults, stream: { _, _, _, _, _, _, _, _ in
      throw URLError(.timedOut)
    })
    chat.logServer = controller
    XCTAssertTrue(chat.send(text: "private prompt", configuration: ServerConfiguration.load(defaults: defaults, apiKey: ""),
      model: "test", thinkingMode: "chat", language: .english))
    for _ in 0..<100 where chat.isSending { try await Task.sleep(for: .milliseconds(10)) }
    XCTAssertTrue(controller.log.contains("Chat request failed"))
    XCTAssertFalse(controller.log.contains("private prompt"))
    let throughput = ThroughputSession()
    throughput.logServer = controller
    throughput.reportFailure(URLError(.cannotConnectToHost))
    XCTAssertTrue(controller.log.contains("Throughput failed"))
    let library = ModelLibrary(defaults: defaults)
    library.logServer = controller
    library.startVerification(InstalledModelInfo(url: temporaryURL(), size: 1, quickIssues: [],
      hasMTP: false, hasDSpark: false, modelKind: .deepSeekV4,
      modelID: "deepseek-v4-flash-0731"))
    for _ in 0..<100 where library.isBusy { try await Task.sleep(for: .milliseconds(10)) }
    XCTAssertFalse(library.isBusy)
    XCTAssertTrue(controller.log.contains("Model verification started"))
    XCTAssertTrue(controller.log.contains("Model verification failed"))
  }

  func testSavedPreferenceAndLegacyDefault() throws {
    let suite = "ServerLogFileTests-\(UUID())"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    var configuration = ServerConfiguration.load(defaults: defaults, apiKey: "")
    XCTAssertFalse(configuration.saveLogsToFile)
    configuration.saveLogsToFile = true
    configuration.save(defaults: defaults)
    XCTAssertTrue(ServerConfiguration.load(defaults: defaults, apiKey: "").saveLogsToFile)
    var object = try XCTUnwrap(JSONSerialization.jsonObject(
      with: XCTUnwrap(defaults.data(forKey: ServerConfiguration.preferenceKey))) as? [String: Any])
    object.removeValue(forKey: "saveLogsToFile")
    defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: ServerConfiguration.preferenceKey)
    XCTAssertFalse(ServerConfiguration.load(defaults: defaults, apiKey: "").saveLogsToFile)
  }

  @MainActor
  func testLogsPageRendersInAllLanguages() throws {
    let suite = "ServerLogFileTests-\(UUID())"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let configuration = ServerConfiguration.load(defaults: defaults, apiKey: "")
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      let view = NSHostingView(rootView: LogsView(server: ServerController(logFileURL: temporaryURL()),
        configuration: .constant(configuration), language: language))
      view.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
      view.layoutSubtreeIfNeeded()
      XCTAssertNotNil(view.bitmapImageRepForCachingDisplay(in: view.bounds))
      if language != .english {
        XCTAssertNotEqual(L10n.string("Save logs to file", language: language), "Save logs to file")
      }
    }
  }

  @MainActor
  private func drain(_ controller: ServerController) async {
    await withCheckedContinuation { continuation in
      controller.logFileQueue.async { continuation.resume() }
    }
  }
}
