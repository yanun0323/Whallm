import Foundation
import XCTest

@testable import DeepSeekRepack

final class XetCheckpointSourceTests: XCTestCase {
  func testCancellingXetRangeStopsHelper() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let executable = directory.appendingPathComponent("slow-helper")
    try Data("#!/bin/sh\nexec /bin/sleep 5\n".utf8).write(to: executable)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    let runtime = XetRuntime(
      python: executable, helper: directory.appendingPathComponent("unused"),
      runtimeDirectory: directory, pythonHome: nil, sitePackages: nil)
    let started = Date()
    let task = Task {
      try await runtime.download(
        modelID: "fixture", revision: "fixture", path: "shard.safetensors", range: 0..<4)
    }
    try await Task.sleep(for: .milliseconds(200))
    task.cancel()
    do {
      _ = try await task.value
      XCTFail("the cancelled Xet helper must not return data")
    } catch is CancellationError {
      XCTAssertLessThan(Date().timeIntervalSince(started), 4)
    }
  }

  func testLiveXetRangeMatchesCheckpointBytes() async throws {
    guard ProcessInfo.processInfo.environment["WHALLM_TEST_XET"] == "1" else {
      throw XCTSkip("Set WHALLM_TEST_XET=1 for the bounded live Xet range check")
    }
    guard let runtime = XetRuntime.current(), XetRuntime.hasReadToken() else {
      throw XCTSkip("The live Xet check needs Python, hf_xet, and a local HF read token")
    }
    let base = HuggingFaceSource(
      modelID: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
      revision: "1bae0e1088993b9b82de08f59de5be5a34e42170")
    let xet = XetCheckpointSource(base: base, runtime: runtime)
    let comparisonBytes: UInt64 = 64 * 1_024 * 1_024
    let xetBytes = ProcessInfo.processInfo.environment["WHALLM_TEST_XET_LONG"] == "1"
      ? 512 * 1_024 * 1_024 : comparisonBytes
    let start: UInt64 = ProcessInfo.processInfo.environment["WHALLM_TEST_XET_OFFSET"] == "1"
      ? 12_345 : 0
    let started = Date()
    let xetData = try await xet.data(
      path: "model-00001-of-00048.safetensors", range: start..<(start + xetBytes))
    let seconds = Date().timeIntervalSince(started)
    let httpData = try await base.data(
      path: "model-00001-of-00048.safetensors", range: start..<(start + comparisonBytes))
    XCTAssertEqual(xetData.count, Int(xetBytes))
    XCTAssertEqual(sha256(Data(xetData.prefix(Int(comparisonBytes)))), sha256(httpData))
    print("Xet range: \(xetBytes) bytes in \(seconds) seconds")
  }
}
