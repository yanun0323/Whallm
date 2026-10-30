import Foundation
import XCTest

@testable import DeepSeekRepack

final class XetCheckpointSourceTests: XCTestCase {
  func testXetHelperReportsTransferProgress() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let executable = directory.appendingPathComponent("progress-helper")
    try Data("#!/bin/sh\nprintf 'abcdefgh' > \"$7\"\nprintf '4\\n'\nsleep 1\nprintf '8\\n'\n".utf8).write(to: executable)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    let runtime = XetRuntime(
      python: executable, helper: directory.appendingPathComponent("unused"),
      runtimeDirectory: directory, pythonHome: nil, sitePackages: nil)
    let updates = TransferProgressProbe()
    let started = Date()
    let result = try await runtime.download(
      modelID: "fixture", revision: "fixture", path: "shard.safetensors", range: 0..<8,
      onProgress: { updates.add($0) })
    XCTAssertEqual(result, Data("abcdefgh".utf8))
    XCTAssertEqual(updates.values, [4, 8])
    XCTAssertLessThan(try XCTUnwrap(updates.firstTime).timeIntervalSince(started), 0.8)
  }

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

  func testStagesAndReadsACompleteXetFile() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let executable = directory.appendingPathComponent("file-helper")
    try Data("#!/bin/sh\nprintf 'abcdefgh' > \"$5\"\nprintf '4\\n8\\n'\n".utf8).write(to: executable)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    let runtime = XetRuntime(
      python: executable, helper: directory.appendingPathComponent("xet_range.py"),
      runtimeDirectory: directory, pythonHome: nil, sitePackages: nil)
    let source = XetCheckpointSource(
      base: HuggingFaceSource(modelID: "fixture", revision: "fixture"), runtime: runtime)
    let updates = TransferProgressProbe()
    let stagedBytes = try await source.stage(path: "shard.safetensors") { updates.add($0) }
    XCTAssertEqual(stagedBytes, 8)
    XCTAssertEqual(updates.values, [4, 8])
    let range = try await source.data(path: "shard.safetensors", range: 2..<6)
    XCTAssertEqual(range, Data("cdef".utf8))
    await source.releaseAll()
  }

  func testLiveXetRangeMatchesCheckpointBytes() async throws {
    guard ProcessInfo.processInfo.environment["WHALLM_TEST_XET"] == "1" else {
      throw XCTSkip("Set WHALLM_TEST_XET=1 for the bounded live Xet range check")
    }
    guard let runtime = XetRuntime.current(), XetRuntime.hasReadToken() else {
      throw XCTSkip("The live Xet check needs Python, hf_xet, and a local HF read token")
    }
    let checkpoint = liveCheckpoint()
    let base = HuggingFaceSource(
      modelID: checkpoint.modelID, revision: checkpoint.revision)
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

  func testLiveParallelXetRanges() async throws {
    guard ProcessInfo.processInfo.environment["WHALLM_TEST_XET_PARALLEL"] == "1" else {
      throw XCTSkip("Set WHALLM_TEST_XET_PARALLEL=1 for the bounded parallel Xet check")
    }
    guard let runtime = XetRuntime.current(), XetRuntime.hasReadToken() else {
      throw XCTSkip("The parallel Xet check needs Python, hf_xet, and a local HF read token")
    }
    let bytes: UInt64 = 512 * 1_024 * 1_024
    let checkpoint = liveCheckpoint()
    let modelID = checkpoint.modelID
    let revision = checkpoint.revision
    let probes = (0..<3).map { _ in TransferProgressProbe() }
    let started = Date()
    let counts = try await withThrowingTaskGroup(of: Int.self) { group in
      for index in 0..<3 {
        let path = String(format: "model-%05d-of-00048.safetensors", index + 1)
        let probe = probes[index]
        group.addTask {
          try await runtime.download(
            modelID: modelID, revision: revision, path: path, range: 0..<bytes,
            onProgress: { probe.add($0) }).count
        }
      }
      var result: [Int] = []
      for try await count in group { result.append(count) }
      return result
    }
    XCTAssertEqual(counts, [Int(bytes), Int(bytes), Int(bytes)])
    print("Parallel Xet: \(3 * bytes) bytes in \(Date().timeIntervalSince(started)) seconds; "
      + "progress updates \(probes.map { $0.values.count })")
  }

  func testLiveNativeXetFileMatchesHFCopy() async throws {
    guard ProcessInfo.processInfo.environment["WHALLM_TEST_XET_FILE"] == "1" else {
      throw XCTSkip("Set WHALLM_TEST_XET_FILE=1 for the bounded native Xet file check")
    }
    guard let runtime = XetRuntime.current(), XetRuntime.hasReadToken() else {
      throw XCTSkip("The native Xet file check needs Python, hf_xet, and a local HF read token")
    }
    let updates = TransferProgressProbe()
    let started = Date()
    let checkpoint = liveCheckpoint()
    let file = try await runtime.stage(
      modelID: checkpoint.modelID,
      revision: checkpoint.revision,
      path: "model-00001-of-00048.safetensors", onProgress: { updates.add($0) })
    defer { try? FileManager.default.removeItem(at: file.url) }
    XCTAssertEqual(file.size, try fileSize(file.url))
    XCTAssertGreaterThan(updates.values.count, 10)
    if let reference = ProcessInfo.processInfo.environment["WHALLM_TEST_XET_REFERENCE"] {
      XCTAssertEqual(try sha256(file.url), try sha256(URL(fileURLWithPath: reference)))
    }
    print("Native Xet file: \(file.size) bytes in \(Date().timeIntervalSince(started)) "
      + "seconds; \(updates.values.count) progress updates")
  }
}

private func liveCheckpoint() -> (modelID: String, revision: String) {
  let environment = ProcessInfo.processInfo.environment
  return (
    environment["WHALLM_TEST_XET_MODEL_ID"] ?? DeepSeekV41Contract.modelID,
    environment["WHALLM_TEST_XET_REVISION"] ?? DeepSeekV41Contract.revision)
}

private final class TransferProgressProbe: @unchecked Sendable {
  private let lock = NSLock()
  private var recorded: [(UInt64, Date)] = []

  func add(_ value: UInt64) { lock.withLock { recorded.append((value, Date())) } }
  var values: [UInt64] { lock.withLock { recorded.map(\.0) } }
  var firstTime: Date? { lock.withLock { recorded.first?.1 } }
}
