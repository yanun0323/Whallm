import Foundation

/// Called on the controller's serial utility queue, never on the UI thread.
struct ServerLogFile: Sendable {
  static let maximumBytes = 10_000_000
  static var defaultURL: URL {
    FileManager.default.homeDirectoryForCurrentUser
      .appending(path: ".whallm/server.log")
  }

  let url: URL
  var maximumBytes = Self.maximumBytes

  func append(_ text: String) throws {
    let manager = FileManager.default
    try manager.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700])
    if !manager.fileExists(atPath: url.path) {
      guard manager.createFile(atPath: url.path, contents: nil,
                               attributes: [.posixPermissions: 0o600]) else {
        throw CocoaError(.fileWriteUnknown)
      }
    }
    try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    let file = try FileHandle(forUpdating: url)
    defer { try? file.close() }
    let size = try file.seekToEnd()
    let incoming = Data(text.utf8)
    if size + UInt64(incoming.count) <= UInt64(maximumBytes) {
      try file.write(contentsOf: incoming)
      return
    }

    // Read only the tail, even if an existing file is already oversized.
    let retainedBytes = max(0, maximumBytes - incoming.count)
    try file.seek(toOffset: size > UInt64(retainedBytes) ? size - UInt64(retainedBytes) : 0)
    var tail = try file.readToEnd() ?? Data()
    tail.append(contentsOf: incoming.suffix(maximumBytes))
    // Prefer whole lines. A single oversized line keeps its newest UTF-8 suffix.
    if let newline = tail.firstIndex(of: 0x0A), newline < tail.index(before: tail.endIndex) {
      tail.removeSubrange(tail.startIndex...newline)
    } else {
      while let byte = tail.first, byte & 0xC0 == 0x80 { tail.removeFirst() }
    }
    try file.seek(toOffset: 0)
    try file.write(contentsOf: tail)
    try file.truncate(atOffset: UInt64(tail.count))
  }
}
