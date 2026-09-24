import CryptoKit
import Foundation
import ImageIO
import AppKit
import SwiftUI
import DeepSeekRepack
import UniformTypeIdentifiers

// Only implemented input types belong here; this is not the checkpoint's full
// modality inventory. Picker and drag/drop use the same selection validation.
enum ChatAttachmentKind: String, CaseIterable, Identifiable {
  case image, document, audio
  var id: Self { self }
  var feature: String {
    switch self { case .image: "imageInput"; case .document: "documentInput"; case .audio: "audioInput" }
  }
  var titleKey: String {
    switch self { case .image: "Images"; case .document: "Documents"; case .audio: "WAV audio" }
  }
  var actionKey: String {
    switch self { case .image: "Attach images…"; case .document: "Attach documents…"; case .audio: "Attach WAV audio…" }
  }
  var maximumCount: Int {
    switch self { case .image: 8; case .document: 4; case .audio: 2 }
  }
  var symbol: String {
    switch self { case .image: "photo"; case .document: "doc.text"; case .audio: "waveform" }
  }
  var limitKey: String {
    switch self {
    case .image: "PNG, JPEG or WebP. Up to 1,048,576 pixels per image."
    case .document: "PDF: 4 pages. PPTX/XLSX: 4 slides/sheets. Office layout is not rendered."
    case .audio: "Audio: 24 kHz, 16-bit PCM WAV, mono or stereo, up to 30 seconds."
    }
  }
  var contentTypes: [UTType] {
    let extensions: [String]
    switch self {
    case .image: extensions = ["png", "jpg", "webp"]
    case .document: extensions = ChatAttachment.documentMIMEs.keys.sorted()
    case .audio: extensions = ["wav"]
    }
    return extensions.compactMap { UTType(filenameExtension: $0) }
  }
  static func supported(modelID: String?) -> Set<Self> {
    guard let descriptor = ModelPackages.descriptors.first(where: { $0.apiModelID == modelID }) else { return [] }
    return Set(allCases.filter { descriptor.supports($0.feature) })
  }
}

struct ChatAttachment: Codable, Identifiable, Sendable, Equatable {
  let id: UUID
  let name: String
  let mime: String
  let sha256: String

  static let documentMIMEs = [
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ]
  var isImage: Bool { mime.hasPrefix("image/") }
  var isAudio: Bool { mime == "audio/wav" }
  var isDocument: Bool { Self.documentMIMEs.values.contains(mime) }
  var kind: ChatAttachmentKind? { isImage ? .image : isDocument ? .document : isAudio ? .audio : nil }
  var formatName: String {
    if isAudio { return "PCM WAV" }
    if let ext = Self.documentMIMEs.first(where: { $0.value == mime })?.key { return ext.uppercased() }
    return mime.split(separator: "/").last.map { $0.uppercased() } ?? mime
  }
  // Display only: limits and integrity are rechecked against bytes on send.
  var storedByteCount: Int? { (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize }

  static var directory: URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appending(path: "Whallm/ChatAttachments", directoryHint: .isDirectory)
  }
  var fileURL: URL { Self.directory.appending(path: "\(id.uuidString).image") }

  static func validate(_ data: Data, declaredMIME: String? = nil) throws -> String {
    if declaredMIME == "audio/wav" {
      // Envelope check only; bounded PCM parsing belongs to the server. No
      // audio decoder or autoplay is invoked by importing or previewing a file.
      guard data.count >= 44, data.count <= 8 * 1_024 * 1_024,
        data.starts(with: Data("RIFF".utf8)), data.subdata(in: 8..<12) == Data("WAVE".utf8) else {
        throw AttachmentError(L10n.string("Choose a PCM WAV file up to 8 MiB."))
      }
      return "audio/wav"
    }
    if let declaredMIME, documentMIMEs.values.contains(declaredMIME) {
      // Envelope check only. Native document parsing stays in the bounded server
      // worker; neither import nor preview executes/renders Office or PDF data.
      let magic = declaredMIME == "application/pdf" ? Data("%PDF-".utf8) : Data([0x50, 0x4b, 0x03, 0x04])
      guard data.count <= 8 * 1_024 * 1_024, data.starts(with: magic) else {
        throw AttachmentError(L10n.string("Choose a PDF, DOCX, PPTX or XLSX file up to 8 MiB."))
      }
      return declaredMIME
    }
    guard !data.isEmpty, data.count <= 8 * 1_024 * 1_024,
      let source = CGImageSourceCreateWithData(data as CFData, nil),
      CGImageSourceGetCount(source) == 1,
      let type = CGImageSourceGetType(source) as String?,
      let mime = ["public.png": "image/png", "public.jpeg": "image/jpeg", "org.webmproject.webp": "image/webp"][type],
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? Int,
      let height = properties[kCGImagePropertyPixelHeight] as? Int,
      width > 0, height > 0, width <= 1_048_576 / height
    else { throw AttachmentError(L10n.string("Choose a still PNG, JPEG or WebP image up to 8 MiB and 1,048,576 pixels.")) }
    return mime
  }

  static func validateSelection(_ attachments: [Self], allowedKinds: Set<ChatAttachmentKind>, language: AppLanguage) throws {
    guard attachments.allSatisfy({ $0.kind.map(allowedKinds.contains) == true }) else {
      throw AttachmentError(L10n.string("Choose a model that supports these attachments, or clear chat history.", language: language))
    }
    guard attachments.count <= 8, ChatAttachmentKind.allCases.allSatisfy({ kind in
      attachments.filter { $0.kind == kind }.count <= kind.maximumCount
    }) else {
      throw AttachmentError(L10n.string("Attachments exceed the server limits. Remove pending files or start a new chat.", language: language))
    }
    var bytes = 0
    for attachment in attachments {
      bytes += try attachment.data().count
      guard bytes <= 32 * 1_024 * 1_024 else {
        throw AttachmentError(L10n.string("Attachments exceed the server limits. Remove pending files or start a new chat.", language: language))
      }
    }
  }

  static func importFiles(_ urls: [URL], existing: [Self], allowedKinds: Set<ChatAttachmentKind>, language: AppLanguage) throws -> [Self] {
    guard !urls.isEmpty, urls.allSatisfy(\.isFileURL) else {
      throw AttachmentError(L10n.string("Choose local files. Web links cannot be attached.", language: language))
    }
    guard existing.count + urls.count <= 8 else {
      throw AttachmentError(L10n.string("Attachments exceed the server limits. Remove pending files or start a new chat.", language: language))
    }
    var imported: [Self] = []
    do {
      for url in urls { imported.append(try importFile(url)) }
      try validateSelection(existing + imported, allowedKinds: allowedKinds, language: language)
      return imported
    } catch {
      for attachment in imported { attachment.remove() }
      throw error
    }
  }

  static func importFile(_ url: URL) throws -> ChatAttachment {
    guard url.isFileURL else { throw AttachmentError(L10n.string("Choose local files. Web links cannot be attached.")) }
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    let size = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard size.isRegularFile == true, let bytes = size.fileSize, bytes <= 8 * 1_024 * 1_024 else {
      throw AttachmentError(L10n.string("Choose a supported attachment up to 8 MiB."))
    }
    let data = try boundedData(url)
    let result = ChatAttachment(id: UUID(), name: url.lastPathComponent,
      mime: try validate(data, declaredMIME: url.pathExtension.lowercased() == "wav" ? "audio/wav" : documentMIMEs[url.pathExtension.lowercased()]),
      sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700])
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
    do {
      try data.write(to: result.fileURL, options: .atomic)
      try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: result.fileURL.path)
    } catch {
      result.remove()
      throw error
    }
    return result
  }

  private static func boundedData(_ url: URL) throws -> Data {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    return try file.read(upToCount: 8 * 1_024 * 1_024 + 1) ?? Data()
  }

  func data() throws -> Data {
    guard let data = try? Self.boundedData(fileURL) else {
      throw AttachmentError(L10n.string("An attachment cannot be read. Remove pending files or clear chat history."))
    }
    guard try Self.validate(data, declaredMIME: mime) == mime,
      SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sha256 else {
      throw AttachmentError(L10n.string("The saved attachment changed. Remove it and attach it again."))
    }
    return data
  }

  func remove() { try? FileManager.default.removeItem(at: fileURL) }
}

struct ChatAttachmentPreview: View {
  let attachment: ChatAttachment
  var language: AppLanguage = .english
  @State private var image: NSImage?

  var body: some View {
    HStack(spacing: 10) {
      Group {
        if let image {
          Image(nsImage: image).resizable().scaledToFit()
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.primary.opacity(0.1)))
        } else {
          Image(systemName: attachment.kind?.symbol ?? "doc")
            .font(.title2).foregroundStyle(.secondary)
        }
      }
      .frame(width: 40, height: 40)
      .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 3) {
        Text(attachment.name).lineLimit(1).truncationMode(.middle).help(attachment.name)
        if let bytes = attachment.storedByteCount {
          Text(L10n.string("%@ · %@", language: language, attachment.formatName,
            Int64(bytes).formatted(.byteCount(style: .memory))))
            .font(.caption).foregroundStyle(.secondary)
        } else {
          Label(L10n.string("File unavailable", language: language), systemImage: "exclamationmark.triangle")
            .font(.caption).foregroundStyle(.secondary)
        }
      }
      .frame(maxWidth: .infinity, alignment: .leading)
    }
    .accessibilityElement(children: .combine)
    .task(id: attachment.id) {
      image = nil
      if attachment.isImage, let data = try? attachment.data(),
        let source = CGImageSourceCreateWithData(data as CFData, nil),
        let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
          kCGImageSourceCreateThumbnailFromImageAlways: true,
          kCGImageSourceCreateThumbnailWithTransform: true,
          kCGImageSourceThumbnailMaxPixelSize: 96,
        ] as CFDictionary) {
        image = NSImage(cgImage: thumbnail, size: .zero)
      }
    }
  }
}

struct AttachmentError: LocalizedError {
  let message: String
  init(_ message: String) { self.message = message }
  var errorDescription: String? { message }
}
