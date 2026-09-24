import Foundation
import AppKit
import SwiftUI
import Combine
import XCTest
@testable import DeepSeekV4SSDApp

final class ChatAttachmentTests: XCTestCase {
  func testUnuploadedImagesCannotSilentlyBecomeTextOnlyMessages() throws {
    let attachment = ChatAttachment(id: UUID(), name: "fixture.png", mime: "image/png", sha256: "fixture")
    var message = ChatMessage(role: "user", content: "Describe", attachments: [attachment])
    XCTAssertThrowsError(try JSONEncoder().encode(message))
    message.uploadedFileIDs = ["file-fixture"]
    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(message)) as? [String: Any])
    let parts = try XCTUnwrap(object["content"] as? [[String: String]])
    XCTAssertEqual(parts, [["type": "text", "text": "Describe"], ["type": "image", "file_id": "file-fixture"]])
    XCTAssertNil(object["attachments"])
    XCTAssertNil(object["sha256"])
  }

  func testImageHistoryPersistsLocalReferencesButNotExpiringServerIDs() throws {
    let suite = "ChatAttachmentTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let attachment = ChatAttachment(id: UUID(), name: "photo.png", mime: "image/png", sha256: "fixture")
    var message = ChatMessage(role: "user", content: "photo", attachments: [attachment])
    message.uploadedFileIDs = ["file-expiring"]
    ChatHistory.save([message], defaults: defaults)
    let restored = try XCTUnwrap(ChatHistory.load(defaults: defaults).first)
    XCTAssertEqual(restored.attachments, [attachment])
    XCTAssertTrue(restored.uploadedFileIDs.isEmpty)
    XCTAssertEqual(restored.attachments[0].fileURL.lastPathComponent, "\(attachment.id.uuidString).image")
  }

  func testDocumentAndImagePartsKeepTheirTypesAndOrder() throws {
    let document = ChatAttachment(id: UUID(), name: "invoice.pdf", mime: "application/pdf", sha256: "fixture")
    let image = ChatAttachment(id: UUID(), name: "photo.png", mime: "image/png", sha256: "fixture")
    var message = ChatMessage(role: "user", content: "Compare", attachments: [document, image])
    message.uploadedFileIDs = ["file-document", "file-image"]
    let body = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(message)) as? [String: Any])
    let parts = try XCTUnwrap(body["content"] as? [[String: String]])
    XCTAssertEqual(parts.map { $0["type"] }, ["text", "file", "image"])
    XCTAssertEqual(parts[1]["file_id"], "file-document")
    XCTAssertFalse(document.isImage)
  }

  func testDocumentEnvelopeChecksDoNotRenderOrExecuteContents() throws {
    XCTAssertEqual(try ChatAttachment.validate(Data("%PDF-fixture".utf8), declaredMIME: "application/pdf"), "application/pdf")
    for ext in ["docx", "pptx", "xlsx"] {
      let mime = try XCTUnwrap(ChatAttachment.documentMIMEs[ext])
      XCTAssertEqual(try ChatAttachment.validate(Data([0x50, 0x4b, 0x03, 0x04]), declaredMIME: mime), mime)
      XCTAssertThrowsError(try ChatAttachment.validate(Data("not a zip".utf8), declaredMIME: mime))
    }
    XCTAssertThrowsError(try ChatAttachment.validate(Data("not a pdf".utf8), declaredMIME: "application/pdf"))
  }

  func testAudioPartsRemainDistinctFromDocumentsAndImages() throws {
    let audio = ChatAttachment(id: UUID(), name: "speech.wav", mime: "audio/wav", sha256: "fixture")
    let doc = ChatAttachment(id: UUID(), name: "text.pdf", mime: "application/pdf", sha256: "fixture")
    var message = ChatMessage(role: "user", content: "Compare", attachments: [audio, doc])
    message.uploadedFileIDs = ["file-audio", "file-document"]
    let body = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(message)) as? [String: Any])
    let parts = try XCTUnwrap(body["content"] as? [[String: String]])
    XCTAssertEqual(parts.map { $0["type"] }, ["text", "input_audio", "file"])
    XCTAssertTrue(audio.isAudio)
    XCTAssertFalse(audio.isDocument)
    XCTAssertFalse(audio.isImage)
    XCTAssertTrue(doc.isDocument)
  }

  func testAudioImportChecksEnvelopeWithoutInvokingPlayback() throws {
    var envelope = Data("RIFF".utf8) + Data(repeating: 0, count: 4) + Data("WAVE".utf8)
    envelope += Data(repeating: 0, count: 32)
    XCTAssertEqual(try ChatAttachment.validate(envelope, declaredMIME: "audio/wav"), "audio/wav")
    XCTAssertThrowsError(try ChatAttachment.validate(Data("not audio".utf8), declaredMIME: "audio/wav"))
    XCTAssertThrowsError(try ChatAttachment.validate(Data(envelope.prefix(12)), declaredMIME: "audio/wav"))
  }

  func testAttachmentKindsFollowImplementedModelCapabilities() {
    XCTAssertEqual(ChatAttachmentKind.supported(modelID: "mimo-v2.6-flash-rl"), [.image, .document, .audio])
    for id in [nil, "unknown", "qwen3.8-flash-next-fp8", "deepseek-v4-flash-0731"] {
      XCTAssertTrue(ChatAttachmentKind.supported(modelID: id).isEmpty)
    }
    XCTAssertEqual(ChatAttachmentKind.audio.contentTypes.compactMap(\.preferredFilenameExtension), ["wav"])
  }

  func testPickerAndDropImportAreAtomicAndRejectWebLinks() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let pdf = directory.appendingPathComponent("invoice.pdf")
    let invalid = directory.appendingPathComponent("not-supported.mp4")
    try Data("%PDF-fixture".utf8).write(to: pdf)
    try Data("not an image".utf8).write(to: invalid)
    let initial = Set((try? FileManager.default.contentsOfDirectory(atPath: ChatAttachment.directory.path)) ?? [])
    for urls in [[pdf, invalid], [pdf, URL(string: "https://example.com/audio.wav")!]] {
      XCTAssertThrowsError(try ChatAttachment.importFiles(urls, existing: [], allowedKinds: [.document], language: .english))
      XCTAssertEqual(Set((try? FileManager.default.contentsOfDirectory(atPath: ChatAttachment.directory.path)) ?? []), initial)
    }
    XCTAssertThrowsError(try ChatAttachment.importFiles([pdf], existing: [], allowedKinds: [], language: .english))
    XCTAssertEqual(Set((try? FileManager.default.contentsOfDirectory(atPath: ChatAttachment.directory.path)) ?? []), initial)
    let imported = try ChatAttachment.importFiles([pdf], existing: [], allowedKinds: [.document], language: .english)
    defer { imported.forEach { $0.remove() } }
    XCTAssertEqual(imported.first?.formatName, "PDF")
    XCTAssertEqual(imported.first?.storedByteCount, 12)
    XCTAssertNoThrow(try ChatAttachment.validateSelection(imported, allowedKinds: [.document], language: .english))
    try FileManager.default.removeItem(at: imported[0].fileURL)
    XCTAssertNil(imported[0].storedByteCount)
    XCTAssertThrowsError(try ChatAttachment.validateSelection(imported, allowedKinds: [.document], language: .english))
  }

  func testSelectionBudgetsIncludeHistoryAndRecheckActualBytes() throws {
    let audio = ChatAttachment(id: UUID(), name: "sound.wav", mime: "audio/wav", sha256: "fixture")
    XCTAssertThrowsError(try ChatAttachment.validateSelection([audio, audio, audio], allowedKinds: [.audio], language: .english))
    let document = ChatAttachment(id: UUID(), name: "text.pdf", mime: "application/pdf", sha256: "fixture")
    XCTAssertThrowsError(try ChatAttachment.validateSelection(Array(repeating: document, count: 5), allowedKinds: [.document], language: .english))
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let pdf = directory.appendingPathComponent("large.pdf")
    let pcm = directory.appendingPathComponent("sound.wav")
    try (Data("%PDF-".utf8) + Data(repeating: 0, count: 8 * 1_024 * 1_024 - 5)).write(to: pdf)
    try (Data("RIFF".utf8) + Data(repeating: 0, count: 4) + Data("WAVE".utf8) + Data(repeating: 0, count: 32)).write(to: pcm)
    let large = try ChatAttachment.importFiles([pdf], existing: [], allowedKinds: [.document], language: .english)[0]
    defer { large.remove() }
    XCTAssertNoThrow(try ChatAttachment.validateSelection(Array(repeating: large, count: 4), allowedKinds: [.document], language: .english))
    XCTAssertThrowsError(try ChatAttachment.importFiles([pcm], existing: Array(repeating: large, count: 4),
      allowedKinds: [.audio, .document], language: .english))
  }

  @MainActor
  func testAttachmentOnlyRequestReportsStagesAndClearsStateOnCompletion() async throws {
    let suite = "ChatAttachmentStages.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let attachment = ChatAttachment(id: UUID(), name: "voice.wav", mime: "audio/wav", sha256: "fixture")
    let session = ChatSession(defaults: defaults, stream: { messages, _, _, _, _, _, progress, receive in
      XCTAssertEqual(messages.last?.attachments, [attachment])
      progress(.uploading(completed: 1, total: 1))
      progress(.preparing)
      progress(.generating)
      receive(ChatDelta(content: "42", reasoningContent: ""))
    })
    var stages: [ChatRequestStage] = []
    let observation = session.$requestStage.compactMap { $0 }.sink { stages.append($0) }
    defer { observation.cancel() }
    XCTAssertTrue(session.send(text: "", configuration: .localDefault, model: "mimo-v2.6-flash-rl",
      thinkingMode: "chat", language: .english, attachments: [attachment]))
    for _ in 0..<100 where session.isSending { try await Task.sleep(for: .milliseconds(10)) }
    XCTAssertFalse(session.isSending)
    XCTAssertNil(session.requestStage)
    XCTAssertEqual(stages, [.uploading(completed: 0, total: 1), .uploading(completed: 1, total: 1), .preparing, .generating])
    XCTAssertEqual(session.messages.last?.content, "42")
  }

  @MainActor
  func testCancellingAnUploadRetainsHistoryButClearsBusyState() async throws {
    let suite = "ChatAttachmentCancel.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let attachment = ChatAttachment(id: UUID(), name: "voice.wav", mime: "audio/wav", sha256: "fixture")
    let started = expectation(description: "upload started")
    let session = ChatSession(defaults: defaults, stream: { _, _, _, _, _, _, progress, _ in
      progress(.uploading(completed: 0, total: 1))
      started.fulfill()
      try await Task.sleep(for: .seconds(30))
    })
    XCTAssertTrue(session.send(text: "", configuration: .localDefault, model: "mimo-v2.6-flash-rl",
      thinkingMode: "chat", language: .english, attachments: [attachment]))
    await fulfillment(of: [started], timeout: 5)
    session.stopGenerating()
    for _ in 0..<100 where session.isSending { try await Task.sleep(for: .milliseconds(10)) }
    XCTAssertFalse(session.isSending)
    XCTAssertNil(session.requestStage)
    XCTAssertNil(session.errorMessage)
    XCTAssertEqual(session.messages.count, 1)
    XCTAssertEqual(session.messages[0].attachments, [attachment])
  }

  @MainActor
  func testAttachmentComposerNativePreviews() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let names = ["invoice-with-a-long-filename-2026.pdf", "spoken-number.wav", "photo.png"]
    let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jMZkAAAAASUVORK5CYII=")!
    let contents = [Data("%PDF-fixture".utf8), Data("RIFF".utf8) + Data(repeating: 0, count: 4) + Data("WAVE".utf8) + Data(repeating: 0, count: 32), png]
    let urls = try zip(names, contents).map { name, data in
      let url = directory.appendingPathComponent(name); try data.write(to: url); return url
    }
    let files = try ChatAttachment.importFiles(urls, existing: [], allowedKinds: [.image, .document, .audio], language: .english)
    defer { files.forEach { $0.remove() } }
    let full = try ChatAttachment.importFiles([urls[0], urls[1], urls[2], urls[0], urls[1], urls[2], urls[0], urls[2]],
      existing: [], allowedKinds: [.image, .document, .audio], language: .english)
    defer { full.forEach { $0.remove() } }
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      for (name, width, dark, pending, kinds) in [
        ("empty", 820.0, true, [ChatAttachment](), Set(ChatAttachmentKind.allCases)),
        ("files", 820.0, true, files, Set(ChatAttachmentKind.allCases)),
        ("narrow", 420.0, true, files, Set(ChatAttachmentKind.allCases)),
        ("text-only", 420.0, true, files, Set<ChatAttachmentKind>()),
        ("full", 820.0, true, full, Set(ChatAttachmentKind.allCases)),
        ("missing", 420.0, true, [ChatAttachment(id: UUID(), name: "removed.wav", mime: "audio/wav", sha256: "fixture")], Set(ChatAttachmentKind.allCases)),
      ] {
        // Match ChatView's viewport ownership rather than treating a
        // ViewThatFits child's unconstrained ideal width as a window minimum.
        let host = NSHostingView(rootView: GeometryReader { geometry in
          ChatAttachmentsView(attachments: pending, history: [], allowedKinds: kinds,
            language: language, disabled: false, choose: { _ in }, addFiles: { _ in true }, remove: { _ in })
            .padding(16)
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .topLeading)
            .background(AppTheme.pageBackground).preferredColorScheme(dark ? .dark : .light)
        })
        host.frame = NSRect(x: 0, y: 0, width: width, height: 310)
        let window = NSWindow(contentRect: NSRect(x: -10_000, y: -10_000, width: width, height: 310),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderBack(nil)
        defer { window.close() }
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        host.layoutSubtreeIfNeeded()
        XCTAssertLessThanOrEqual(host.fittingSize.height, 310)
        XCTAssertLessThanOrEqual(host.fittingSize.width, width)
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        if let path = ProcessInfo.processInfo.environment["WHALLM_ATTACHMENT_PREVIEWS"] {
          let output = URL(fileURLWithPath: path)
          try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
          try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            .write(to: output.appendingPathComponent("attachments-\(language.rawValue)-\(name).png"))
        }
      }
    }
  }

  func testImageValidationRejectsBadAndOversizedBytes() throws {
    XCTAssertThrowsError(try ChatAttachment.validate(Data("not an image".utf8)))
    XCTAssertThrowsError(try ChatAttachment.validate(Data(repeating: 0, count: 8 * 1_024 * 1_024 + 1)))
    let png = try XCTUnwrap(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jMZkAAAAASUVORK5CYII="))
    XCTAssertEqual(try ChatAttachment.validate(png), "image/png")
  }
}
