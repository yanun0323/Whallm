import AppKit
import Combine
import SwiftUI
import XCTest

@testable import DeepSeekV4SSDApp

final class ChatStreamDecoderTests: XCTestCase {
  func testChatSeedValidationAndRequestEncoding() throws {
    for (input, expected) in [("", nil), ("  ", nil), ("0", UInt32(0)),
                              (" 42 ", UInt32(42)), ("4294967295", UInt32.max)] {
      let seed = try ChatClient.parseSeed(input, language: .english)
      XCTAssertEqual(seed, expected)
      let request = try ChatClient.makeRequest(
        messages: [ChatMessage(role: "user", content: "Hello")],
        baseURL: URL(string: "http://127.0.0.1:11434")!, apiKey: "",
        model: "test-model", thinkingMode: "chat", seed: seed, enableTestTool: false)
      let body = try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody))
        as? [String: Any])
      XCTAssertEqual((body["seed"] as? NSNumber)?.uint32Value, expected)
      if expected == nil { XCTAssertNil(body["seed"]) }
    }
    for input in ["-1", "+1", "1.5", "1e3", "true", "4294967296", "１２", "1 2"] {
      XCTAssertThrowsError(try ChatClient.parseSeed(input, language: .english))
    }
  }

  @MainActor
  func testChatSeedIsPassedForEachRequestWithoutBecomingADefault() async throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    var seeds: [UInt32?] = []
    let session = ChatSession(defaults: defaults, stream: { _, _, _, _, _, seed, _, receive in
      seeds.append(seed)
      receive(ChatDelta(content: "answer", reasoningContent: ""))
    })
    for seed: UInt32? in [42, nil, 0] {
      XCTAssertTrue(session.send(
        text: "Hello", configuration: .localDefault, model: "test-model",
        thinkingMode: "chat", language: .english, seed: seed))
      for _ in 0..<100 where session.isSending {
        try await Task.sleep(for: .milliseconds(10))
      }
      XCTAssertFalse(session.isSending)
    }
    XCTAssertEqual(seeds, [42, nil, 0])
  }

  @MainActor
  func testLongChatStaysWithinViewport() throws {
    let suite = "ChatViewportTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    ChatHistory.save([
      ChatMessage(role: "user", content: "Explain in detail"),
      ChatMessage(role: "assistant", content: String(repeating: "Long response line\n", count: 500))
    ], defaults: defaults)
    let session = ChatSession(defaults: defaults)
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      let host = NSHostingView(rootView: ChatView(
        configuration: .localDefault, server: ServerController(), session: session,
        language: language))
      for height in [600.0, 900.0] {
        host.frame = NSRect(x: 0, y: 0, width: 1_000, height: height)
        host.layoutSubtreeIfNeeded()
        XCTAssertLessThanOrEqual(host.fittingSize.height, height,
          "The transcript must not increase the window's minimum height")
        XCTAssertEqual(host.frame.height, height)
      }
    }
  }

  @MainActor
  func testChatSeedLayoutPreview() throws {
    guard let directory = ProcessInfo.processInfo.environment["WHALLM_CHAT_PREVIEWS"] else { return }
    let url = URL(fileURLWithPath: directory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    let suite = "ChatSeedPreview.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let session = ChatSession(defaults: defaults)
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      let host = NSHostingView(rootView: ChatView(
        configuration: .localDefault, server: ServerController(), session: session,
        language: language).preferredColorScheme(.dark))
      host.frame = NSRect(x: 0, y: 0, width: 1_000, height: 700)
      let window = NSWindow(contentRect: NSRect(x: -10_000, y: -10_000, width: 1_000, height: 700),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentView = host
      window.orderBack(nil)
      defer { window.close() }
      RunLoop.main.run(until: Date().addingTimeInterval(0.1))
      host.layoutSubtreeIfNeeded()
      let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
      host.cacheDisplay(in: host.bounds, to: bitmap)
      try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        .write(to: url.appendingPathComponent("chat-\(language.rawValue).png"))
    }
  }

  func testChatModelSelectionUsesAliasAndRestoresSavedSelection() {
    let models = [
      CatalogModel(id: "deepseek-v4-flash-0731", alias: "work-model"),
      CatalogModel(id: "qwen3.8-flash-next-fp8", alias: nil),
    ]

    XCTAssertEqual(
      resolvedChatModelName(savedName: "deepseek-v4-flash-0731", models: models),
      "work-model"
    )
    XCTAssertEqual(
      resolvedChatModelName(savedName: "qwen3.8-flash-next-fp8", models: models),
      "qwen3.8-flash-next-fp8"
    )
    XCTAssertEqual(
      resolvedChatModelName(savedName: "missing", models: models),
      "work-model"
    )
    XCTAssertNil(resolvedChatModelName(savedName: "work-model", models: []))
  }

  func testDecoderPreservesThinkingAndAnswerDeltas() throws {
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line: #"data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}"#),
      .delta(ChatDelta(content: "", reasoningContent: "plan"))
    )
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line: #"data: {"choices":[{"delta":{"content":"Hello"}}]}"#),
      .delta(ChatDelta(content: "Hello", reasoningContent: ""))
    )
    let toolCall = ChatToolCall(
      id: "call_1",
      type: "function",
      function: .init(
        name: "get_current_time",
        arguments: #"{"time_zone":"Asia/Taipei"}"#
      )
    )
    let toolDelta = ChatToolCallDelta(
      index: 0,
      id: "call_1",
      type: "function",
      function: .init(
        name: "get_current_time",
        arguments: #"{"time_zone":"Asia/Taipei"}"#
      )
    )
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line:
          #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_current_time","arguments":"{\"time_zone\":\"Asia/Taipei\"}"}}]}}]}"#
      ),
      .delta(ChatDelta(content: "", reasoningContent: "", toolCalls: [toolDelta]))
    )
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line: #"data: {"choices":[],"usage":{"prompt_tokens":34,"completion_tokens":12}}"#),
      .usage(promptTokens: 34, completionTokens: 12)
    )
    XCTAssertEqual(try ChatStreamDecoder.decode(line: "data: [DONE]"), .done)
    XCTAssertNil(try ChatStreamDecoder.decode(line: ""))
    XCTAssertEqual(
      ChatMetrics(
        promptTokens: 34,
        completionTokens: 12,
        elapsedSeconds: 3,
        firstTokenSeconds: 1
      ).tokensPerSecond,
      6
    )

    var message = ChatMessage(role: "assistant", content: "")
    for line in [
      #"data: {"choices":[{"delta":{"reasoning_content":"p"}}]}"#,
      #"data: {"choices":[{"delta":{"reasoning_content":"lan"}}]}"#,
      #"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
      #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_current_time","arguments":"{\"time_zone\":\"Asia/Taipei\"}"}}]}}]}"#,
    ] {
      if case .delta(let delta) = try ChatStreamDecoder.decode(line: line) {
        message.append(delta)
      }
    }
    XCTAssertEqual(message.reasoningContent, "plan")
    XCTAssertEqual(message.content, "Hello")
    XCTAssertEqual(message.toolCalls, [toolCall])

    XCTAssertThrowsError(
      try ChatStreamDecoder.decode(
        line: #"data: {"error":{"message":"Tool call 格式無效。"}}"#
      )
    ) { error in
      XCTAssertEqual(error.localizedDescription, "Tool call 格式無效。")
    }

    if case .delta(let delta) = try ChatStreamDecoder.decode(
      line:
        #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tai"}}]}}]}"#
    ) {
      message.append(delta)
    }
    XCTAssertEqual(
      message.toolCalls[0].function.arguments,
      #"{"time_zone":"Asia/Taipei"}Tai"#
    )
  }

  func testChatHistoryPersistenceRestoresLocalFields() throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let message = ChatMessage(
      role: "assistant",
      content: "answer",
      reasoningContent: "reason",
      toolCalls: [
        ChatToolCall(
          id: "call_1",
          type: "function",
          function: .init(name: "get_current_time", arguments: "{}")
        )
      ],
      modelName: "work-model"
    )

    ChatHistory.save([message], defaults: defaults)
    let restored = try XCTUnwrap(ChatHistory.load(defaults: defaults).first)

    XCTAssertEqual(restored.id, message.id)
    XCTAssertEqual(restored.role, message.role)
    XCTAssertEqual(restored.content, message.content)
    XCTAssertEqual(restored.reasoningContent, message.reasoningContent)
    XCTAssertEqual(restored.toolCalls, message.toolCalls)
    XCTAssertEqual(restored.modelName, "work-model")
  }

  func testOldChatHistoryLoadsWithoutAModelName() throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let id = UUID()
    defaults.set(
      try JSONSerialization.data(
        withJSONObject: [
          [
            "id": id.uuidString,
            "role": "assistant",
            "content": "old answer",
            "reasoningContent": "",
            "toolCalls": [],
          ]
        ]),
      forKey: "chatMessages"
    )

    let message = try XCTUnwrap(ChatHistory.load(defaults: defaults).first)

    XCTAssertEqual(message.id, id)
    XCTAssertNil(message.modelName)
  }

  @MainActor
  func testGenerationContinuesAfterLeavingAndReturningToChat() async throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    // Hold the second delta until navigation finishes; runner load must not
    // decide whether generation is still in progress when the view returns.
    let gate = AsyncStream<Void>.makeStream()
    defer { gate.continuation.finish() }
    let session = ChatSession(
      defaults: defaults,
      stream: { _, _, _, _, _, _, _, receive in
        receive(ChatDelta(content: "first", reasoningContent: ""))
        var iterator = gate.stream.makeAsyncIterator()
        _ = await iterator.next()
        try Task.checkCancellation()
        receive(ChatDelta(content: " second", reasoningContent: ""))
      }
    )
    let firstPublished = expectation(description: "first delta published")
    let generationFinished = expectation(description: "generation finished")
    let firstSubscription = session.$messages
      .filter { $0.last?.content == "first" }.prefix(1)
      .sink { _ in firstPublished.fulfill() }
    let finishSubscription = session.$isSending.dropFirst()
      .filter { !$0 }.prefix(1)
      .sink { _ in generationFinished.fulfill() }
    defer {
      firstSubscription.cancel()
      finishSubscription.cancel()
      session.stopGenerating()
    }
    let visibility = ChatVisibility()
    let hostingView = NSHostingView(
      rootView: ChatNavigationHarness(
        visibility: visibility,
        server: ServerController(),
        session: session
      ))
    hostingView.frame = NSRect(x: 0, y: 0, width: 1_000, height: 700)
    hostingView.layoutSubtreeIfNeeded()

    XCTAssertTrue(
      session.send(
        text: "Hello",
        configuration: .localDefault,
        model: "test-model",
        thinkingMode: "chat",
        language: .english
      ))
    await fulfillment(of: [firstPublished], timeout: 5)
    XCTAssertEqual(session.messages.last?.content, "first")

    visibility.showsChat = false
    await Task.yield()
    hostingView.layoutSubtreeIfNeeded()
    visibility.showsChat = true
    await Task.yield()
    hostingView.layoutSubtreeIfNeeded()
    XCTAssertTrue(session.isSending)
    gate.continuation.yield(())
    gate.continuation.finish()
    await fulfillment(of: [generationFinished], timeout: 5)

    XCTAssertEqual(session.messages.last?.content, "first second")
    XCTAssertFalse(session.isSending)
  }

  @MainActor
  func testRapidStreamingCoalescesMessagePublications() async throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let session = ChatSession(
      defaults: defaults,
      stream: { _, _, _, _, _, _, _, receive in
        for _ in 0..<80 {
          receive(ChatDelta(content: "x", reasoningContent: ""))
        }
      }
    )
    var messagePublicationCount = 0
    let observation = session.$messages.dropFirst().sink { _ in
      messagePublicationCount += 1
    }
    defer { observation.cancel() }

    XCTAssertTrue(
      session.send(
        text: "Hello",
        configuration: .localDefault,
        model: "test-model",
        thinkingMode: "chat",
        language: .english
      ))
    for _ in 0..<100 where session.isSending {
      try await Task.sleep(for: .milliseconds(10))
    }

    XCTAssertFalse(session.isSending)
    XCTAssertEqual(session.messages.last?.content, String(repeating: "x", count: 80))
    XCTAssertLessThanOrEqual(messagePublicationCount, 10)
  }

  @MainActor
  func testStreamingFlushesPendingTextBeforeReportingAnError() async throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let session = ChatSession(
      defaults: defaults,
      stream: { _, _, _, _, _, _, _, receive in
        receive(ChatDelta(content: "partial", reasoningContent: ""))
        throw NSError(domain: "ChatStreamDecoderTests", code: 1)
      }
    )

    XCTAssertTrue(
      session.send(
        text: "Hello",
        configuration: .localDefault,
        model: "test-model",
        thinkingMode: "chat",
        language: .english
      ))
    for _ in 0..<100 where session.isSending {
      try await Task.sleep(for: .milliseconds(10))
    }

    XCTAssertFalse(session.isSending)
    XCTAssertEqual(session.messages.last?.content, "partial")
    XCTAssertNotNil(session.errorMessage)
  }
}

@MainActor
private final class ChatVisibility: ObservableObject {
  @Published var showsChat = true
}

private struct ChatNavigationHarness: View {
  @ObservedObject var visibility: ChatVisibility
  @ObservedObject var server: ServerController
  @ObservedObject var session: ChatSession

  var body: some View {
    Group {
      if visibility.showsChat {
        ChatView(
          configuration: .localDefault,
          server: server,
          session: session,
          language: .english
        )
      } else {
        Text("Other")
      }
    }
  }
}
