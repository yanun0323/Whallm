import Foundation

struct ChatToolCall: Codable, Equatable, Identifiable, Sendable {
  struct Function: Codable, Equatable, Sendable {
    var name: String
    var arguments: String
  }

  var id: String
  var type: String
  var function: Function
}

struct ChatToolCallDelta: Decodable, Equatable, Sendable {
  struct Function: Decodable, Equatable, Sendable {
    let name: String?
    let arguments: String?
  }

  let index: Int
  let id: String?
  let type: String?
  let function: Function
}

struct ChatMessage: Encodable, Identifiable, Sendable {
  let id: UUID
  let role: String
  var content: String
  var reasoningContent: String
  var toolCalls: [ChatToolCall]
  var modelName: String?
  var attachments: [ChatAttachment]
  var uploadedFileIDs: [String] = []

  init(
    id: UUID = UUID(),
    role: String,
    content: String,
    reasoningContent: String = "",
    toolCalls: [ChatToolCall] = [],
    modelName: String? = nil,
    attachments: [ChatAttachment] = []
  ) {
    self.id = id
    self.role = role
    self.content = content
    self.reasoningContent = reasoningContent
    self.toolCalls = toolCalls
    self.modelName = modelName
    self.attachments = attachments
  }

  enum CodingKeys: String, CodingKey {
    case role, content
    case toolCalls = "tool_calls"
  }

  func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(role, forKey: .role)
    if attachments.isEmpty {
      try values.encode(content, forKey: .content)
    } else {
      guard uploadedFileIDs.count == attachments.count else {
        throw AttachmentError(L10n.string("Upload every attachment before sending the message."))
      }
      var parts = [["type": "text", "text": content]]
      parts += zip(attachments, uploadedFileIDs).map { ["type": $0.0.isImage ? "image" : $0.0.isAudio ? "input_audio" : "file", "file_id": $0.1] }
      try values.encode(parts, forKey: .content)
    }
    if !toolCalls.isEmpty {
      try values.encode(toolCalls, forKey: .toolCalls)
    }
  }

  mutating func append(_ delta: ChatDelta) {
    content += delta.content
    reasoningContent += delta.reasoningContent
    for update in delta.toolCalls {
      if toolCalls.indices.contains(update.index) {
        if let id = update.id { toolCalls[update.index].id = id }
        if let type = update.type { toolCalls[update.index].type = type }
        if let name = update.function.name {
          toolCalls[update.index].function.name += name
        }
        toolCalls[update.index].function.arguments += update.function.arguments ?? ""
      } else if update.index == toolCalls.count,
        let id = update.id,
        let type = update.type,
        let name = update.function.name
      {
        toolCalls.append(
          ChatToolCall(
            id: id,
            type: type,
            function: .init(name: name, arguments: update.function.arguments ?? "")
          )
        )
      }
    }
  }
}

enum ChatHistory {
  private static let preferenceKey = "chatMessages"

  private struct StoredMessage: Codable {
    let id: UUID
    let role: String
    let content: String
    let reasoningContent: String
    let toolCalls: [ChatToolCall]
    let modelName: String?
    let attachments: [ChatAttachment]?

    init(_ message: ChatMessage) {
      id = message.id
      role = message.role
      content = message.content
      reasoningContent = message.reasoningContent
      toolCalls = message.toolCalls
      modelName = message.modelName
      attachments = message.attachments
    }

    var message: ChatMessage {
      ChatMessage(
        id: id,
        role: role,
        content: content,
        reasoningContent: reasoningContent,
        toolCalls: toolCalls,
        modelName: modelName,
        attachments: attachments ?? []
      )
    }
  }

  static func load(defaults: UserDefaults = .standard) -> [ChatMessage] {
    guard let data = defaults.data(forKey: preferenceKey),
      let stored = try? JSONDecoder().decode([StoredMessage].self, from: data)
    else { return [] }
    return stored.map(\.message)
  }

  static func save(_ messages: [ChatMessage], defaults: UserDefaults = .standard) {
    guard let data = try? JSONEncoder().encode(messages.map(StoredMessage.init)) else { return }
    defaults.set(data, forKey: preferenceKey)
  }
}

struct ChatDelta: Decodable, Equatable, Sendable {
  let content: String
  let reasoningContent: String
  let toolCalls: [ChatToolCallDelta]

  init(content: String, reasoningContent: String, toolCalls: [ChatToolCallDelta] = []) {
    self.content = content
    self.reasoningContent = reasoningContent
    self.toolCalls = toolCalls
  }

  enum CodingKeys: String, CodingKey {
    case content
    case reasoningContent = "reasoning_content"
    case toolCalls = "tool_calls"
  }

  init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    content = try values.decodeIfPresent(String.self, forKey: .content) ?? ""
    reasoningContent =
      try values.decodeIfPresent(String.self, forKey: .reasoningContent) ?? ""
    toolCalls = try values.decodeIfPresent([ChatToolCallDelta].self, forKey: .toolCalls) ?? []
  }
}

enum ChatStreamEvent: Equatable {
  case delta(ChatDelta)
  case usage(promptTokens: Int, completionTokens: Int)
  case done
}

enum ChatStreamDecoder {
  private struct Chunk: Decodable {
    struct Choice: Decodable {
      let delta: ChatDelta
    }

    struct Usage: Decodable {
      let promptTokens: Int
      let completionTokens: Int

      enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
      }
    }

    struct Failure: Decodable {
      let message: String
    }

    let choices: [Choice]?
    let usage: Usage?
    let error: Failure?
  }

  static func decode(line: String) throws -> ChatStreamEvent? {
    guard line.hasPrefix("data:") else { return nil }
    let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
    if payload == "[DONE]" { return .done }
    let chunk = try JSONDecoder().decode(Chunk.self, from: Data(payload.utf8))
    if let error = chunk.error { throw ChatError(error.message) }
    if let usage = chunk.usage {
      return .usage(
        promptTokens: usage.promptTokens,
        completionTokens: usage.completionTokens
      )
    }
    guard let delta = chunk.choices?.first?.delta else { return nil }
    return .delta(delta)
  }
}

struct ChatMetrics: Equatable, Sendable {
  let promptTokens: Int
  let completionTokens: Int
  let elapsedSeconds: Double
  let firstTokenSeconds: Double

  var tokensPerSecond: Double {
    let generationSeconds = elapsedSeconds - firstTokenSeconds
    return generationSeconds > 0 ? Double(completionTokens) / generationSeconds : 0
  }
}

enum ChatRequestStage: Equatable, Sendable {
  case uploading(completed: Int, total: Int)
  case preparing
  case generating

  func label(language: AppLanguage) -> String {
    switch self {
    case let .uploading(completed, total):
      return L10n.string("Uploading attachments: %lld / %lld", language: language, Int64(completed), Int64(total))
    case .preparing: return L10n.string("Preparing response…", language: language)
    case .generating: return L10n.string("Generating", language: language)
    }
  }
}

enum ChatClient {
  private struct Tool: Encodable {
    struct Function: Encodable {
      struct Parameters: Encodable {
        struct Property: Encodable {
          let type = "string"
          let description = "IANA time zone, such as Asia/Taipei."
        }

        let type = "object"
        let properties = ["time_zone": Property()]
        let required = ["time_zone"]
        let additionalProperties = false

        enum CodingKeys: String, CodingKey {
          case type, properties, required
          case additionalProperties = "additionalProperties"
        }
      }

      let name = "get_current_time"
      let description = "Get the current date and time for a time zone."
      let parameters = Parameters()
    }

    let type = "function"
    let function = Function()
  }

  private struct Payload: Encodable {
    struct StreamOptions: Encodable {
      let includeUsage = true

      enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
      }
    }

    let model: String
    let messages: [ChatMessage]
    let thinkingMode: String
    let seed: UInt32?
    let tools: [Tool]?
    let stream = true
    let streamOptions = StreamOptions()

    enum CodingKeys: String, CodingKey {
      case model, messages, stream, seed
      case thinkingMode = "thinking_mode"
      case streamOptions = "stream_options"
      case tools
    }
  }

  private struct ErrorResponse: Decodable {
    struct Detail: Decodable {
      let message: String
    }
    let error: Detail
  }

  static func parseSeed(_ text: String, language: AppLanguage) throws -> UInt32? {
    let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty else { return nil }
    guard value.utf8.allSatisfy({ (48...57).contains($0) }), let seed = UInt32(value) else {
      throw ChatError(L10n.string(
        "Enter a whole number from 0 to 4294967295, or leave blank for automatic.",
        language: language))
    }
    return seed
  }

  static func makeRequest(
    messages: [ChatMessage],
    baseURL: URL,
    apiKey: String,
    model: String,
    thinkingMode: String,
    seed: UInt32? = nil,
    enableTestTool: Bool
  ) throws -> URLRequest {
    var request = URLRequest(url: baseURL.appending(path: "v1/chat/completions"))
    request.httpMethod = "POST"
    request.timeoutInterval = 3_600
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if !apiKey.isEmpty {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONEncoder().encode(
      Payload(
        model: model,
        messages: messages,
        thinkingMode: thinkingMode,
        seed: seed,
        tools: enableTestTool ? [Tool()] : nil
      ))
    return request
  }

  static func stream(
    messages: [ChatMessage],
    baseURL: URL,
    apiKey: String,
    model: String,
    thinkingMode: String,
    seed: UInt32? = nil,
    enableTestTool: Bool,
    progress: @MainActor @escaping (ChatRequestStage) -> Void = { _ in },
    receive: @MainActor @escaping (ChatDelta) -> Void
  ) async throws -> ChatMetrics {
    let prepared = try await uploadAttachments(messages, baseURL: baseURL, apiKey: apiKey, progress: progress)
    defer {
      let files = prepared.flatMap(\.uploadedFileIDs)
      Task { await deleteAssets(files, baseURL: baseURL, apiKey: apiKey) }
    }
    try Task.checkCancellation()
    await progress(.preparing)
    let request = try makeRequest(
      messages: prepared, baseURL: baseURL, apiKey: apiKey, model: model,
      thinkingMode: thinkingMode, seed: seed, enableTestTool: enableTestTool)
    let clock = ContinuousClock()
    let start = clock.now
    let (bytes, response) = try await URLSession.shared.bytes(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw ChatError(L10n.string("The server did not return an HTTP response."))
    }
    guard (200..<300).contains(http.statusCode) else {
      var data = Data()
      for try await byte in bytes { data.append(byte) }
      let detail = try? JSONDecoder().decode(ErrorResponse.self, from: data)
      throw ChatError(
        detail?.error.message
          ?? L10n.string("The server returned HTTP %lld.", Int64(http.statusCode)))
    }

    var promptTokens = 0
    var completionTokens = 0
    var firstDelta: ContinuousClock.Instant?
    for try await line in bytes.lines {
      guard let event = try ChatStreamDecoder.decode(line: line) else { continue }
      switch event {
      case .delta(let delta):
        if !delta.content.isEmpty || !delta.reasoningContent.isEmpty || !delta.toolCalls.isEmpty {
          if firstDelta == nil { await progress(.generating) }
          firstDelta = firstDelta ?? clock.now
          await receive(delta)
        }
      case .usage(let prompt, let completion):
        promptTokens = prompt
        completionTokens = completion
      case .done:
        let end = clock.now
        return ChatMetrics(
          promptTokens: promptTokens,
          completionTokens: completionTokens,
          elapsedSeconds: seconds(from: start.duration(to: end)),
          firstTokenSeconds: seconds(from: start.duration(to: firstDelta ?? end))
        )
      }
    }
    throw ChatError(L10n.string("The streaming response ended before [DONE]."))
  }

  static func uploadAttachments(_ messages: [ChatMessage], baseURL: URL, apiKey: String,
    progress: @MainActor @escaping (ChatRequestStage) -> Void = { _ in }
  ) async throws -> [ChatMessage] {
    let attachments = messages.flatMap(\.attachments)
    guard !attachments.isEmpty else { return messages }
    let total = Set(attachments.map(\.id)).count
    await progress(.uploading(completed: 0, total: total))
    func request(_ path: String) -> URLRequest {
      var request = URLRequest(url: baseURL.appending(path: path))
      request.timeoutInterval = 60
      if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
      return request
    }
    let (capData, capResponse) = try await URLSession.shared.data(for: request("api/capabilities"))
    guard (capResponse as? HTTPURLResponse)?.statusCode == 200,
      let capabilities = try JSONSerialization.jsonObject(with: capData) as? [String: Any],
      let limits = capabilities["media"] as? [String: Any],
      let maxImages = limits["max_images"] as? Int,
      let maxBytes = limits["max_request_bytes"] as? Int,
      let maxAsset = limits["max_asset_bytes"] as? Int,
      attachments.filter(\.isImage).count <= maxImages,
      attachments.filter(\.isDocument).count <= (limits["max_documents"] as? Int ?? 0),
      attachments.filter(\.isAudio).count <= (limits["max_audios"] as? Int ?? 0) else {
      throw AttachmentError(L10n.string("Attachments exceed the server limits. Remove pending files or start a new chat."))
    }
    let types = (limits["image_mime_types"] as? [String] ?? []) + (limits["document_mime_types"] as? [String] ?? [])
      + (limits["audio_mime_types"] as? [String] ?? [])
    guard attachments.allSatisfy({ types.contains($0.mime) }) else {
      throw AttachmentError(L10n.string("The server does not support this attachment type."))
    }
    var prepared = messages
    var uploaded: [UUID: String] = [:]
    var complete = false
    defer {
      if !complete {
        let files = Array(uploaded.values)
        Task { await deleteAssets(files, baseURL: baseURL, apiKey: apiKey) }
      }
    }
    var bytes = 0
    for attachment in attachments {
      try Task.checkCancellation()
      let data = try attachment.data()
      bytes += data.count
      guard bytes <= maxBytes, data.count <= maxAsset else {
        throw AttachmentError(L10n.string("Attachments exceed the server limits. Remove pending files or start a new chat."))
      }
      if uploaded[attachment.id] != nil { continue }
      var upload = request("api/assets")
      upload.httpMethod = "POST"
      upload.setValue(attachment.mime, forHTTPHeaderField: "Content-Type")
      upload.httpBody = data
      let (body, response) = try await URLSession.shared.data(for: upload)
      guard (response as? HTTPURLResponse)?.statusCode == 201,
        let file = try JSONSerialization.jsonObject(with: body) as? [String: Any], let id = file["id"] as? String else {
        let detail = try? JSONDecoder().decode(ErrorResponse.self, from: body)
        throw AttachmentError(detail?.error.message ?? L10n.string("The attachment upload failed. Try again."))
      }
      guard id.range(of: "^file-[0-9a-f]{48}$", options: .regularExpression) != nil else {
        throw AttachmentError(L10n.string("The attachment upload failed. Try again."))
      }
      uploaded[attachment.id] = id
      guard file["sha256"] as? String == attachment.sha256, file["bytes"] as? Int == data.count else {
        throw AttachmentError(L10n.string("The attachment upload failed. Try again."))
      }
      await progress(.uploading(completed: uploaded.count, total: total))
    }
    for i in prepared.indices {
      prepared[i].uploadedFileIDs = prepared[i].attachments.compactMap { uploaded[$0.id] }
    }
    complete = true
    return prepared
  }

  private static func deleteAssets(_ files: [String], baseURL: URL, apiKey: String) async {
    for id in Set(files) {
      guard id.range(of: "^file-[0-9a-f]{48}$", options: .regularExpression) != nil else { continue }
      var request = URLRequest(url: baseURL.appending(path: "api/assets/\(id)"))
      request.httpMethod = "DELETE"
      request.timeoutInterval = 10
      if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
      _ = try? await URLSession.shared.data(for: request)
    }
  }

  private static func seconds(from duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }
}

private struct ChatError: LocalizedError {
  let message: String

  init(_ message: String) {
    self.message = message
  }

  var errorDescription: String? { message }
}
