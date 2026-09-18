import CryptoKit
import Foundation

struct Repacker {
  private let source: any CheckpointSource
  private let chunkSize: UInt64 = 8 * 1_024 * 1_024
  private let conversionBatchSize: UInt64 = 64 * 1_024 * 1_024
  private let mergeGap: UInt64 = 64 * 1_024
  private let receiptBatchSize = 256
  private let downloadConcurrency = 8
  private let conversionConcurrency = 2

  init(source: any CheckpointSource) {
    self.source = source
  }

  func run(
    plan: RepackPlan,
    output: URL,
    progress: (@Sendable (RepackProgress) -> Void)?,
    companions requestedCompanions: [CompanionFile]? = nil
  ) async throws -> InstalledManifest {
    let fileManager = FileManager.default
    let output = output.standardizedFileURL
    let partial = output.appendingPathExtension("partial")
    let receiptURL = partial.appendingPathComponent("repack-receipt.json")
    guard !fileManager.fileExists(atPath: output.path) else {
      throw RepackError.destinationExists(output.path)
    }

    let parent = output.deletingLastPathComponent()
    try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
    if !fileManager.fileExists(atPath: partial.path),
      let available = try parent.resourceValues(
        forKeys: [.volumeAvailableCapacityForImportantUsageKey]
      ).volumeAvailableCapacityForImportantUsage,
      available >= 0,
      UInt64(available) < plan.installedBytes
    {
      throw RepackError.insufficientStorage(required: plan.installedBytes, available: available)
    }

    var handles: [String: FileHandle] = [:]
    var receipt: RepackReceipt?
    var dirtyFiles = Set<String>()
    do {
      let prepared = try prepare(plan: plan, partial: partial, receiptURL: receiptURL)
      handles = prepared.handles
      receipt = prepared.receipt

      var copied: UInt64 = 0
      var downloadedBytes: UInt64 = 0
      var pendingReceiptCount = 0
      var pending: [WorkChunk] = []
      for span in makeSpans(plan.copies) {
        var position = span.start
        while position < span.end {
          try Task.checkCancellation()
          let end = min(position + chunkSize, span.end)
          let id = chunkID(sourceFile: span.sourceFile, range: position..<end)
          let payloadBytes = payloadByteCount(span: span, range: position..<end)
          let completedChunkIsValid =
            try receipt?.completed[id].map { expectedDigest in
              try autoreleasepool {
                try destinationDigest(span: span, range: position..<end, handles: handles)
                  == expectedDigest
              }
            } ?? false
          if completedChunkIsValid {
            copied += payloadBytes
            position = end
            progress?(
              RepackProgress(
                copiedBytes: copied,
                downloadedBytes: downloadedBytes,
                totalBytes: plan.checkpointTensorBytes
              ))
          } else {
            pending.append(WorkChunk(span: span, range: position..<end, id: id))
          }
          position = end
        }
      }
      try await withThrowingTaskGroup(of: DownloadedChunk.self) { group in
        var iterator = pending.makeIterator()
        for _ in 0..<min(downloadConcurrency, pending.count) {
          guard let work = iterator.next() else { break }
          group.addTask { DownloadedChunk(work: work, data: try await read(work)) }
        }
        while let downloaded = try await group.next() {
          let work = downloaded.work
          let written = try write(
            data: downloaded.data,
            span: work.span,
            range: work.range,
            handles: handles,
            dirtyFiles: &dirtyFiles
          )
          copied += written.bytes
          downloadedBytes += written.bytes
          receipt?.completed[work.id] = written.digest
          pendingReceiptCount += 1
          progress?(
            RepackProgress(
              copiedBytes: copied,
              downloadedBytes: downloadedBytes,
              totalBytes: plan.checkpointTensorBytes
            ))
          if pendingReceiptCount == receiptBatchSize, let receipt {
            try persist(
              receipt: receipt,
              handles: handles,
              dirtyFiles: &dirtyFiles,
              receiptURL: receiptURL
            )
            pendingReceiptCount = 0
          }
          if let next = iterator.next() {
            group.addTask { DownloadedChunk(work: next, data: try await read(next)) }
          }
        }
      }
      var pendingConversions: [ConversionWork] = []
      for conversion in plan.expertConversions ?? [] {
        let layout = try conversionLayout(conversion, plan: plan)
        try Task.checkCancellation()
        let range = conversionSourceRange(conversion, layout: layout)
        let scaleRange = conversionScaleRange(conversion, layout: layout)
        let id = try conversionChunkID(conversion: conversion, plan: plan)
        let completedChunkIsValid = try receipt?.completed[id].map { expectedDigest in
          try conversionDestinationDigest(
            conversion: conversion, layout: layout, handles: handles) == expectedDigest
        } ?? false
        let sourceByteCount = range.upperBound - range.lowerBound
          + scaleRange.upperBound - scaleRange.lowerBound
        if completedChunkIsValid {
          copied += sourceByteCount
        } else {
          pendingConversions.append(
            ConversionWork(
              conversion: conversion, layout: layout, range: range,
              scaleRange: scaleRange, id: id, sourceByteCount: sourceByteCount))
        }
        progress?(
          RepackProgress(
            copiedBytes: copied,
            downloadedBytes: downloadedBytes,
            totalBytes: plan.checkpointTensorBytes))
      }
      let scaleBundles = try await readConversionScales(pendingConversions)
      let conversionBatches = makeConversionBatches(pendingConversions)
      try await withThrowingTaskGroup(of: ConvertedBatch.self) { group in
        var iterator = conversionBatches.makeIterator()
        for _ in 0..<min(conversionConcurrency, conversionBatches.count) {
          guard let batch = iterator.next() else { break }
          group.addTask { try await convert(batch, scaleBundles: scaleBundles) }
        }
        while let convertedBatch = try await group.next() {
          for converted in convertedBatch.chunks {
            let digest = try writeConversion(
              (converted.weights, converted.scales),
              conversion: converted.work.conversion,
              layout: converted.work.layout,
              handles: handles,
              dirtyFiles: &dirtyFiles)
            copied += converted.work.sourceByteCount
            downloadedBytes += converted.work.sourceByteCount
            receipt?.completed[converted.work.id] = digest
            pendingReceiptCount += 1
            if pendingReceiptCount >= receiptBatchSize, let receipt {
              try persist(
                receipt: receipt,
                handles: handles,
                dirtyFiles: &dirtyFiles,
                receiptURL: receiptURL)
              pendingReceiptCount = 0
            }
            progress?(
              RepackProgress(
                copiedBytes: copied,
                downloadedBytes: downloadedBytes,
                totalBytes: plan.checkpointTensorBytes))
          }
          if let next = iterator.next() {
            group.addTask { try await convert(next, scaleBundles: scaleBundles) }
          }
        }
      }
      guard copied == plan.checkpointTensorBytes else {
        throw RepackError.invalidPlan(
          "copied \(copied) bytes; expected \(plan.checkpointTensorBytes)"
        )
      }
      if let receipt {
        try persist(
          receipt: receipt,
          handles: handles,
          dirtyFiles: &dirtyFiles,
          receiptURL: receiptURL
        )
      }
      for handle in handles.values { try handle.close() }
      handles.removeAll()

      var installedFiles = try plan.files.map { file -> InstalledFile in
        let url = try safeFileURL(root: partial, path: file.path)
        let size = try fileSize(url)
        guard size == file.size else {
          throw RepackError.invalidPlan("installed size mismatch for \(file.path)")
        }
        return InstalledFile(path: file.path, size: size, sha256: try sha256(url))
      }
      var companions = requestedCompanions ?? {
        ModelPackages.companions(for: plan.modelKind ?? .deepSeekV4)
      }()
      if plan.modelKind == .deepSeekV41 && plan.dspark != nil {
        companions.append(CompanionFile(source: "inference/config.json", destination: "inference/config.json"))
      }
      for companion in companions {
        let data = try await read(path: companion.source)
        let url = try safeFileURL(root: partial, path: companion.destination)
        try fileManager.createDirectory(
          at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        installedFiles.append(
          InstalledFile(
            path: companion.destination,
            size: UInt64(data.count),
            sha256: sha256(data)
          ))
      }
      let manifest = InstalledManifest(
        formatVersion: plan.formatVersion,
        modelID: plan.modelID,
        revision: plan.revision,
        layerCount: plan.layerCount,
        expertCount: plan.expertCount,
        selectedExpertCount: plan.selectedExpertCount,
        expertBlobSize: plan.expertBlobSize,
        files: installedFiles,
        commonTensors: plan.commonTensors,
        expertRegions: plan.expertRegions,
        dspark: plan.dspark,
        mtp: nil,
        modelKind: plan.modelKind,
        maximumContext: plan.maximumContext,
        expertQuantization: plan.expertQuantization,
        ngram: plan.ngram,
        engram: plan.engram
      )
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(manifest).write(
        to: partial.appendingPathComponent("manifest.json"),
        options: .atomic
      )
      try fileManager.removeItem(at: receiptURL)
      try fileManager.moveItem(at: partial, to: output)
      return manifest
    } catch {
      if let receipt {
        try? persist(
          receipt: receipt,
          handles: handles,
          dirtyFiles: &dirtyFiles,
          receiptURL: receiptURL
        )
      }
      for handle in handles.values { try? handle.close() }
      throw error
    }
  }

  func repair(
    plan: RepackPlan,
    output: URL,
    invalidFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)?
  ) async throws -> InstalledManifest {
    let fileManager = FileManager.default
    let output = output.standardizedFileURL
    let partial = output.appendingPathExtension("partial")
    let receiptURL = partial.appendingPathComponent("repack-receipt.json")
    guard fileManager.fileExists(atPath: output.path) else {
      throw RepackError.invalidPlan("installed model is missing: \(output.path)")
    }
    guard !fileManager.fileExists(atPath: partial.path) else {
      throw RepackError.invalidPlan("partial repair already exists: \(partial.path)")
    }

    try fileManager.moveItem(at: output, to: partial)
    var handles: [String: FileHandle] = [:]
    var receipt: RepackReceipt?
    var dirtyFiles = Set<String>()
    do {
      let prepared = try prepare(plan: plan, partial: partial, receiptURL: receiptURL)
      handles = prepared.handles
      receipt = prepared.receipt
      for file in plan.files where invalidFiles.contains(file.path) {
        guard let handle = handles[file.path] else {
          throw RepackError.invalidPlan("missing repair destination \(file.path)")
        }
        try handle.truncate(atOffset: 0)
        try handle.truncate(atOffset: file.size)
        dirtyFiles.insert(file.path)
      }

      var preparedCount = 0
      for span in makeSpans(plan.copies) {
        var position = span.start
        while position < span.end {
          try Task.checkCancellation()
          let end = min(position + chunkSize, span.end)
          let range = position..<end
          let touchesInvalidFile = span.copies.contains { copy in
            invalidFiles.contains(copy.destinationFile)
              && max(range.lowerBound, copy.sourceOffset)
                < min(range.upperBound, copy.sourceOffset + copy.length)
          }
          if !touchesInvalidFile {
            let id = chunkID(sourceFile: span.sourceFile, range: range)
            receipt?.completed[id] = try autoreleasepool {
              try destinationDigest(span: span, range: range, handles: handles)
            }
            preparedCount += 1
            if preparedCount == receiptBatchSize, let receipt {
              try persist(
                receipt: receipt,
                handles: handles,
                dirtyFiles: &dirtyFiles,
                receiptURL: receiptURL
              )
              preparedCount = 0
            }
          }
          position = end
        }
      }
      for conversion in plan.expertConversions ?? []
      where !invalidFiles.contains(conversion.destinationFile)
      {
        let layout = try conversionLayout(conversion, plan: plan)
        let id = try conversionChunkID(conversion: conversion, plan: plan)
        receipt?.completed[id] = try conversionDestinationDigest(
          conversion: conversion, layout: layout, handles: handles)
        preparedCount += 1
        if preparedCount == receiptBatchSize, let receipt {
          try persist(
            receipt: receipt,
            handles: handles,
            dirtyFiles: &dirtyFiles,
            receiptURL: receiptURL)
          preparedCount = 0
        }
      }
      if let receipt {
        try persist(
          receipt: receipt,
          handles: handles,
          dirtyFiles: &dirtyFiles,
          receiptURL: receiptURL
        )
      }
      for handle in handles.values { try handle.close() }
      handles.removeAll()
    } catch {
      if let receipt {
        try? persist(
          receipt: receipt,
          handles: handles,
          dirtyFiles: &dirtyFiles,
          receiptURL: receiptURL
        )
      }
      for handle in handles.values { try? handle.close() }
      throw error
    }
    return try await run(plan: plan, output: output, progress: progress)
  }

  private struct RepackReceipt: Codable {
    let formatVersion: Int
    let modelID: String
    let revision: String
    let planDigest: String
    var completed: [String: String]
  }

  private func prepare(
    plan: RepackPlan,
    partial: URL,
    receiptURL: URL
  ) throws -> (handles: [String: FileHandle], receipt: RepackReceipt) {
    let fileManager = FileManager.default
    let digest = try planDigest(plan)
    let exists = fileManager.fileExists(atPath: partial.path)
    if exists {
      let receipt =
        if fileManager.fileExists(atPath: receiptURL.path) {
          try JSONDecoder().decode(RepackReceipt.self, from: Data(contentsOf: receiptURL))
        } else {
          RepackReceipt(
            formatVersion: 1,
            modelID: plan.modelID,
            revision: plan.revision,
            planDigest: digest,
            completed: [:]
          )
        }
      guard receipt.formatVersion == 1,
        receipt.modelID == plan.modelID,
        receipt.revision == plan.revision,
        receipt.planDigest == digest
      else {
        throw RepackError.invalidPlan("partial install receipt does not match the repack plan")
      }
      var handles: [String: FileHandle] = [:]
      for file in plan.files {
        let url = try safeFileURL(root: partial, path: file.path)
        try fileManager.createDirectory(
          at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fileManager.fileExists(atPath: url.path) {
          guard fileManager.createFile(atPath: url.path, contents: nil) else {
            throw RepackError.invalidPlan("cannot create \(file.path)")
          }
        }
        let handle = try FileHandle(forUpdating: url)
        if try fileSize(url) != file.size {
          try handle.truncate(atOffset: file.size)
        }
        handles[file.path] = handle
      }
      return (handles, receipt)
    }

    try fileManager.createDirectory(at: partial, withIntermediateDirectories: false)
    var handles: [String: FileHandle] = [:]
    for file in plan.files {
      let url = try safeFileURL(root: partial, path: file.path)
      try fileManager.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      guard fileManager.createFile(atPath: url.path, contents: nil) else {
        throw RepackError.invalidPlan("cannot create \(file.path)")
      }
      let handle = try FileHandle(forUpdating: url)
      try handle.truncate(atOffset: file.size)
      handles[file.path] = handle
    }
    let receipt = RepackReceipt(
      formatVersion: 1,
      modelID: plan.modelID,
      revision: plan.revision,
      planDigest: digest,
      completed: [:]
    )
    var dirtyFiles = Set<String>()
    try persist(
      receipt: receipt,
      handles: handles,
      dirtyFiles: &dirtyFiles,
      receiptURL: receiptURL
    )
    return (handles, receipt)
  }

  private func persist(
    receipt: RepackReceipt,
    handles: [String: FileHandle],
    dirtyFiles: inout Set<String>,
    receiptURL: URL
  ) throws {
    for path in dirtyFiles.sorted() {
      try handles[path]?.synchronize()
    }
    dirtyFiles.removeAll(keepingCapacity: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    try encoder.encode(receipt).write(to: receiptURL, options: .atomic)
  }

  private func planDigest(_ plan: RepackPlan) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return sha256(try encoder.encode(plan))
  }

  private func chunkID(sourceFile: String, range: Range<UInt64>) -> String {
    "\(sourceFile):\(range.lowerBound)-\(range.upperBound)"
  }

  private struct ConversionLayout: Sendable {
    let rows: Int
    let columns: Int
    let scaleRows: Int
    let scaleColumns: Int
    let weight: ExpertRegion
    let scale: ExpertRegion
  }

  private struct ConversionWork: Sendable {
    let conversion: ExpertConversion
    let layout: ConversionLayout
    let range: Range<UInt64>
    let scaleRange: Range<UInt64>
    let id: String
    let sourceByteCount: UInt64
  }

  private struct ConvertedChunk: Sendable {
    let work: ConversionWork
    let weights: Data
    let scales: Data
  }

  private struct ConversionBatch: Sendable {
    let sourceFile: String
    let range: Range<UInt64>
    let works: [ConversionWork]
  }

  private struct ConvertedBatch: Sendable {
    let chunks: [ConvertedChunk]
  }

  private struct ScaleBundle: Sendable {
    let range: Range<UInt64>
    let data: Data
  }

  private func makeConversionBatches(_ works: [ConversionWork]) -> [ConversionBatch] {
    let sorted = works.sorted {
      ($0.conversion.sourceFile, $0.range.lowerBound)
        < ($1.conversion.sourceFile, $1.range.lowerBound)
    }
    var batches: [ConversionBatch] = []
    var current: [ConversionWork] = []
    var sourceFile = ""
    var range: Range<UInt64> = 0..<0
    for work in sorted {
      let canAppend = !current.isEmpty
        && work.conversion.sourceFile == sourceFile
        && work.range.lowerBound == range.upperBound
        && work.range.upperBound - range.lowerBound <= conversionBatchSize
      if canAppend {
        current.append(work)
        range = range.lowerBound..<work.range.upperBound
        continue
      }
      if !current.isEmpty {
        batches.append(ConversionBatch(sourceFile: sourceFile, range: range, works: current))
      }
      sourceFile = work.conversion.sourceFile
      range = work.range
      current = [work]
    }
    if !current.isEmpty {
      batches.append(ConversionBatch(sourceFile: sourceFile, range: range, works: current))
    }
    return batches
  }

  private func readConversionScales(_ works: [ConversionWork]) async throws
    -> [String: ScaleBundle]
  {
    var ranges: [String: Range<UInt64>] = [:]
    for work in works {
      let file = work.conversion.sourceScaleFile
      if let range = ranges[file] {
        let lowerBound = min(range.lowerBound, work.scaleRange.lowerBound)
        let upperBound = max(range.upperBound, work.scaleRange.upperBound)
        ranges[file] = lowerBound..<upperBound
      } else {
        ranges[file] = work.scaleRange
      }
    }
    let requests = ranges.sorted { $0.key < $1.key }
    return try await withThrowingTaskGroup(
      of: (String, Range<UInt64>, Data).self,
      returning: [String: ScaleBundle].self
    ) { group in
      var iterator = requests.makeIterator()
      for _ in 0..<min(conversionConcurrency, requests.count) {
        guard let request = iterator.next() else { break }
        group.addTask {
          (request.key, request.value, try await read(path: request.key, range: request.value))
        }
      }
      var result: [String: ScaleBundle] = [:]
      while let (file, range, data) = try await group.next() {
        result[file] = ScaleBundle(range: range, data: data)
        if let request = iterator.next() {
          group.addTask {
            (request.key, request.value, try await read(path: request.key, range: request.value))
          }
        }
      }
      return result
    }
  }

  private func convert(
    _ batch: ConversionBatch,
    scaleBundles: [String: ScaleBundle]
  ) async throws -> ConvertedBatch {
    let sourceData = try await read(path: batch.sourceFile, range: batch.range)
    var chunks: [ConvertedChunk] = []
    chunks.reserveCapacity(batch.works.count)
    for work in batch.works {
      try Task.checkCancellation()
      guard let scaleBundle = scaleBundles[work.conversion.sourceScaleFile],
        batch.range.lowerBound <= work.range.lowerBound,
        work.range.upperBound <= batch.range.upperBound,
        scaleBundle.range.lowerBound <= work.scaleRange.lowerBound,
        work.scaleRange.upperBound <= scaleBundle.range.upperBound
      else {
        throw RepackError.invalidPlan("MXFP4 source range is outside its download batch")
      }
      let sourceStart = Int(work.range.lowerBound - batch.range.lowerBound)
      let sourceEnd = sourceStart + Int(work.range.count)
      let scaleStart = Int(work.scaleRange.lowerBound - scaleBundle.range.lowerBound)
      let scaleEnd = scaleStart + Int(work.scaleRange.count)
      let quantized = try MXFP4.quantizeFP8(
        Data(sourceData[sourceStart..<sourceEnd]),
        inverseScales: Data(scaleBundle.data[scaleStart..<scaleEnd]),
        rows: work.layout.rows,
        columns: work.layout.columns)
      chunks.append(
        ConvertedChunk(work: work, weights: quantized.weights, scales: quantized.scales))
    }
    return ConvertedBatch(chunks: chunks)
  }

  private func conversionLayout(_ conversion: ExpertConversion, plan: RepackPlan) throws
    -> ConversionLayout
  {
    guard conversion.sourceDType == "F8_E4M3", conversion.sourceShape.count == 2,
      conversion.sourceScaleDType == "BF16", conversion.sourceScaleShape.count == 2,
      plan.expertQuantization?.conversionVersion != nil,
      let weight = plan.expertRegions.first(where: { $0.name == conversion.weightRegion }),
      let scale = plan.expertRegions.first(where: { $0.name == conversion.scaleRegion })
    else {
      throw RepackError.invalidPlan("invalid MXFP4 conversion for \(conversion.tensor)")
    }
    let rows = conversion.sourceShape[0]
    let columns = conversion.sourceShape[1]
    let scaleRows = conversion.sourceScaleShape[0]
    let scaleColumns = conversion.sourceScaleShape[1]
    guard plan.expertQuantization?.mode == "mxfp4",
      plan.expertQuantization?.bits == 4,
      plan.expertQuantization?.groupSize == 32,
      conversion.expert >= 0, conversion.expert < plan.expertCount,
      conversion.destinationRow >= 0,
      rows.isMultiple(of: 128), columns.isMultiple(of: 128),
      scaleRows == rows / 128, scaleColumns == columns / 128,
      UInt64((conversion.destinationRow + rows) * columns / 2) <= weight.length,
      UInt64((conversion.destinationRow + rows) * columns / 32) <= scale.length
    else {
      throw RepackError.invalidPlan("MXFP4 conversion layout does not match the expert blob")
    }
    return ConversionLayout(
      rows: rows, columns: columns, scaleRows: scaleRows, scaleColumns: scaleColumns,
      weight: weight, scale: scale)
  }

  private func conversionSourceRange(
    _ conversion: ExpertConversion,
    layout: ConversionLayout
  ) -> Range<UInt64> {
    let bytes = UInt64(layout.rows * layout.columns)
    return conversion.sourceOffset..<(conversion.sourceOffset + bytes)
  }

  private func conversionScaleRange(
    _ conversion: ExpertConversion,
    layout: ConversionLayout
  ) -> Range<UInt64> {
    let bytes = UInt64(layout.scaleRows * layout.scaleColumns * 2)
    return conversion.sourceScaleOffset..<(conversion.sourceScaleOffset + bytes)
  }

  private func conversionChunkID(
    conversion: ExpertConversion,
    plan: RepackPlan
  ) throws -> String {
    guard let version = plan.expertQuantization?.conversionVersion else {
      throw RepackError.invalidPlan("missing expert quantization conversion version")
    }
    return "mxfp4-v\(version):\(conversion.tensor)"
  }

  private func writeConversion(
    _ quantized: (weights: Data, scales: Data),
    conversion: ExpertConversion,
    layout: ConversionLayout,
    handles: [String: FileHandle],
    dirtyFiles: inout Set<String>
  ) throws -> String {
    guard let handle = handles[conversion.destinationFile] else {
      throw RepackError.invalidPlan("missing destination \(conversion.destinationFile)")
    }
    let base = UInt64(conversion.expert) * QwenContract.expertBlobSize
    let weightOffset = UInt64(conversion.destinationRow * layout.columns / 2)
    let scaleOffset = UInt64(conversion.destinationRow * layout.columns / 32)
    try handle.seek(toOffset: base + layout.weight.offset + weightOffset)
    try handle.write(contentsOf: quantized.weights)
    try handle.seek(toOffset: base + layout.scale.offset + scaleOffset)
    try handle.write(contentsOf: quantized.scales)
    dirtyFiles.insert(conversion.destinationFile)
    var hasher = SHA256()
    hasher.update(data: quantized.weights)
    hasher.update(data: quantized.scales)
    return hex(hasher.finalize())
  }

  private func conversionDestinationDigest(
    conversion: ExpertConversion,
    layout: ConversionLayout,
    handles: [String: FileHandle]
  ) throws -> String {
    guard let handle = handles[conversion.destinationFile] else {
      throw RepackError.invalidPlan("missing destination \(conversion.destinationFile)")
    }
    let base = UInt64(conversion.expert) * QwenContract.expertBlobSize
    var hasher = SHA256()
    let slices = [
      (layout.weight, UInt64(conversion.destinationRow * layout.columns / 2),
        quantizedLength(rows: layout.rows, columns: layout.columns, divisor: 2)),
      (layout.scale, UInt64(conversion.destinationRow * layout.columns / 32),
        quantizedLength(rows: layout.rows, columns: layout.columns, divisor: 32)),
    ]
    for (region, offset, length) in slices {
      try handle.seek(toOffset: base + region.offset + offset)
      guard let data = try handle.read(upToCount: length), data.count == length
      else {
        throw RepackError.invalidPlan("cannot validate converted data for \(conversion.tensor)")
      }
      hasher.update(data: data)
    }
    return hex(hasher.finalize())
  }

  private func quantizedLength(rows: Int, columns: Int, divisor: Int) -> Int {
    rows * columns / divisor
  }

  private func payloadByteCount(span: SourceSpan, range: Range<UInt64>) -> UInt64 {
    span.copies.reduce(UInt64(0)) { result, copy in
      let start = max(range.lowerBound, copy.sourceOffset)
      let end = min(range.upperBound, copy.sourceOffset + copy.length)
      return result + (start < end ? end - start : 0)
    }
  }

  private func write(
    data: Data,
    span: SourceSpan,
    range: Range<UInt64>,
    handles: [String: FileHandle],
    dirtyFiles: inout Set<String>
  ) throws -> (bytes: UInt64, digest: String) {
    var hasher = SHA256()
    var bytes: UInt64 = 0
    for copy in span.copies {
      let copyEnd = copy.sourceOffset + copy.length
      let start = max(range.lowerBound, copy.sourceOffset)
      let end = min(range.upperBound, copyEnd)
      guard start < end else { continue }
      guard let handle = handles[copy.destinationFile] else {
        throw RepackError.invalidPlan("missing destination \(copy.destinationFile)")
      }
      let slice = Data(data[Int(start - range.lowerBound)..<Int(end - range.lowerBound)])
      try handle.seek(toOffset: copy.destinationOffset + start - copy.sourceOffset)
      try handle.write(contentsOf: slice)
      hasher.update(data: slice)
      dirtyFiles.insert(copy.destinationFile)
      bytes += end - start
    }
    return (bytes, hex(hasher.finalize()))
  }

  private func destinationDigest(
    span: SourceSpan,
    range: Range<UInt64>,
    handles: [String: FileHandle]
  ) throws -> String {
    var hasher = SHA256()
    for copy in span.copies {
      let copyEnd = copy.sourceOffset + copy.length
      let start = max(range.lowerBound, copy.sourceOffset)
      let end = min(range.upperBound, copyEnd)
      guard start < end else { continue }
      guard let handle = handles[copy.destinationFile] else {
        throw RepackError.invalidPlan("missing destination \(copy.destinationFile)")
      }
      let count = Int(end - start)
      try handle.seek(toOffset: copy.destinationOffset + start - copy.sourceOffset)
      guard let data = try handle.read(upToCount: count), data.count == count else {
        throw RepackError.invalidPlan("cannot validate partial data for \(copy.tensor)")
      }
      hasher.update(data: data)
    }
    return hex(hasher.finalize())
  }

  private func read(path: String, range: Range<UInt64>) async throws -> Data {
    var lastError: Error?
    for attempt in 0..<3 {
      do {
        return try await source.data(path: path, range: range)
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        lastError = error
        if attempt < 2 {
          try await Task.sleep(for: .milliseconds(500 * (attempt + 1)))
        }
      }
    }
    throw lastError ?? RepackError.badResponse("checkpoint range request failed")
  }

  private func read(path: String) async throws -> Data {
    var lastError: Error?
    for attempt in 0..<3 {
      do {
        return try await source.data(path: path)
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        lastError = error
        if attempt < 2 {
          try await Task.sleep(for: .milliseconds(500 * (attempt + 1)))
        }
      }
    }
    throw lastError ?? RepackError.badResponse("checkpoint request failed")
  }

  private struct SourceSpan: Sendable {
    let sourceFile: String
    var start: UInt64
    var end: UInt64
    var copies: [TensorCopy]
  }

  private struct WorkChunk: Sendable {
    let span: SourceSpan
    let range: Range<UInt64>
    let id: String
  }

  private struct DownloadedChunk: Sendable {
    let work: WorkChunk
    let data: Data
  }

  private func read(_ work: WorkChunk) async throws -> Data {
    try await read(path: work.span.sourceFile, range: work.range)
  }

  private func makeSpans(_ copies: [TensorCopy]) -> [SourceSpan] {
    var result: [SourceSpan] = []
    for (sourceFile, fileCopies) in Dictionary(grouping: copies, by: \.sourceFile).sorted(by: {
      $0.key < $1.key
    }) {
      for copy in fileCopies.sorted(by: { $0.sourceOffset < $1.sourceOffset }) {
        let copyEnd = copy.sourceOffset + copy.length
        if let lastIndex = result.indices.last,
          result[lastIndex].sourceFile == sourceFile,
          copy.sourceOffset <= result[lastIndex].end + mergeGap
        {
          result[lastIndex].end = max(result[lastIndex].end, copyEnd)
          result[lastIndex].copies.append(copy)
        } else {
          result.append(
            SourceSpan(
              sourceFile: sourceFile,
              start: copy.sourceOffset,
              end: copyEnd,
              copies: [copy]
            ))
        }
      }
    }
    return result
  }
}

public enum InstalledModel {
  public static func removeDSpark(at root: URL) throws -> InstalledManifest {
    let root = root.standardizedFileURL
    let current = try loadManifest(at: root)
    guard current.dspark != nil else { return current }
    let files = current.files.filter {
      !$0.path.hasPrefix("dspark/") && $0.path != "inference/config.json"
    }
    let manifest = InstalledManifest(
      formatVersion: current.formatVersion,
      modelID: current.modelID,
      revision: current.revision,
      layerCount: current.layerCount,
      expertCount: current.expertCount,
      selectedExpertCount: current.selectedExpertCount,
      expertBlobSize: current.expertBlobSize,
      files: files,
      commonTensors: current.commonTensors,
      expertRegions: current.expertRegions,
      mtp: current.mtp,
      modelKind: current.modelKind,
      maximumContext: current.maximumContext,
      expertQuantization: current.expertQuantization,
      ngram: current.ngram,
      engram: current.engram
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(manifest).write(
      to: root.appendingPathComponent("manifest.json"),
      options: .atomic
    )
    try? FileManager.default.removeItem(at: root.appendingPathComponent("dspark"))
    try? FileManager.default.removeItem(
      at: root.appendingPathComponent("inference/config.json"))
    return manifest
  }

  public static func loadManifest(at root: URL) throws -> InstalledManifest {
    let root = root.standardizedFileURL
    let data = try Data(contentsOf: root.appendingPathComponent("manifest.json"))
    return try decodeManifest(data)
  }

  static func decodeManifest(_ data: Data) throws -> InstalledManifest {
    let manifest = try JSONDecoder().decode(InstalledManifest.self, from: data)
    return try ModelPackages.package(for: manifest).validate(manifest)
  }

  public static func verify(at root: URL) throws -> InstalledManifest {
    let verification = try audit(at: root)
    if let issue = verification.issues.first {
      switch issue.kind {
      case .missing:
        throw RepackError.invalidPlan("installed file is missing: \(issue.path)")
      case .sizeMismatch:
        throw RepackError.invalidPlan("installed size mismatch for \(issue.path)")
      case .checksumMismatch:
        throw RepackError.invalidPlan("installed SHA-256 mismatch for \(issue.path)")
      }
    }
    return verification.manifest
  }

  public static func audit(
    at root: URL,
    progress: (@Sendable (VerificationProgress) -> Void)? = nil
  ) throws -> InstalledModelVerification {
    let root = root.standardizedFileURL
    return try audit(manifest: loadManifest(at: root), at: root, progress: progress)
  }

  static func audit(
    manifest: InstalledManifest,
    at root: URL,
    progress: (@Sendable (VerificationProgress) -> Void)? = nil
  ) throws -> InstalledModelVerification {
    let root = root.standardizedFileURL
    let totalBytes = try manifest.files.reduce(UInt64(0)) { total, file in
      let sum = total.addingReportingOverflow(file.size)
      guard !sum.overflow else {
        throw RepackError.invalidPlan("installed file sizes overflow")
      }
      return sum.partialValue
    }
    var checkedBytes: UInt64 = 0
    var issues: [InstalledFileIssue] = []
    progress?(VerificationProgress(checkedBytes: 0, totalBytes: totalBytes))
    for file in manifest.files.sorted(by: { $0.path < $1.path }) {
      try Task.checkCancellation()
      let url = try safeFileURL(root: root, path: file.path)
      guard let size = try? fileSize(url) else {
        issues.append(InstalledFileIssue(path: file.path, kind: .missing))
        checkedBytes += file.size
        progress?(VerificationProgress(checkedBytes: checkedBytes, totalBytes: totalBytes))
        continue
      }
      guard size == file.size else {
        issues.append(InstalledFileIssue(path: file.path, kind: .sizeMismatch))
        checkedBytes += file.size
        progress?(VerificationProgress(checkedBytes: checkedBytes, totalBytes: totalBytes))
        continue
      }
      let start = checkedBytes
      let digest = try sha256(url) { fileBytes in
        progress?(
          VerificationProgress(
            checkedBytes: start + fileBytes,
            totalBytes: totalBytes
          ))
      }
      if digest != file.sha256 {
        issues.append(InstalledFileIssue(path: file.path, kind: .checksumMismatch))
      }
      checkedBytes += file.size
      progress?(VerificationProgress(checkedBytes: checkedBytes, totalBytes: totalBytes))
    }
    return InstalledModelVerification(manifest: manifest, issues: issues)
  }
}

func safeFileURL(root: URL, path: String) throws -> URL {
  guard !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
    throw RepackError.invalidPlan("unsafe installed path \(path)")
  }
  let url = root.appendingPathComponent(path).standardizedFileURL
  guard url.path.hasPrefix(root.path + "/") else {
    throw RepackError.invalidPlan("installed path escapes the model directory: \(path)")
  }
  return url
}

func fileSize(_ url: URL) throws -> UInt64 {
  let values = try url.resourceValues(forKeys: [
    .fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey,
  ])
  guard values.isRegularFile == true, values.isSymbolicLink != true, let size = values.fileSize
  else {
    throw RepackError.invalidPlan("installed file is missing: \(url.path)")
  }
  return UInt64(size)
}

func sha256(_ url: URL, progress: ((UInt64) -> Void)? = nil) throws -> String {
  let handle = try FileHandle(forReadingFrom: url)
  defer { try? handle.close() }
  var hasher = SHA256()
  var processedBytes: UInt64 = 0
  while try autoreleasepool(invoking: {
    try Task.checkCancellation()
    guard let data = try handle.read(upToCount: 8 * 1_024 * 1_024), !data.isEmpty else {
      return false
    }
    hasher.update(data: data)
    processedBytes += UInt64(data.count)
    progress?(processedBytes)
    return true
  }) {}
  return hex(hasher.finalize())
}

func sha256(_ data: Data) -> String {
  hex(SHA256.hash(data: data))
}

func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
  digest.map { String(format: "%02x", $0) }.joined()
}
