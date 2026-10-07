import Foundation

/// Independent controls. Stored choices are not changed by disabling MTP.
enum QwenOptimization: String, CaseIterable, Identifiable {
  case pooledIndex, ngramLookup, compiledNorm, phaseMemory, mtpPolicy

  var id: String { rawValue }

  var keyPath: WritableKeyPath<ModelAdvancedSettings, Bool?> {
    switch self {
    case .pooledIndex: \.qwenPooledIndexCache
    case .ngramLookup: \.qwenNgramLookupOptimized
    case .compiledNorm: \.qwenCompileTensorOps
    case .phaseMemory: \.qwenPhaseMemory
    case .mtpPolicy: \.qwenMTPPolicy
    }
  }

  var title: String {
    switch self {
    case .pooledIndex: "Reuse QSA pooled keys"
    case .ngramLookup: "Optimize N-gram reads"
    case .compiledNorm: "Compile Qwen tensor operations"
    case .phaseMemory: "Adjust memory between prompt and answer"
    case .mtpPolicy: "Customize MTP strategy"
    }
  }

  var hint: String {
    switch self {
    case .pooledIndex:
      "Reuses completed attention-index blocks instead of recalculating them. Retains extra index data."
    case .ngramLookup:
      "Reads repeated N-gram rows once per lookup and uses a decoding table."
    case .compiledNorm:
      "Compiles grouped normalization and single-token hyper-connection operations. Multi-token batches keep the original mixing path. Rounding and generated text may change."
    case .phaseMemory:
      "Limits the expert cache to half its capacity while reading a prompt, keeping room for one layer when possible. Restores full capacity for answers. May cause extra SSD reads."
    case .mtpPolicy:
      "MTP output can differ slightly from non-MTP. Off uses 2 draft tokens and stops after 32 zero-acceptance rounds in a row."
    }
  }
}
