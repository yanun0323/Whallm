import Foundation
import DeepSeekRepack

enum AdvancedSettingImpact {
  static func key(for label: String, modelKind: ModelKind? = nil) -> String {
    if modelKind == .deepSeekV41 && ["Compress attention cache", "Compress attention index"].contains(label) {
      return "Save cache memory, but packing and unpacking may slow processing."
    }
    return switch label {
    case "Expert cache GiB", "DSpark expert cache GiB":
      "Retain more experts to reduce SSD reads, at the cost of memory. Recommended: about %@ GiB. Applies on next load."
    case "MTP expert cache GiB":
      "Retain more experts to reduce SSD reads, at the cost of memory. Default: about %@ GiB. Applies on next load."
    case "Use MTP", "Use DSpark":
      "May speed up generation, but uses more memory and can be slower for some requests."
    case "DSpark confidence threshold":
      "Reject weak drafts earlier, but fewer accepted drafts may reduce the speed gain. 0 keeps all drafts."
    case "Use approximate mode":
      "May speed up generation, but answer quality may drop. Unavailable with MTP or DSpark."
    case "Compress attention cache", "Compress attention index":
      "Save memory, but compression adds work and may slightly affect answer quality."
    case "Use BF16 KV cache":
      "Reduce cache quantization error, but may use more memory."
    case "Use ANE for prefill":
      "May shorten prefill, but needs extra memory and may slightly change output."
    case "ANE Prefill share":
      "More ANE work may shorten prefill, but adds memory use and may change output. Default and recommended: 0; 0 uses GPU only, 1 uses ANE only."
    case "Use layer-major prefill", "Batch expert calculations":
      "Shorten prefill, but peak memory may rise and output may differ slightly."
    case "Prefill acceleration":
      "Shorten prefill, but may use more memory. Requires layer-major prefill."
    case "Prefill step size":
      "Larger batches may shorten prefill, but raise peak memory. 0 automatically selects 128, 256, or 1024. Default: 0."
    case "MoE prefill step size":
      "Larger batches may shorten prefill, but raise peak memory. 0 selects automatically."
    case "Layer-major prefill threshold":
      "Avoid whole-layer overhead on shorter prompts, but fewer prompts benefit from faster prefill. Default: 1024."
    case "Read the next expert layer ahead", "Compute experts as they load":
      "Reduce SSD waiting, but may need more temporary memory."
    case "Read workers":
      "More workers may reduce SSD waiting, but use more memory; too many can slow things down. Recommended: 4."
    case "Prefetch read workers":
      "More workers may reduce SSD waiting, but use more memory; too many can slow things down. Default: 2."
    case "Prompt cache":
      "Off saves memory but repeats work. Memory reuses quickly but clears on unload. Disk survives restarts but adds I/O."
    case "Prompt cache entries":
      "Keep more conversation history to shorten repeated prefill, at the cost of memory. Recommended: %lld."
    case "Prompt cache GiB":
      "Keep more conversation history to shorten repeated prefill, at the cost of memory. Recommended: 8 GiB."
    case "Expert cache eviction":
      "LRU adapts quickly but forgets older use. LFU favors frequent use but adapts slowly. Route-aware balances layers but adds bookkeeping."
    case "Search candidate positions only":
      "Speed up attention search, but limiting candidates may affect output."
    case "Reduce decoder prefill work":
      "Shorten prefill and reduce temporary memory; full-model validation is still limited."
    case "Max tokens":
      "Allow longer answers, but generation may take more time and memory."
    case "Temperature", "Top P":
      "Increase answer variety, but responses may be less consistent."
    case "Top K":
      "Consider more candidates for varied answers, but responses may be less focused. 0 removes the limit."
    case "Use adaptive sampling":
      "Choose sampling for chat or thinking automatically, overriding the manual values below."
    case "Memory limit GiB", "MLX memory guideline GiB":
      "A higher guideline may reduce waiting, but allows more memory use. 0 is automatic; too little memory can fail requests."
    case "Warmup prompt":
      "Prepare matching prompts for faster reuse, but add startup time and cache memory."
    default: ""
    }
  }
}
