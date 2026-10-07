import DeepSeekRepack

/// UI names come from the same catalog as the Model page. Never use these
/// labels to select a model, send an API request, or identify benchmark data.
enum ModelDisplayName {
  static func resolve(_ name: String, models: [CatalogModel] = []) -> String {
    let modelID = models.first { $0.id == name }?.id
      ?? models.first { $0.alias == name }?.id
      ?? name
    return ModelPackages.descriptors.first { $0.apiModelID == modelID }?.displayName ?? name
  }
}

extension CatalogModel {
  var displayName: String { ModelDisplayName.resolve(id) }
}

extension ChatMessage {
  func modelDisplayName(models: [CatalogModel] = []) -> String? {
    modelName.map { ModelDisplayName.resolve($0, models: models) }
  }
}

extension LivePerformance {
  func modelDisplayLabel(language: AppLanguage) -> String {
    if let loadedModel { return ModelDisplayName.resolve(loadedModel) }
    if let loadingModel {
      return L10n.string("Loading %@", language: language, ModelDisplayName.resolve(loadingModel))
    }
    return L10n.string("No model loaded", language: language)
  }
}
